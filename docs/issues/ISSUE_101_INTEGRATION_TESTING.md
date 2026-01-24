# Issue #101: Integration Testing & Wikipedia Validation

## Summary
Comprehensive integration testing of the complete CSS layout pipeline, with specific validation against Wikipedia article rendering.

## Goals

1. End-to-end pipeline testing
2. Visual regression testing against browser reference
3. Performance benchmarking
4. Wikipedia-specific validation

## Test Categories

### 1. Pipeline Integration Tests

Test that all pipeline stages work together correctly.

```rust
#[test]
fn test_full_pipeline_simple() {
    let html = r#"
        <html>
        <head>
            <style>
                .container { width: 400px; margin: 20px; }
                .item { margin: 10px; padding: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="item">Item 1</div>
                <div class="item">Item 2</div>
            </div>
        </body>
        </html>
    "#;

    let result = DocumentPipeline::process(html);

    // Verify parsing
    assert!(result.element_count > 5);

    // Verify CSS application
    let container = find_element_by_class(&result, "container");
    assert_eq!(result.styles[container].width, 400.0);
    assert_eq!(result.styles[container].margin_left, 20.0);

    // Verify layout
    let items = find_elements_by_class(&result, "item");
    assert!(result.boxes[items[1]].y > result.boxes[items[0]].y);
}
```

### 2. CSS Cascade Integration Tests

```rust
#[test]
fn test_cascade_specificity_integration() {
    let html = r#"
        <html>
        <head>
            <style>
                div { color: red; }
                .blue { color: blue; }
                #green { color: green; }
            </style>
        </head>
        <body>
            <div id="green" class="blue">Text</div>
        </body>
        </html>
    "#;

    let result = DocumentPipeline::process(html);
    let div = find_element_by_id(&result, "green");

    // #green should win (specificity 100 > 10 > 1)
    assert_eq!(result.styles[div].color, rgba(0, 128, 0, 255));
}
```

### 3. External CSS Loading Integration Tests

```rust
#[tokio::test]
async fn test_external_css_loading() {
    // Serve test CSS from local server
    let server = TestServer::new();
    server.serve("/style.css", ".hidden { display: none; }");

    let html = format!(r#"
        <html>
        <head>
            <link rel="stylesheet" href="{}/style.css">
        </head>
        <body>
            <div class="hidden">Hidden</div>
            <div class="visible">Visible</div>
        </body>
        </html>
    "#, server.url());

    let result = DocumentPipeline::process_async(&html).await;

    let hidden = find_element_by_class(&result, "hidden");
    let visible = find_element_by_class(&result, "visible");

    assert_eq!(result.styles[hidden].display, DISPLAY_NONE);
    assert_eq!(result.styles[visible].display, DISPLAY_BLOCK);
}
```

### 4. Wikipedia-Specific Tests

```rust
#[tokio::test]
async fn test_wikipedia_rust_article() {
    let url = "https://en.wikipedia.org/wiki/Rust_(programming_language)";
    let result = DocumentPipeline::load_url(url).await.unwrap();

    // Basic sanity checks
    assert!(result.element_count > 1000, "Wikipedia has many elements");
    assert!(result.css_rule_count > 100, "Wikipedia CSS has many rules");

    // Layout sanity
    let root = &result.boxes[0];
    assert!(root.height > 0.0, "Root should have height");
    assert!(root.height < 100000.0, "Root height should be reasonable: {}", root.height);

    // Hidden elements should be hidden
    let hidden_elements = find_elements_by_class(&result, "mw-hidden");
    for elem_idx in &hidden_elements {
        assert_eq!(result.styles[*elem_idx].display, DISPLAY_NONE,
            "mw-hidden elements should have display:none");
    }

    // Sidebar should be hidden or positioned
    let sidebar = find_element_by_id(&result, "mw-panel");
    if let Some(idx) = sidebar {
        let style = &result.styles[idx];
        assert!(
            style.display == DISPLAY_NONE ||
            style.position == POSITION_ABSOLUTE ||
            style.position == POSITION_FIXED,
            "Sidebar should be hidden or positioned"
        );
    }
}

#[tokio::test]
async fn test_wikipedia_no_text_overlap() {
    let url = "https://en.wikipedia.org/wiki/Rust_(programming_language)";
    let result = DocumentPipeline::load_url(url).await.unwrap();

    // Collect all visible text elements
    let text_elements: Vec<_> = (0..result.element_count)
        .filter(|&i| {
            result.elements[i].element_type == ELEM_TEXT &&
            result.styles[i].display != DISPLAY_NONE &&
            result.boxes[i].height > 0.0
        })
        .collect();

    // Check for Y-position overlaps
    for i in 0..text_elements.len() {
        for j in i+1..text_elements.len() {
            let a = text_elements[i];
            let b = text_elements[j];

            let box_a = &result.boxes[a];
            let box_b = &result.boxes[b];

            // Check if boxes overlap vertically
            let y_overlap = !(box_a.y + box_a.height <= box_b.y ||
                             box_b.y + box_b.height <= box_a.y);

            // Check if boxes overlap horizontally
            let x_overlap = !(box_a.x + box_a.width <= box_b.x ||
                             box_b.x + box_b.width <= box_a.x);

            assert!(!(y_overlap && x_overlap),
                "Text elements {} and {} overlap at ({}, {}) and ({}, {})",
                a, b, box_a.x, box_a.y, box_b.x, box_b.y);
        }
    }
}

#[tokio::test]
async fn test_wikipedia_content_visible() {
    let url = "https://en.wikipedia.org/wiki/Rust_(programming_language)";
    let result = DocumentPipeline::load_url(url).await.unwrap();

    // Article title should be visible
    let h1_elements = find_elements_by_tag(&result, "h1");
    assert!(!h1_elements.is_empty(), "Should have h1 element");

    let title = h1_elements[0];
    assert_eq!(result.styles[title].display, DISPLAY_BLOCK);
    assert!(result.boxes[title].height > 0.0);

    // Article content should be visible
    let content = find_element_by_id(&result, "mw-content-text");
    if let Some(idx) = content {
        assert_eq!(result.styles[idx].display, DISPLAY_BLOCK);
        assert!(result.boxes[idx].height > 0.0);
    }
}
```

### 5. Visual Regression Tests

Compare rendered output against browser reference screenshots.

```rust
#[test]
fn test_visual_regression_simple_layout() {
    let html = include_str!("testdata/simple_layout.html");
    let result = DocumentPipeline::process(html);

    // Render to image
    let image = render_to_image(&result, 800, 600);

    // Compare against reference
    let reference = load_image("testdata/simple_layout_reference.png");
    let diff = compare_images(&image, &reference);

    assert!(diff < 0.01, "Visual difference {} exceeds threshold", diff);
}

// Generate reference images using headless Chrome
fn generate_reference_images() {
    let chrome = headless_chrome::Browser::new()?;
    let tab = chrome.new_tab()?;

    for test_file in glob("testdata/*.html") {
        tab.navigate_to(&format!("file://{}", test_file))?;
        let screenshot = tab.capture_screenshot()?;
        let ref_path = test_file.replace(".html", "_reference.png");
        std::fs::write(ref_path, screenshot)?;
    }
}
```

### 6. Performance Benchmarks

```rust
#[bench]
fn bench_wikipedia_full_pipeline(b: &mut Bencher) {
    let html = include_str!("testdata/wikipedia.html");
    let css = include_str!("testdata/wikipedia.css");

    b.iter(|| {
        let result = DocumentPipeline::process_with_css(html, css);
        black_box(result)
    });
}

#[bench]
fn bench_css_parsing(b: &mut Bencher) {
    let css = include_str!("testdata/wikipedia.css");

    b.iter(|| {
        let result = GpuCSSParser::parse(css);
        black_box(result)
    });
}

#[bench]
fn bench_selector_matching(b: &mut Bencher) {
    let html = include_str!("testdata/wikipedia.html");
    let css = include_str!("testdata/wikipedia.css");

    let elements = parse_html(html);
    let rules = parse_css(css);

    b.iter(|| {
        let result = match_selectors(&elements, &rules);
        black_box(result)
    });
}

#[bench]
fn bench_layout(b: &mut Bencher) {
    let html = include_str!("testdata/wikipedia.html");
    let css = include_str!("testdata/wikipedia.css");

    let elements = parse_html(html);
    let styles = compute_styles(&elements, css);

    b.iter(|| {
        let result = compute_layout(&elements, &styles);
        black_box(result)
    });
}
```

### 7. Edge Case Tests

```rust
#[test]
fn test_deeply_nested_elements() {
    // Create 100-level deep nesting
    let mut html = String::new();
    for _ in 0..100 {
        html.push_str("<div>");
    }
    html.push_str("Content");
    for _ in 0..100 {
        html.push_str("</div>");
    }

    let result = DocumentPipeline::process(&html);
    assert!(result.boxes[100].height > 0.0);
}

#[test]
fn test_many_siblings() {
    // 1000 siblings
    let mut html = "<div>".to_string();
    for i in 0..1000 {
        html.push_str(&format!("<span>{}</span>", i));
    }
    html.push_str("</div>");

    let result = DocumentPipeline::process(&html);
    assert_eq!(result.element_count, 1002);  // div + 1000 spans + text
}

#[test]
fn test_empty_elements() {
    let html = r#"
        <div></div>
        <div>   </div>
        <div><span></span></div>
    "#;

    let result = DocumentPipeline::process(html);

    // Empty elements should have zero content height
    for i in 0..result.element_count {
        if result.elements[i].first_child < 0 {
            // No children = truly empty
            assert!(result.boxes[i].content_height <= 0.1);
        }
    }
}

#[test]
fn test_invalid_css() {
    let html = r#"
        <style>
            .broken { color: ; }
            .incomplete { margin:
            .valid { color: red; }
        </style>
        <div class="broken valid">Text</div>
    "#;

    // Should not crash, and valid rules should still apply
    let result = DocumentPipeline::process(html);
    let div = find_element_by_class(&result, "valid");
    assert_eq!(result.styles[div].color, rgba(255, 0, 0, 255));
}
```

## Test Data Files

Create test fixtures:

```
testdata/
├── simple_layout.html          # Basic block layout
├── simple_layout_reference.png # Browser reference screenshot
├── margin_collapsing.html      # Margin collapsing scenarios
├── inline_layout.html          # Inline formatting context
├── complex_selectors.html      # Various CSS selectors
├── wikipedia.html              # Wikipedia article snapshot
├── wikipedia.css               # Wikipedia CSS snapshot
└── stress_test.html            # Large document for performance
```

## Continuous Integration

```yaml
# .github/workflows/test.yml
name: Layout Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest  # Need Metal
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run unit tests
        run: cargo test

      - name: Run integration tests
        run: cargo test --test integration

      - name: Run benchmarks
        run: cargo bench --bench layout_bench

      - name: Wikipedia validation
        run: cargo test --test wikipedia_validation
```

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Wikipedia root height | < 50,000px | ~150,000px |
| Text overlap count | 0 | Many |
| mw-hidden display:none | 100% | ~5% |
| Pipeline time (Wikipedia) | < 100ms | ~150ms |
| CSS rule parsing | > 500 rules | Unknown |
| Selector matching | 100% correct | Unknown |

## Files to Create

| File | Description |
|------|-------------|
| `tests/integration/mod.rs` | Integration test harness |
| `tests/integration/pipeline_tests.rs` | Full pipeline tests |
| `tests/integration/wikipedia_tests.rs` | Wikipedia-specific tests |
| `tests/integration/visual_regression.rs` | Visual comparison tests |
| `benches/layout_bench.rs` | Performance benchmarks |
| `testdata/` | Test fixtures directory |

## Acceptance Criteria

1. [ ] Full pipeline integration tests pass
2. [ ] External CSS loading works in tests
3. [ ] Wikipedia article loads without errors
4. [ ] Wikipedia root height < 50,000px
5. [ ] No text element overlaps
6. [ ] mw-hidden elements have display:none
7. [ ] Performance benchmarks established
8. [ ] Visual regression test framework set up
9. [ ] CI pipeline configured
10. [ ] All tests pass
