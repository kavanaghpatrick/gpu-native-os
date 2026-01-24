# Issue #94: External CSS Loading

## Summary
Fetch external stylesheets referenced in `<link>` tags and `<style>` elements, concatenate into a single CSS buffer for GPU processing.

## Problem
Wikipedia and most websites use external CSS files for styling. Without loading these, we miss critical rules like `display: none` for hidden elements.

## Solution

### CPU-side implementation (I/O only)

```rust
pub struct CSSLoader {
    client: reqwest::Client,
}

impl CSSLoader {
    /// Extract CSS URLs from HTML
    pub fn extract_stylesheet_urls(html: &str, base_url: &str) -> Vec<String> {
        // Find all <link rel="stylesheet" href="...">
        // Find all @import url(...) in <style> tags
        // Resolve relative URLs against base_url
    }

    /// Fetch all stylesheets in parallel
    pub async fn fetch_stylesheets(urls: Vec<String>) -> Vec<(String, String)> {
        // Parallel fetch using futures::join_all
        // Return (url, css_content) pairs
    }

    /// Extract inline <style> content
    pub fn extract_inline_styles(html: &str) -> Vec<String> {
        // Find all <style>...</style> content
    }

    /// Concatenate all CSS into single buffer
    pub fn build_css_buffer(
        inline_styles: Vec<String>,
        external_styles: Vec<(String, String)>,
    ) -> (Vec<u8>, Vec<CSSSourceInfo>) {
        // Concatenate with source tracking for error reporting
    }
}

#[repr(C)]
pub struct CSSSourceInfo {
    pub start_offset: u32,
    pub length: u32,
    pub source_url_hash: u32,  // For debugging
    pub is_inline: u32,
}
```

## Pseudocode

```
FUNCTION load_document_with_css(url):
    // Step 1: Fetch HTML
    html = fetch(url)
    base_url = extract_base_url(url, html)

    // Step 2: Extract stylesheet references
    stylesheet_urls = []
    inline_styles = []

    FOR each tag in html:
        IF tag is <link rel="stylesheet" href="...">:
            href = resolve_url(tag.href, base_url)
            stylesheet_urls.append(href)

        IF tag is <style>:
            inline_styles.append(tag.content)

        IF tag is <style> with @import:
            FOR each @import url(...):
                import_url = resolve_url(url, base_url)
                stylesheet_urls.append(import_url)

    // Step 3: Fetch external stylesheets (parallel)
    external_css = parallel_fetch(stylesheet_urls)

    // Step 4: Build combined CSS buffer
    // Order matters for cascade: earlier = lower priority
    css_buffer = ""
    source_info = []

    // User agent stylesheet first (lowest priority)
    css_buffer += USER_AGENT_CSS
    source_info.append(CSSSourceInfo{...})

    // External stylesheets in document order
    FOR (url, content) in external_css:
        source_info.append(CSSSourceInfo{
            start_offset: len(css_buffer),
            length: len(content),
            source_url_hash: hash(url),
            is_inline: false
        })
        css_buffer += content

    // Inline styles last (highest priority for same specificity)
    FOR content in inline_styles:
        source_info.append(CSSSourceInfo{
            start_offset: len(css_buffer),
            length: len(content),
            is_inline: true
        })
        css_buffer += content

    RETURN (html, css_buffer, source_info)
```

## User Agent Stylesheet

Minimal default styles (CSS 2.1 Appendix D):

```css
/* Block elements */
html, body, div, p, h1, h2, h3, h4, h5, h6,
ul, ol, li, dl, dt, dd, table, tr, td, th,
form, fieldset, header, footer, main, section,
article, aside, nav, figure, figcaption, pre,
blockquote, address { display: block; }

/* Headings */
h1 { font-size: 2em; font-weight: bold; margin: 0.67em 0; }
h2 { font-size: 1.5em; font-weight: bold; margin: 0.83em 0; }
h3 { font-size: 1.17em; font-weight: bold; margin: 1em 0; }
h4 { font-size: 1em; font-weight: bold; margin: 1.33em 0; }
h5 { font-size: 0.83em; font-weight: bold; margin: 1.67em 0; }
h6 { font-size: 0.67em; font-weight: bold; margin: 2.33em 0; }

/* Lists */
ul, ol { padding-left: 40px; margin: 1em 0; }
li { display: list-item; }

/* Hidden elements */
head, script, style, meta, link, title { display: none; }

/* Inline elements */
span, a, em, strong, code, b, i, u, s, small, big,
sub, sup, abbr, cite, q, label { display: inline; }

/* Default body margin */
body { margin: 8px; }

/* Paragraphs */
p { margin: 1em 0; }

/* Links */
a { color: blue; text-decoration: underline; }

/* Preformatted */
pre, code { font-family: monospace; }
pre { white-space: pre; margin: 1em 0; }
```

## API

```rust
impl DocumentPipeline {
    /// Load document with external CSS
    pub async fn load_with_css(&mut self, url: &str) -> Result<(), Error> {
        let (html, css_buffer, source_info) = CSSLoader::load_document_with_css(url).await?;

        // Upload to GPU
        self.html_buffer.copy_from_slice(&html);
        self.css_buffer.copy_from_slice(&css_buffer);
        self.css_source_buffer.copy_from_slice(&source_info);

        // Run pipeline
        self.process()
    }
}
```

## Tests

### Test 1: Extract stylesheet URLs
```rust
#[test]
fn test_extract_stylesheet_urls() {
    let html = r#"
        <html>
        <head>
            <link rel="stylesheet" href="/styles/main.css">
            <link rel="stylesheet" href="https://cdn.example.com/lib.css">
            <link rel="icon" href="/favicon.ico">
        </head>
        </html>
    "#;

    let urls = CSSLoader::extract_stylesheet_urls(html, "https://example.com/page");

    assert_eq!(urls, vec![
        "https://example.com/styles/main.css",
        "https://cdn.example.com/lib.css",
    ]);
}
```

### Test 2: Extract inline styles
```rust
#[test]
fn test_extract_inline_styles() {
    let html = r#"
        <html>
        <head>
            <style>
                .hidden { display: none; }
            </style>
        </head>
        <body>
            <style>
                .red { color: red; }
            </style>
        </body>
        </html>
    "#;

    let styles = CSSLoader::extract_inline_styles(html);

    assert_eq!(styles.len(), 2);
    assert!(styles[0].contains(".hidden"));
    assert!(styles[1].contains(".red"));
}
```

### Test 3: Resolve relative URLs
```rust
#[test]
fn test_resolve_urls() {
    let base = "https://en.wikipedia.org/wiki/Rust";

    assert_eq!(
        resolve_url("/static/style.css", base),
        "https://en.wikipedia.org/static/style.css"
    );

    assert_eq!(
        resolve_url("../common.css", base),
        "https://en.wikipedia.org/common.css"
    );

    assert_eq!(
        resolve_url("https://cdn.wiki.org/x.css", base),
        "https://cdn.wiki.org/x.css"
    );
}
```

### Test 4: CSS buffer ordering
```rust
#[test]
fn test_css_buffer_ordering() {
    let inline = vec!["/* inline */".to_string()];
    let external = vec![
        ("http://a.com/1.css".to_string(), "/* external 1 */".to_string()),
        ("http://a.com/2.css".to_string(), "/* external 2 */".to_string()),
    ];

    let (buffer, info) = CSSLoader::build_css_buffer(inline, external);
    let css = String::from_utf8(buffer).unwrap();

    // User agent first, then external in order, then inline last
    assert!(css.find("/* external 1 */") < css.find("/* external 2 */"));
    assert!(css.find("/* external 2 */") < css.find("/* inline */"));
}
```

### Test 5: Integration with Wikipedia
```rust
#[tokio::test]
async fn test_wikipedia_css_loading() {
    let url = "https://en.wikipedia.org/wiki/Rust_(programming_language)";
    let (html, css, _) = CSSLoader::load_document_with_css(url).await.unwrap();

    // Should have fetched external CSS
    assert!(css.len() > 100_000); // Wikipedia CSS is large

    // Should contain MediaWiki styles
    let css_str = String::from_utf8_lossy(&css);
    assert!(css_str.contains(".mw-"));  // MediaWiki classes
    assert!(css_str.contains("display")); // display rules
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/css_loader.rs` | Create | CSS loading implementation |
| `src/gpu_os/document/mod.rs` | Modify | Export css_loader module |
| `src/gpu_os/document/user_agent.css` | Create | Default stylesheet |
| `tests/test_issue_94_css_loading.rs` | Create | Integration tests |
| `Cargo.toml` | Modify | Add reqwest, url crates |

## Acceptance Criteria

1. [ ] Extract `<link rel="stylesheet">` URLs from HTML
2. [ ] Extract `<style>` inline content
3. [ ] Fetch external stylesheets in parallel
4. [ ] Handle relative URL resolution
5. [ ] Build concatenated CSS buffer with source tracking
6. [ ] Include user agent stylesheet
7. [ ] All tests pass
8. [ ] Wikipedia CSS loads successfully (>100KB)
