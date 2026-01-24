//! Test suite for Issue #64: Link Navigation System
//!
//! Tests link extraction, URL resolution, and navigation history.

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, LinkExtractor, UrlResolver, DocumentLinks,
    NavigationHistory, NavigationRequest, HistoryEntry, LinkTarget,
};

// ======= LINK EXTRACTION =======

fn parse_html(html: &[u8]) -> (
    Vec<rust_experiment::gpu_os::document::Element>,
    Vec<rust_experiment::gpu_os::document::Token>,
) {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");

    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);
    (elements, tokens)
}

#[test]
fn test_extract_links() {
    let html = b"<a href=\"https://example.com\">Link</a>";
    let (elements, tokens) = parse_html(html);

    let links = LinkExtractor::extract_links(&elements, &tokens, html);

    assert_eq!(links.len(), 1, "Should extract 1 link");
    assert_eq!(links[0].href, "https://example.com");
    assert!(!links[0].is_fragment);
}

#[test]
fn test_extract_multiple_links() {
    let html = b"<a href=\"/page1\">Page 1</a><a href=\"/page2\">Page 2</a>";
    let (elements, tokens) = parse_html(html);

    let links = LinkExtractor::extract_links(&elements, &tokens, html);

    assert_eq!(links.len(), 2, "Should extract 2 links");
    assert_eq!(links[0].href, "/page1");
    assert_eq!(links[1].href, "/page2");
}

#[test]
fn test_extract_fragment_link() {
    let html = b"<a href=\"#section\">Jump</a>";
    let (elements, tokens) = parse_html(html);

    let links = LinkExtractor::extract_links(&elements, &tokens, html);

    assert_eq!(links.len(), 1);
    assert!(links[0].is_fragment);
    assert_eq!(links[0].fragment, Some("section".to_string()));
}

#[test]
fn test_extract_link_with_target() {
    let html = b"<a href=\"/page\" target=\"_blank\">External</a>";
    let (elements, tokens) = parse_html(html);

    let links = LinkExtractor::extract_links(&elements, &tokens, html);

    assert_eq!(links.len(), 1);
    assert_eq!(links[0].target, LinkTarget::Blank);
}

// ======= URL RESOLUTION =======

#[test]
fn test_resolve_absolute_url() {
    let base = "https://example.com/page.html";
    let href = "https://other.com/test.html";
    assert_eq!(UrlResolver::resolve(base, href), href);
}

#[test]
fn test_resolve_protocol_relative() {
    let base = "https://example.com/page.html";
    let href = "//cdn.example.com/script.js";
    assert_eq!(UrlResolver::resolve(base, href), "https://cdn.example.com/script.js");
}

#[test]
fn test_resolve_fragment_only() {
    let base = "https://example.com/page.html";
    let href = "#section";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/page.html#section");
}

#[test]
fn test_resolve_fragment_with_existing_fragment() {
    let base = "https://example.com/page.html#old";
    let href = "#new";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/page.html#new");
}

#[test]
fn test_resolve_root_relative() {
    let base = "https://example.com/path/page.html";
    let href = "/other/test.html";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/other/test.html");
}

#[test]
fn test_resolve_relative() {
    let base = "https://example.com/path/page.html";
    let href = "other.html";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/path/other.html");
}

#[test]
fn test_resolve_parent_relative() {
    let base = "https://example.com/path/sub/page.html";
    let href = "../other.html";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/path/other.html");
}

#[test]
fn test_resolve_complex_relative() {
    let base = "https://example.com/a/b/c/page.html";
    let href = "../../d/e.html";
    assert_eq!(UrlResolver::resolve(base, href), "https://example.com/a/d/e.html");
}

#[test]
fn test_same_origin() {
    let base = "https://example.com/page.html";
    assert!(UrlResolver::is_same_origin(base, "https://example.com/other.html"));
    assert!(UrlResolver::is_same_origin(base, "https://example.com/path/page.html"));
    assert!(!UrlResolver::is_same_origin(base, "https://other.com/page.html"));
    assert!(!UrlResolver::is_same_origin(base, "http://example.com/page.html")); // Different protocol
}

#[test]
fn test_extract_fragment() {
    assert_eq!(UrlResolver::extract_fragment("https://example.com#section"), Some("section".to_string()));
    assert_eq!(UrlResolver::extract_fragment("https://example.com"), None);
    assert_eq!(UrlResolver::extract_fragment("#section"), Some("section".to_string()));
}

// ======= DOCUMENT LINKS =======

#[test]
fn test_document_links() {
    let html = b"<div><a href=\"/page1\">Link 1</a><p>text</p><a href=\"#anchor\">Anchor</a></div>";
    let (elements, tokens) = parse_html(html);

    let doc_links = DocumentLinks::new(&elements, &tokens, html, "https://example.com/base.html");

    assert_eq!(doc_links.all_links().len(), 2);

    // Resolve relative link
    let resolved = doc_links.resolve_href("/page1");
    assert_eq!(resolved, "https://example.com/page1");
}

#[test]
fn test_get_link_for_element() {
    let html = b"<a href=\"/link\"><span>Click</span></a>";
    let (elements, tokens) = parse_html(html);

    let doc_links = DocumentLinks::new(&elements, &tokens, html, "https://example.com/");

    // Find the span element (child of <a>)
    let span_idx = elements.iter()
        .position(|e| e.element_type == 2)  // ELEM_SPAN = 2
        .expect("Should find span");

    // Should find parent link
    let link = doc_links.get_link_for_element(span_idx as u32, &elements);
    assert!(link.is_some(), "Should find link for span's parent");
}

// ======= NAVIGATION HISTORY =======

#[test]
fn test_history_push() {
    let mut history = NavigationHistory::new();
    assert!(history.is_empty());

    history.push("https://page1.com");
    assert_eq!(history.len(), 1);
    assert_eq!(history.current_url(), Some("https://page1.com"));

    history.push("https://page2.com");
    assert_eq!(history.len(), 2);
    assert_eq!(history.current_url(), Some("https://page2.com"));
}

#[test]
fn test_history_back_forward() {
    let mut history = NavigationHistory::new();
    history.push("https://page1.com");
    history.push("https://page2.com");
    history.push("https://page3.com");

    assert!(history.can_go_back());
    assert!(!history.can_go_forward());

    // Back to page2
    let entry = history.back();
    assert!(entry.is_some());
    assert_eq!(history.current_url(), Some("https://page2.com"));
    assert!(history.can_go_forward());

    // Back to page1
    history.back();
    assert_eq!(history.current_url(), Some("https://page1.com"));
    assert!(!history.can_go_back());

    // Forward to page2
    history.forward();
    assert_eq!(history.current_url(), Some("https://page2.com"));

    // Forward to page3
    history.forward();
    assert_eq!(history.current_url(), Some("https://page3.com"));
    assert!(!history.can_go_forward());
}

#[test]
fn test_history_push_clears_forward() {
    let mut history = NavigationHistory::new();
    history.push("https://page1.com");
    history.push("https://page2.com");
    history.push("https://page3.com");

    // Go back twice
    history.back();
    history.back();

    // Push new page (should clear forward history)
    history.push("https://page4.com");

    assert_eq!(history.len(), 2);
    assert!(!history.can_go_forward());
    assert_eq!(history.current_url(), Some("https://page4.com"));
}

#[test]
fn test_history_scroll_preservation() {
    let mut history = NavigationHistory::new();
    history.push("https://page1.com");
    history.update_scroll(100.0, 500.0);

    history.push("https://page2.com");
    history.update_scroll(0.0, 1000.0);

    // Go back
    let entry = history.back().unwrap();
    assert_eq!(entry.scroll_x, 100.0);
    assert_eq!(entry.scroll_y, 500.0);
}

#[test]
fn test_history_max_size() {
    let mut history = NavigationHistory::with_max_size(3);

    history.push("https://page1.com");
    history.push("https://page2.com");
    history.push("https://page3.com");
    history.push("https://page4.com");

    assert_eq!(history.len(), 3);
    assert_eq!(history.entries()[0].url, "https://page2.com");
}

#[test]
fn test_history_replace() {
    let mut history = NavigationHistory::new();
    history.push("https://page1.com");
    history.push("https://page2.com");

    history.replace("https://page2-modified.com");

    assert_eq!(history.len(), 2);
    assert_eq!(history.current_url(), Some("https://page2-modified.com"));
}

// ======= NAVIGATION REQUEST =======

#[test]
fn test_navigation_request() {
    let req = NavigationRequest::new("https://example.com/page.html");
    assert!(!req.is_fragment);
    assert!(req.add_to_history);
    assert_eq!(req.fragment, None);

    let frag_req = NavigationRequest::fragment("section");
    assert!(frag_req.is_fragment);
    assert_eq!(frag_req.fragment, Some("section".to_string()));

    let replace_req = NavigationRequest::replace("https://example.com");
    assert!(!replace_req.add_to_history);
}

#[test]
fn test_navigation_request_with_fragment() {
    let req = NavigationRequest::new("https://example.com/page.html#section");
    assert!(!req.is_fragment);  // Full URL with fragment, not fragment-only
    assert_eq!(req.fragment, Some("section".to_string()));
}
