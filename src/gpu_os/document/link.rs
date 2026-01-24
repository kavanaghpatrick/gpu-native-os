//! GPU-Native Link System
//!
//! Extracts and manages hyperlinks from parsed documents.
//! Provides URL resolution and link detection for navigation.

use super::parser::{Element, ELEM_A};
use super::tokenizer::Token;

/// Link target type
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkTarget {
    /// Same frame (default)
    Self_,
    /// New window/tab
    Blank,
    /// Parent frame
    Parent,
    /// Top frame
    Top,
}

impl Default for LinkTarget {
    fn default() -> Self {
        LinkTarget::Self_
    }
}

/// Information about a link in the document
#[derive(Clone, Debug)]
pub struct LinkInfo {
    /// Element index of the <a> element
    pub element_id: u32,
    /// Raw href attribute value
    pub href: String,
    /// Link target
    pub target: LinkTarget,
    /// Whether this is a fragment-only link (#anchor)
    pub is_fragment: bool,
    /// Fragment portion if present
    pub fragment: Option<String>,
}

/// Extracts links from parsed document
pub struct LinkExtractor;

impl LinkExtractor {
    /// Extract all links from a parsed document
    pub fn extract_links(
        elements: &[Element],
        tokens: &[Token],
        html: &[u8],
    ) -> Vec<LinkInfo> {
        let mut links = Vec::new();

        for (idx, elem) in elements.iter().enumerate() {
            if elem.element_type == ELEM_A {
                if let Some(link_info) = Self::extract_link_info(idx as u32, elem, tokens, html) {
                    links.push(link_info);
                }
            }
        }

        links
    }

    /// Extract href and target from an <a> element
    fn extract_link_info(
        element_id: u32,
        elem: &Element,
        tokens: &[Token],
        html: &[u8],
    ) -> Option<LinkInfo> {
        let token = tokens.get(elem.token_index as usize)?;
        let tag_content = std::str::from_utf8(&html[token.start as usize..token.end as usize]).ok()?;

        // Extract href attribute
        let href = Self::extract_attribute(tag_content, "href")?;

        // Extract target attribute (defaults to _self)
        let target = Self::extract_attribute(tag_content, "target")
            .map(|t| match t.as_str() {
                "_blank" => LinkTarget::Blank,
                "_parent" => LinkTarget::Parent,
                "_top" => LinkTarget::Top,
                _ => LinkTarget::Self_,
            })
            .unwrap_or_default();

        let is_fragment = href.starts_with('#');
        let fragment = if href.contains('#') {
            href.split('#').nth(1).map(|s| s.to_string())
        } else {
            None
        };

        Some(LinkInfo {
            element_id,
            href,
            target,
            is_fragment,
            fragment,
        })
    }

    /// Extract an attribute value from tag content
    fn extract_attribute(tag_content: &str, attr_name: &str) -> Option<String> {
        // Look for attr="value" or attr='value'
        let search = format!("{}=", attr_name);
        let start = tag_content.find(&search)?;
        let after_eq = start + search.len();

        let bytes = tag_content.as_bytes();
        if after_eq >= bytes.len() {
            return None;
        }

        let quote = bytes[after_eq] as char;
        if quote != '"' && quote != '\'' {
            return None;
        }

        let value_start = after_eq + 1;
        let mut value_end = value_start;
        while value_end < bytes.len() && bytes[value_end] as char != quote {
            value_end += 1;
        }

        Some(tag_content[value_start..value_end].to_string())
    }
}

/// URL resolution utilities
pub struct UrlResolver;

impl UrlResolver {
    /// Resolve a potentially relative URL against a base URL
    pub fn resolve(base: &str, href: &str) -> String {
        let href = href.trim();

        // Absolute URL
        if href.starts_with("http://") || href.starts_with("https://") {
            return href.to_string();
        }

        // Protocol-relative URL
        if href.starts_with("//") {
            let protocol = base.split("://").next().unwrap_or("https");
            return format!("{}:{}", protocol, href);
        }

        // Fragment-only URL
        if href.starts_with('#') {
            let base_without_fragment = base.split('#').next().unwrap_or(base);
            return format!("{}{}", base_without_fragment, href);
        }

        // Data URL or javascript: - pass through
        if href.starts_with("data:") || href.starts_with("javascript:") {
            return href.to_string();
        }

        // Extract origin from base
        let origin = Self::extract_origin(base);

        // Root-relative URL
        if href.starts_with('/') {
            return format!("{}{}", origin, href);
        }

        // Relative URL
        let base_path = Self::extract_base_path(base);
        Self::normalize_path(&format!("{}/{}", base_path, href))
    }

    /// Extract origin (protocol + host + port) from URL
    fn extract_origin(url: &str) -> String {
        if let Some(idx) = url.find("://") {
            let after_protocol = idx + 3;
            if let Some(path_start) = url[after_protocol..].find('/') {
                return url[..after_protocol + path_start].to_string();
            }
        }
        url.to_string()
    }

    /// Extract base path (URL without filename)
    fn extract_base_path(url: &str) -> String {
        let without_fragment = url.split('#').next().unwrap_or(url);
        let without_query = without_fragment.split('?').next().unwrap_or(without_fragment);

        if let Some(last_slash) = without_query.rfind('/') {
            // Only take if it's after the protocol
            if let Some(protocol_end) = without_query.find("://") {
                if last_slash > protocol_end + 2 {
                    return without_query[..last_slash].to_string();
                }
            }
        }
        without_query.to_string()
    }

    /// Normalize a path by resolving . and ..
    fn normalize_path(path: &str) -> String {
        // Split into origin and path parts
        let (origin, rest) = if let Some(idx) = path.find("://") {
            let after_protocol = idx + 3;
            if let Some(path_start) = path[after_protocol..].find('/') {
                let split_at = after_protocol + path_start;
                (&path[..split_at], &path[split_at..])
            } else {
                (path, "/")
            }
        } else {
            ("", path)
        };

        // Normalize the path portion
        let mut segments: Vec<&str> = Vec::new();
        for segment in rest.split('/') {
            match segment {
                "" | "." => {} // Skip empty and current-dir
                ".." => { segments.pop(); }
                s => segments.push(s),
            }
        }

        if rest.ends_with('/') || rest.ends_with("/.") || rest.ends_with("/..") {
            format!("{}/{}/", origin, segments.join("/"))
        } else {
            format!("{}/{}", origin, segments.join("/"))
        }
    }

    /// Check if URL is same-origin
    pub fn is_same_origin(base: &str, url: &str) -> bool {
        let base_origin = Self::extract_origin(base);
        let url_origin = Self::extract_origin(url);
        base_origin == url_origin
    }

    /// Extract fragment from URL
    pub fn extract_fragment(url: &str) -> Option<String> {
        url.split('#').nth(1).map(|s| s.to_string())
    }
}

/// Document state that tracks links
pub struct DocumentLinks {
    links: Vec<LinkInfo>,
    base_url: String,
}

impl DocumentLinks {
    /// Create new document links from parsed document
    pub fn new(
        elements: &[Element],
        tokens: &[Token],
        html: &[u8],
        base_url: &str,
    ) -> Self {
        let links = LinkExtractor::extract_links(elements, tokens, html);
        Self {
            links,
            base_url: base_url.to_string(),
        }
    }

    /// Get all links in the document
    pub fn all_links(&self) -> &[LinkInfo] {
        &self.links
    }

    /// Check if an element is a link, returns link info if so
    pub fn get_link(&self, element_id: u32) -> Option<&LinkInfo> {
        self.links.iter().find(|l| l.element_id == element_id)
    }

    /// Check if any ancestor of an element is a link
    pub fn get_link_for_element(&self, element_id: u32, elements: &[Element]) -> Option<&LinkInfo> {
        // First check the element itself
        if let Some(link) = self.get_link(element_id) {
            return Some(link);
        }

        // Walk up parent chain
        let mut current = element_id;
        while let Some(elem) = elements.get(current as usize) {
            if elem.parent < 0 {
                break;
            }
            current = elem.parent as u32;
            if let Some(link) = self.get_link(current) {
                return Some(link);
            }
        }

        None
    }

    /// Resolve a link href to absolute URL
    pub fn resolve_href(&self, href: &str) -> String {
        UrlResolver::resolve(&self.base_url, href)
    }

    /// Update base URL
    pub fn set_base_url(&mut self, base_url: &str) {
        self.base_url = base_url.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_resolve_fragment() {
        let base = "https://example.com/page.html";
        let href = "#section";
        assert_eq!(UrlResolver::resolve(base, href), "https://example.com/page.html#section");
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
    fn test_extract_origin() {
        assert_eq!(UrlResolver::extract_origin("https://example.com/path"), "https://example.com");
        assert_eq!(UrlResolver::extract_origin("https://example.com:8080/path"), "https://example.com:8080");
    }

    #[test]
    fn test_is_same_origin() {
        let base = "https://example.com/page.html";
        assert!(UrlResolver::is_same_origin(base, "https://example.com/other.html"));
        assert!(!UrlResolver::is_same_origin(base, "https://other.com/page.html"));
    }
}
