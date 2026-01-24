//! Navigation History System
//!
//! Manages browser-like navigation history with back/forward support.

/// A single entry in the navigation history
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    /// URL of the page
    pub url: String,
    /// Scroll position X
    pub scroll_x: f32,
    /// Scroll position Y
    pub scroll_y: f32,
    /// Page title (if available)
    pub title: String,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            scroll_x: 0.0,
            scroll_y: 0.0,
            title: String::new(),
        }
    }

    /// Create entry with scroll position
    pub fn with_scroll(url: &str, scroll_x: f32, scroll_y: f32) -> Self {
        Self {
            url: url.to_string(),
            scroll_x,
            scroll_y,
            title: String::new(),
        }
    }
}

/// Navigation history manager
///
/// Implements browser-like back/forward navigation with state preservation.
#[derive(Clone, Debug)]
pub struct NavigationHistory {
    /// History entries
    entries: Vec<HistoryEntry>,
    /// Current position in history (0-indexed)
    current: usize,
    /// Maximum history size
    max_size: usize,
}

impl Default for NavigationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl NavigationHistory {
    /// Create a new empty navigation history
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            current: 0,
            max_size: 50,  // Default max history
        }
    }

    /// Create with custom max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            current: 0,
            max_size,
        }
    }

    /// Push a new entry to history
    ///
    /// This clears any forward history (entries after current position).
    pub fn push(&mut self, url: &str) {
        self.push_entry(HistoryEntry::new(url));
    }

    /// Push a new entry with scroll position
    pub fn push_with_scroll(&mut self, url: &str, scroll_x: f32, scroll_y: f32) {
        self.push_entry(HistoryEntry::with_scroll(url, scroll_x, scroll_y));
    }

    /// Push a history entry
    fn push_entry(&mut self, entry: HistoryEntry) {
        // If we're not at the end, truncate forward history
        if self.current + 1 < self.entries.len() {
            self.entries.truncate(self.current + 1);
        }

        // Add new entry
        self.entries.push(entry);
        self.current = self.entries.len() - 1;

        // Trim old entries if over max size
        if self.entries.len() > self.max_size {
            let remove_count = self.entries.len() - self.max_size;
            self.entries.drain(0..remove_count);
            self.current = self.current.saturating_sub(remove_count);
        }
    }

    /// Go back in history
    ///
    /// Returns the previous entry, or None if at beginning.
    pub fn back(&mut self) -> Option<&HistoryEntry> {
        if self.can_go_back() {
            self.current -= 1;
            Some(&self.entries[self.current])
        } else {
            None
        }
    }

    /// Go forward in history
    ///
    /// Returns the next entry, or None if at end.
    pub fn forward(&mut self) -> Option<&HistoryEntry> {
        if self.can_go_forward() {
            self.current += 1;
            Some(&self.entries[self.current])
        } else {
            None
        }
    }

    /// Check if we can go back
    pub fn can_go_back(&self) -> bool {
        self.current > 0
    }

    /// Check if we can go forward
    pub fn can_go_forward(&self) -> bool {
        self.current + 1 < self.entries.len()
    }

    /// Get current entry
    pub fn current(&self) -> Option<&HistoryEntry> {
        self.entries.get(self.current)
    }

    /// Get current URL
    pub fn current_url(&self) -> Option<&str> {
        self.current().map(|e| e.url.as_str())
    }

    /// Update scroll position for current entry
    pub fn update_scroll(&mut self, scroll_x: f32, scroll_y: f32) {
        if let Some(entry) = self.entries.get_mut(self.current) {
            entry.scroll_x = scroll_x;
            entry.scroll_y = scroll_y;
        }
    }

    /// Update title for current entry
    pub fn update_title(&mut self, title: &str) {
        if let Some(entry) = self.entries.get_mut(self.current) {
            entry.title = title.to_string();
        }
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries (for display)
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Get current index
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current = 0;
    }

    /// Replace current entry (used for fragment navigation that shouldn't add new entry)
    pub fn replace(&mut self, url: &str) {
        if let Some(entry) = self.entries.get_mut(self.current) {
            entry.url = url.to_string();
        } else {
            // No entries yet, push new one
            self.push(url);
        }
    }
}

/// Navigation state machine
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NavigationState {
    /// Idle, ready for user interaction
    Idle,
    /// Loading a new page
    Loading { url: String },
    /// Navigation was cancelled
    Cancelled,
    /// Navigation failed
    Failed { url: String, reason: String },
}

impl Default for NavigationState {
    fn default() -> Self {
        NavigationState::Idle
    }
}

/// Navigation request
#[derive(Clone, Debug)]
pub struct NavigationRequest {
    /// Target URL
    pub url: String,
    /// Whether this is a user-initiated navigation
    pub user_initiated: bool,
    /// Whether to add to history
    pub add_to_history: bool,
    /// Whether this is a fragment-only navigation
    pub is_fragment: bool,
    /// Target fragment (if any)
    pub fragment: Option<String>,
}

impl NavigationRequest {
    /// Create a normal navigation request
    pub fn new(url: &str) -> Self {
        let is_fragment = url.starts_with('#');
        let fragment = if url.contains('#') {
            url.split('#').nth(1).map(|s| s.to_string())
        } else {
            None
        };

        Self {
            url: url.to_string(),
            user_initiated: true,
            add_to_history: true,
            is_fragment,
            fragment,
        }
    }

    /// Create a fragment-only navigation (same-page scroll)
    pub fn fragment(fragment: &str) -> Self {
        Self {
            url: format!("#{}", fragment),
            user_initiated: true,
            add_to_history: true,
            is_fragment: true,
            fragment: Some(fragment.to_string()),
        }
    }

    /// Create a replacement navigation (doesn't add to history)
    pub fn replace(url: &str) -> Self {
        let is_fragment = url.starts_with('#');
        let fragment = if url.contains('#') {
            url.split('#').nth(1).map(|s| s.to_string())
        } else {
            None
        };

        Self {
            url: url.to_string(),
            user_initiated: true,
            add_to_history: false,
            is_fragment,
            fragment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_current() {
        let mut history = NavigationHistory::new();
        assert!(history.is_empty());

        history.push("https://example.com");
        assert_eq!(history.current_url(), Some("https://example.com"));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_back_forward() {
        let mut history = NavigationHistory::new();
        history.push("https://page1.com");
        history.push("https://page2.com");
        history.push("https://page3.com");

        assert_eq!(history.current_url(), Some("https://page3.com"));
        assert!(history.can_go_back());
        assert!(!history.can_go_forward());

        // Go back
        let prev = history.back();
        assert!(prev.is_some());
        assert_eq!(history.current_url(), Some("https://page2.com"));
        assert!(history.can_go_forward());

        // Go back again
        history.back();
        assert_eq!(history.current_url(), Some("https://page1.com"));
        assert!(!history.can_go_back());

        // Go forward
        history.forward();
        assert_eq!(history.current_url(), Some("https://page2.com"));

        // Go forward again
        history.forward();
        assert_eq!(history.current_url(), Some("https://page3.com"));
        assert!(!history.can_go_forward());
    }

    #[test]
    fn test_push_clears_forward_history() {
        let mut history = NavigationHistory::new();
        history.push("https://page1.com");
        history.push("https://page2.com");
        history.push("https://page3.com");

        // Go back twice
        history.back();
        history.back();
        assert_eq!(history.current_url(), Some("https://page1.com"));

        // Push new page (should clear page2 and page3)
        history.push("https://page4.com");
        assert_eq!(history.len(), 2);
        assert_eq!(history.current_url(), Some("https://page4.com"));
        assert!(!history.can_go_forward());
    }

    #[test]
    fn test_max_size() {
        let mut history = NavigationHistory::with_max_size(3);

        history.push("https://page1.com");
        history.push("https://page2.com");
        history.push("https://page3.com");
        assert_eq!(history.len(), 3);

        // Push one more, should remove oldest
        history.push("https://page4.com");
        assert_eq!(history.len(), 3);
        assert_eq!(history.entries()[0].url, "https://page2.com");
    }

    #[test]
    fn test_scroll_update() {
        let mut history = NavigationHistory::new();
        history.push("https://example.com");

        history.update_scroll(100.0, 500.0);

        let entry = history.current().unwrap();
        assert_eq!(entry.scroll_x, 100.0);
        assert_eq!(entry.scroll_y, 500.0);
    }

    #[test]
    fn test_replace() {
        let mut history = NavigationHistory::new();
        history.push("https://page1.com");
        history.push("https://page2.com");

        history.replace("https://page2-modified.com");

        assert_eq!(history.len(), 2);
        assert_eq!(history.current_url(), Some("https://page2-modified.com"));
    }
}
