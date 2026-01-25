//! Result rendering for GPU Shell
//!
//! Formats values for display in the terminal.

use super::value::{Value, FileRow, GroupRow, SearchHit, format_size, format_time};

/// Renderer for table output
pub struct TableRenderer {
    pub max_width: usize,
    pub max_rows: usize,
}

impl Default for TableRenderer {
    fn default() -> Self {
        Self {
            max_width: 120,
            max_rows: 100,
        }
    }
}

impl TableRenderer {
    /// Render a value to a string
    pub fn render(&self, value: &Value, path_buffer: Option<&[u8]>) -> String {
        match value {
            Value::Files { rows, path_buffer, source_path } => {
                self.render_files(rows, path_buffer, source_path)
            }
            Value::SearchResults { hits, pattern } => {
                self.render_search_results(hits, pattern)
            }
            Value::Groups { rows, group_field } => {
                self.render_groups(rows, group_field)
            }
            Value::Count(n) => format!("{}", n),
            Value::Sum(n) => format_size(*n),
            Value::Text(s) => s.clone(),
            Value::Void => "(no result)".to_string(),
        }
    }

    /// Render file listing
    fn render_files(&self, rows: &[FileRow], path_buffer: &[u8], source_path: &str) -> String {
        if rows.is_empty() {
            return format!("No files found in {}", source_path);
        }

        let display_rows: Vec<_> = rows.iter().take(self.max_rows).collect();
        let truncated = rows.len() > self.max_rows;

        // Calculate column widths
        let mut path_width = 4; // "path"
        let mut size_width = 4; // "size"
        let mut modified_width = 8; // "modified"

        for row in &display_rows {
            let path = self.get_path(row, path_buffer);
            let display_path = self.truncate_path(&path, 60);
            path_width = path_width.max(display_path.len());

            let size_str = if row.is_dir { "<DIR>".to_string() } else { format_size(row.size) };
            size_width = size_width.max(size_str.len());

            let modified_str = format_time(row.modified);
            modified_width = modified_width.max(modified_str.len());
        }

        // Cap widths
        path_width = path_width.min(60);
        size_width = size_width.min(12);
        modified_width = modified_width.min(15);

        let mut output = String::new();

        // Header
        output.push_str(&format!(
            "┌{:─<path_w$}┬{:─<size_w$}┬{:─<mod_w$}┐\n",
            "", "", "",
            path_w = path_width + 2,
            size_w = size_width + 2,
            mod_w = modified_width + 2
        ));

        output.push_str(&format!(
            "│ {:<path_w$} │ {:>size_w$} │ {:<mod_w$} │\n",
            "path", "size", "modified",
            path_w = path_width,
            size_w = size_width,
            mod_w = modified_width
        ));

        output.push_str(&format!(
            "├{:─<path_w$}┼{:─<size_w$}┼{:─<mod_w$}┤\n",
            "", "", "",
            path_w = path_width + 2,
            size_w = size_width + 2,
            mod_w = modified_width + 2
        ));

        // Rows
        for row in &display_rows {
            let path = self.get_path(row, path_buffer);
            let display_path = self.truncate_path(&path, path_width);
            let size_str = if row.is_dir { "<DIR>".to_string() } else { format_size(row.size) };
            let modified_str = format_time(row.modified);

            output.push_str(&format!(
                "│ {:<path_w$} │ {:>size_w$} │ {:<mod_w$} │\n",
                display_path, size_str, modified_str,
                path_w = path_width,
                size_w = size_width,
                mod_w = modified_width
            ));
        }

        output.push_str(&format!(
            "└{:─<path_w$}┴{:─<size_w$}┴{:─<mod_w$}┘\n",
            "", "", "",
            path_w = path_width + 2,
            size_w = size_width + 2,
            mod_w = modified_width + 2
        ));

        // Summary
        let total_size: u64 = rows.iter().filter(|r| !r.is_dir).map(|r| r.size).sum();
        let file_count = rows.iter().filter(|r| !r.is_dir).count();
        let dir_count = rows.iter().filter(|r| r.is_dir).count();

        output.push_str(&format!(
            "{} files, {} directories ({})",
            file_count, dir_count, format_size(total_size)
        ));

        if truncated {
            output.push_str(&format!(" [showing first {} of {}]", self.max_rows, rows.len()));
        }

        output
    }

    /// Render search results
    fn render_search_results(&self, hits: &[SearchHit], pattern: &str) -> String {
        if hits.is_empty() {
            return format!("No matches found for '{}'", pattern);
        }

        let display_hits: Vec<_> = hits.iter().take(self.max_rows).collect();
        let truncated = hits.len() > self.max_rows;

        let mut path_width = 4;
        let mut line_width = 4;
        let content_width = 60;

        for hit in &display_hits {
            let display_path = self.truncate_path(&hit.file_path, 40);
            path_width = path_width.max(display_path.len()).min(40);
            line_width = line_width.max(hit.line_number.to_string().len()).min(6);
        }

        let mut output = String::new();

        // Header
        output.push_str(&format!(
            "┌{:─<path_w$}┬{:─<line_w$}┬{:─<content_w$}┐\n",
            "", "", "",
            path_w = path_width + 2,
            line_w = line_width + 2,
            content_w = content_width + 2
        ));

        output.push_str(&format!(
            "│ {:<path_w$} │ {:>line_w$} │ {:<content_w$} │\n",
            "file", "line", "content",
            path_w = path_width,
            line_w = line_width,
            content_w = content_width
        ));

        output.push_str(&format!(
            "├{:─<path_w$}┼{:─<line_w$}┼{:─<content_w$}┤\n",
            "", "", "",
            path_w = path_width + 2,
            line_w = line_width + 2,
            content_w = content_width + 2
        ));

        for hit in &display_hits {
            let display_path = self.truncate_path(&hit.file_path, path_width);
            let content = self.truncate_string(&hit.content, content_width);

            output.push_str(&format!(
                "│ {:<path_w$} │ {:>line_w$} │ {:<content_w$} │\n",
                display_path, hit.line_number, content,
                path_w = path_width,
                line_w = line_width,
                content_w = content_width
            ));
        }

        output.push_str(&format!(
            "└{:─<path_w$}┴{:─<line_w$}┴{:─<content_w$}┘\n",
            "", "", "",
            path_w = path_width + 2,
            line_w = line_width + 2,
            content_w = content_width + 2
        ));

        output.push_str(&format!("{} matches for '{}'", hits.len(), pattern));

        if truncated {
            output.push_str(&format!(" [showing first {}]", self.max_rows));
        }

        output
    }

    /// Render grouped results
    fn render_groups(&self, rows: &[GroupRow], group_field: &str) -> String {
        if rows.is_empty() {
            return "No groups".to_string();
        }

        let display_rows: Vec<_> = rows.iter().take(self.max_rows).collect();
        let truncated = rows.len() > self.max_rows;

        let mut key_width = group_field.len();
        let mut count_width = 5;
        let mut size_width = 10;

        for row in &display_rows {
            key_width = key_width.max(row.key.len()).min(40);
            count_width = count_width.max(row.count.to_string().len());
            size_width = size_width.max(format_size(row.total_size).len());
        }

        let mut output = String::new();

        // Header
        output.push_str(&format!(
            "┌{:─<key_w$}┬{:─<count_w$}┬{:─<size_w$}┐\n",
            "", "", "",
            key_w = key_width + 2,
            count_w = count_width + 2,
            size_w = size_width + 2
        ));

        output.push_str(&format!(
            "│ {:<key_w$} │ {:>count_w$} │ {:>size_w$} │\n",
            group_field, "count", "total_size",
            key_w = key_width,
            count_w = count_width,
            size_w = size_width
        ));

        output.push_str(&format!(
            "├{:─<key_w$}┼{:─<count_w$}┼{:─<size_w$}┤\n",
            "", "", "",
            key_w = key_width + 2,
            count_w = count_width + 2,
            size_w = size_width + 2
        ));

        for row in &display_rows {
            let key = self.truncate_string(&row.key, key_width);
            output.push_str(&format!(
                "│ {:<key_w$} │ {:>count_w$} │ {:>size_w$} │\n",
                key, row.count, format_size(row.total_size),
                key_w = key_width,
                count_w = count_width,
                size_w = size_width
            ));
        }

        output.push_str(&format!(
            "└{:─<key_w$}┴{:─<count_w$}┴{:─<size_w$}┘\n",
            "", "", "",
            key_w = key_width + 2,
            count_w = count_width + 2,
            size_w = size_width + 2
        ));

        output.push_str(&format!("{} groups", rows.len()));

        if truncated {
            output.push_str(&format!(" [showing first {}]", self.max_rows));
        }

        output
    }

    /// Get path from a file row
    fn get_path(&self, row: &FileRow, path_buffer: &[u8]) -> String {
        let start = row.path_offset as usize;
        let end = start + row.path_len as usize;
        if end <= path_buffer.len() {
            String::from_utf8_lossy(&path_buffer[start..end]).to_string()
        } else {
            "<invalid>".to_string()
        }
    }

    /// Truncate a path for display
    fn truncate_path(&self, path: &str, max_len: usize) -> String {
        if path.len() <= max_len {
            return path.to_string();
        }

        // Try to keep the filename visible
        if let Some(pos) = path.rfind('/') {
            let filename = &path[pos + 1..];
            if filename.len() < max_len - 4 {
                let available = max_len - filename.len() - 4; // ".../"
                return format!("...{}", &path[path.len() - max_len + 3..]);
            }
        }

        format!("{}...", &path[..max_len - 3])
    }

    /// Truncate a string for display
    fn truncate_string(&self, s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }
}
