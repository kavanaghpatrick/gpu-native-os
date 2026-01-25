//! Value types for GPU Shell
//!
//! Represents data flowing through pipelines - GPU buffers, scalars, tables.

use std::fmt;

/// A row in the file table
#[derive(Clone, Debug)]
#[repr(C)]
pub struct FileRow {
    pub path_offset: u32,
    pub path_len: u16,
    pub size: u64,
    pub modified: u64,
    pub is_dir: bool,
    pub ext_hash: u32,
    pub ext_bytes: [u8; 8], // Extension string for display
}

impl FileRow {
    /// Get extension as string
    pub fn ext(&self) -> String {
        let len = self.ext_bytes.iter().position(|&b| b == 0).unwrap_or(8);
        String::from_utf8_lossy(&self.ext_bytes[..len]).to_string()
    }
}

/// Column metadata for schema
#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub col_type: ColumnType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnType {
    String,
    U64,
    Bool,
}

/// Schema describing a table's columns
#[derive(Clone, Debug)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn file_schema() -> Self {
        Self {
            columns: vec![
                Column { name: "path".into(), col_type: ColumnType::String },
                Column { name: "size".into(), col_type: ColumnType::U64 },
                Column { name: "modified".into(), col_type: ColumnType::U64 },
                Column { name: "ext".into(), col_type: ColumnType::String },
                Column { name: "is_dir".into(), col_type: ColumnType::Bool },
            ],
        }
    }

    pub fn search_schema() -> Self {
        Self {
            columns: vec![
                Column { name: "file".into(), col_type: ColumnType::String },
                Column { name: "line".into(), col_type: ColumnType::U64 },
                Column { name: "content".into(), col_type: ColumnType::String },
            ],
        }
    }

    pub fn group_schema() -> Self {
        Self {
            columns: vec![
                Column { name: "key".into(), col_type: ColumnType::String },
                Column { name: "count".into(), col_type: ColumnType::U64 },
                Column { name: "total_size".into(), col_type: ColumnType::U64 },
            ],
        }
    }
}

/// Search hit result
#[derive(Clone, Debug)]
pub struct SearchHit {
    pub file_path: String,
    pub line_number: u32,
    pub content: String,
}

/// Group aggregation result
#[derive(Clone, Debug)]
pub struct GroupRow {
    pub key: String,
    pub count: u64,
    pub total_size: u64,
}

/// Value flowing through a pipeline
#[derive(Clone, Debug)]
pub enum Value {
    /// File listing result
    Files {
        rows: Vec<FileRow>,
        path_buffer: Vec<u8>,
        source_path: String,
    },

    /// Search results
    SearchResults {
        hits: Vec<SearchHit>,
        pattern: String,
    },

    /// Grouped aggregation
    Groups {
        rows: Vec<GroupRow>,
        group_field: String,
    },

    /// Scalar count
    Count(u64),

    /// Scalar sum
    Sum(u64),

    /// Text content (for cat)
    Text(String),

    /// No result
    Void,
}

impl Value {
    /// Get row count if applicable
    pub fn len(&self) -> Option<usize> {
        match self {
            Value::Files { rows, .. } => Some(rows.len()),
            Value::SearchResults { hits, .. } => Some(hits.len()),
            Value::Groups { rows, .. } => Some(rows.len()),
            _ => None,
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len().map(|l| l == 0).unwrap_or(true)
    }

    /// Get as file rows
    pub fn as_files(&self) -> Option<(&Vec<FileRow>, &Vec<u8>)> {
        match self {
            Value::Files { rows, path_buffer, .. } => Some((rows, path_buffer)),
            _ => None,
        }
    }

    /// Get schema for this value type
    pub fn schema(&self) -> Schema {
        match self {
            Value::Files { .. } => Schema::file_schema(),
            Value::SearchResults { .. } => Schema::search_schema(),
            Value::Groups { .. } => Schema::group_schema(),
            _ => Schema { columns: vec![] },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Count(n) => write!(f, "{}", n),
            Value::Sum(n) => write!(f, "{}", n),
            Value::Text(s) => write!(f, "{}", s),
            Value::Void => write!(f, "(void)"),
            Value::Files { rows, .. } => write!(f, "{} files", rows.len()),
            Value::SearchResults { hits, .. } => write!(f, "{} matches", hits.len()),
            Value::Groups { rows, .. } => write!(f, "{} groups", rows.len()),
        }
    }
}

/// Format byte size for display
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format timestamp for display
pub fn format_time(timestamp: u64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH, Duration};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();

    let diff = now.saturating_sub(timestamp);

    if diff < 60 {
        "just now".to_string()
    } else if diff < 3600 {
        format!("{} min ago", diff / 60)
    } else if diff < 86400 {
        format!("{} hours ago", diff / 3600)
    } else if diff < 86400 * 7 {
        format!("{} days ago", diff / 86400)
    } else if diff < 86400 * 30 {
        format!("{} weeks ago", diff / (86400 * 7))
    } else if diff < 86400 * 365 {
        format!("{} months ago", diff / (86400 * 30))
    } else {
        format!("{} years ago", diff / (86400 * 365))
    }
}
