//! Pipeline executor for GPU Shell
//!
//! Executes parsed pipelines, chaining GPU operations.

use super::{GpuShell, FileCache};
use super::value::{Value, FileRow, GroupRow, SearchHit};
use super::parser::{Pipeline, Command, Predicate, PredicateOp, PredicateValue};

/// Execution error
#[derive(Debug)]
pub enum ExecError {
    TypeMismatch(String),
    InvalidField(String),
    IoError(String),
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::TypeMismatch(s) => write!(f, "Type mismatch: {}", s),
            ExecError::InvalidField(s) => write!(f, "Invalid field: {}", s),
            ExecError::IoError(s) => write!(f, "I/O error: {}", s),
        }
    }
}

/// Execute a pipeline
pub fn execute(shell: &mut GpuShell, pipeline: &Pipeline) -> Result<Value, String> {
    let mut current = Value::Void;

    for cmd in &pipeline.commands {
        current = execute_command(shell, cmd, current)?;
    }

    Ok(current)
}

/// Execute a single command
fn execute_command(shell: &mut GpuShell, cmd: &Command, input: Value) -> Result<Value, String> {
    match cmd {
        Command::Files { path } => {
            let cache = shell.get_or_load_files(path)?;
            Ok(Value::Files {
                rows: cache.entries.clone(),
                path_buffer: cache.path_buffer.clone(),
                source_path: cache.path.clone(),
            })
        }

        Command::Search { pattern, path } => {
            // If we have file input, search those files
            // Otherwise search the specified path or current directory
            let search_path = path.clone().unwrap_or_else(|| ".".to_string());

            // Simple grep implementation for now
            // TODO: Use GPU content search
            let hits = search_files(&search_path, pattern)?;

            Ok(Value::SearchResults {
                hits,
                pattern: pattern.clone(),
            })
        }

        Command::Where { predicate } => {
            match input {
                Value::Files { rows, path_buffer, source_path } => {
                    let filtered = filter_files(&rows, &path_buffer, predicate)?;
                    Ok(Value::Files {
                        rows: filtered,
                        path_buffer,
                        source_path,
                    })
                }
                Value::SearchResults { hits, pattern } => {
                    let filtered = filter_search_hits(&hits, predicate)?;
                    Ok(Value::SearchResults { hits: filtered, pattern })
                }
                _ => Err("where requires a data source (files, search)".into()),
            }
        }

        Command::Sort { field, descending } => {
            match input {
                Value::Files { mut rows, path_buffer, source_path } => {
                    sort_files(&mut rows, &path_buffer, field, *descending)?;
                    Ok(Value::Files { rows, path_buffer, source_path })
                }
                Value::Groups { mut rows, group_field } => {
                    sort_groups(&mut rows, field, *descending)?;
                    Ok(Value::Groups { rows, group_field })
                }
                _ => Err("sort requires a data source".into()),
            }
        }

        Command::Head { n } => {
            match input {
                Value::Files { rows, path_buffer, source_path } => {
                    let rows = rows.into_iter().take(*n).collect();
                    Ok(Value::Files { rows, path_buffer, source_path })
                }
                Value::SearchResults { hits, pattern } => {
                    let hits = hits.into_iter().take(*n).collect();
                    Ok(Value::SearchResults { hits, pattern })
                }
                Value::Groups { rows, group_field } => {
                    let rows = rows.into_iter().take(*n).collect();
                    Ok(Value::Groups { rows, group_field })
                }
                _ => Err("head requires a data source".into()),
            }
        }

        Command::Tail { n } => {
            match input {
                Value::Files { rows, path_buffer, source_path } => {
                    let len = rows.len();
                    let skip = len.saturating_sub(*n);
                    let rows = rows.into_iter().skip(skip).collect();
                    Ok(Value::Files { rows, path_buffer, source_path })
                }
                Value::SearchResults { hits, pattern } => {
                    let len = hits.len();
                    let skip = len.saturating_sub(*n);
                    let hits = hits.into_iter().skip(skip).collect();
                    Ok(Value::SearchResults { hits, pattern })
                }
                _ => Err("tail requires a data source".into()),
            }
        }

        Command::Group { field } => {
            match input {
                Value::Files { rows, path_buffer, .. } => {
                    let groups = group_files(&rows, &path_buffer, field)?;
                    Ok(Value::Groups {
                        rows: groups,
                        group_field: field.clone(),
                    })
                }
                _ => Err("group requires files input".into()),
            }
        }

        Command::Count => {
            let count = match &input {
                Value::Files { rows, .. } => rows.len() as u64,
                Value::SearchResults { hits, .. } => hits.len() as u64,
                Value::Groups { rows, .. } => rows.len() as u64,
                _ => 0,
            };
            Ok(Value::Count(count))
        }

        Command::Sum { field } => {
            match input {
                Value::Files { rows, .. } => {
                    let sum = sum_files(&rows, field)?;
                    Ok(Value::Sum(sum))
                }
                Value::Groups { rows, .. } => {
                    let sum: u64 = rows.iter().map(|r| r.total_size).sum();
                    Ok(Value::Sum(sum))
                }
                _ => Err("sum requires files or groups".into()),
            }
        }

        Command::Cat { path } => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            Ok(Value::Text(content))
        }

        Command::Help => {
            Ok(Value::Text(HELP_TEXT.to_string()))
        }

        Command::Select { fields } => {
            // For now, just pass through (projection would filter columns in display)
            Ok(input)
        }

        Command::Unique => {
            // Deduplicate based on path
            match input {
                Value::Files { rows, path_buffer, source_path } => {
                    // Simple dedup by path hash
                    let mut seen = std::collections::HashSet::new();
                    let rows: Vec<_> = rows.into_iter()
                        .filter(|r| seen.insert((r.path_offset, r.path_len)))
                        .collect();
                    Ok(Value::Files { rows, path_buffer, source_path })
                }
                _ => Ok(input),
            }
        }

        Command::Dups { path } => {
            // TODO: Integrate with duplicate_finder.rs
            Err("dups command not yet implemented".into())
        }
    }
}

/// Filter files by predicate
fn filter_files(rows: &[FileRow], path_buffer: &[u8], pred: &Predicate) -> Result<Vec<FileRow>, String> {
    let mut result = Vec::new();

    for row in rows {
        if matches_predicate(row, path_buffer, pred)? {
            result.push(row.clone());
        }
    }

    Ok(result)
}

/// Check if a row matches a predicate
fn matches_predicate(row: &FileRow, path_buffer: &[u8], pred: &Predicate) -> Result<bool, String> {
    let field = pred.field.to_lowercase();

    match field.as_str() {
        "size" => {
            let val = row.size;
            match (&pred.op, &pred.value) {
                (PredicateOp::Gt, PredicateValue::Size(n)) => Ok(val > *n),
                (PredicateOp::Gt, PredicateValue::Number(n)) => Ok(val > *n),
                (PredicateOp::Gte, PredicateValue::Size(n)) => Ok(val >= *n),
                (PredicateOp::Gte, PredicateValue::Number(n)) => Ok(val >= *n),
                (PredicateOp::Lt, PredicateValue::Size(n)) => Ok(val < *n),
                (PredicateOp::Lt, PredicateValue::Number(n)) => Ok(val < *n),
                (PredicateOp::Lte, PredicateValue::Size(n)) => Ok(val <= *n),
                (PredicateOp::Lte, PredicateValue::Number(n)) => Ok(val <= *n),
                (PredicateOp::Eq, PredicateValue::Size(n)) => Ok(val == *n),
                (PredicateOp::Eq, PredicateValue::Number(n)) => Ok(val == *n),
                _ => Err(format!("Invalid operator for size field")),
            }
        }

        "modified" => {
            let val = row.modified;
            match (&pred.op, &pred.value) {
                (PredicateOp::Gt, PredicateValue::Duration(n)) => Ok(val > *n),
                (PredicateOp::Gte, PredicateValue::Duration(n)) => Ok(val >= *n),
                (PredicateOp::Lt, PredicateValue::Duration(n)) => Ok(val < *n),
                (PredicateOp::Lte, PredicateValue::Duration(n)) => Ok(val <= *n),
                _ => Err("modified requires duration value (e.g., 'yesterday', '1 hour ago')".into()),
            }
        }

        "ext" | "extension" => {
            let ext = row.ext();
            match (&pred.op, &pred.value) {
                (PredicateOp::Eq, PredicateValue::String(s)) => Ok(ext.eq_ignore_ascii_case(s)),
                (PredicateOp::NotEq, PredicateValue::String(s)) => Ok(!ext.eq_ignore_ascii_case(s)),
                (PredicateOp::Contains, PredicateValue::String(s)) => {
                    Ok(ext.to_lowercase().contains(&s.to_lowercase()))
                }
                _ => Err("Invalid operator for ext field".into()),
            }
        }

        "name" | "path" => {
            let start = row.path_offset as usize;
            let end = start + row.path_len as usize;
            let path = String::from_utf8_lossy(&path_buffer[start..end]);

            // For "name", extract just the filename
            let val = if field == "name" {
                std::path::Path::new(path.as_ref())
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.to_string())
            } else {
                path.to_string()
            };

            match (&pred.op, &pred.value) {
                (PredicateOp::Eq, PredicateValue::String(s)) => Ok(val == *s),
                (PredicateOp::NotEq, PredicateValue::String(s)) => Ok(val != *s),
                (PredicateOp::Contains, PredicateValue::String(s)) => {
                    Ok(val.to_lowercase().contains(&s.to_lowercase()))
                }
                (PredicateOp::StartsWith, PredicateValue::String(s)) => {
                    Ok(val.to_lowercase().starts_with(&s.to_lowercase()))
                }
                (PredicateOp::EndsWith, PredicateValue::String(s)) => {
                    Ok(val.to_lowercase().ends_with(&s.to_lowercase()))
                }
                _ => Err("Invalid operator for name/path field".into()),
            }
        }

        "is_dir" | "isdir" | "dir" => {
            match (&pred.op, &pred.value) {
                (PredicateOp::Eq, PredicateValue::Bool(b)) => Ok(row.is_dir == *b),
                (PredicateOp::Eq, PredicateValue::String(s)) => {
                    Ok(row.is_dir == s.eq_ignore_ascii_case("true"))
                }
                _ => Err("is_dir requires boolean value".into()),
            }
        }

        _ => Err(format!("Unknown field: {}", field)),
    }
}

/// Sort files by field
fn sort_files(rows: &mut [FileRow], path_buffer: &[u8], field: &str, desc: bool) -> Result<(), String> {
    match field.to_lowercase().as_str() {
        "size" => {
            rows.sort_by(|a, b| {
                let cmp = a.size.cmp(&b.size);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        "modified" => {
            rows.sort_by(|a, b| {
                let cmp = a.modified.cmp(&b.modified);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        "name" | "path" => {
            rows.sort_by(|a, b| {
                let path_a = get_path(a, path_buffer);
                let path_b = get_path(b, path_buffer);
                let cmp = path_a.cmp(&path_b);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        "ext" | "extension" => {
            rows.sort_by(|a, b| {
                let cmp = a.ext().cmp(&b.ext());
                if desc { cmp.reverse() } else { cmp }
            });
        }
        _ => return Err(format!("Cannot sort by field: {}", field)),
    }
    Ok(())
}

/// Sort groups by field
fn sort_groups(rows: &mut [GroupRow], field: &str, desc: bool) -> Result<(), String> {
    match field.to_lowercase().as_str() {
        "count" => {
            rows.sort_by(|a, b| {
                let cmp = a.count.cmp(&b.count);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        "total_size" | "size" => {
            rows.sort_by(|a, b| {
                let cmp = a.total_size.cmp(&b.total_size);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        "key" | "name" => {
            rows.sort_by(|a, b| {
                let cmp = a.key.cmp(&b.key);
                if desc { cmp.reverse() } else { cmp }
            });
        }
        _ => return Err(format!("Cannot sort groups by: {}", field)),
    }
    Ok(())
}

/// Group files by field
fn group_files(rows: &[FileRow], path_buffer: &[u8], field: &str) -> Result<Vec<GroupRow>, String> {
    use std::collections::HashMap;

    let mut groups: HashMap<String, (u64, u64)> = HashMap::new();

    for row in rows {
        let key = match field.to_lowercase().as_str() {
            "ext" | "extension" => {
                let ext = row.ext();
                if ext.is_empty() { "(no extension)".to_string() } else { ext }
            }
            "dir" | "directory" => {
                let path = get_path(row, path_buffer);
                std::path::Path::new(&path)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/".to_string())
            }
            "is_dir" => {
                if row.is_dir { "directory".to_string() } else { "file".to_string() }
            }
            _ => return Err(format!("Cannot group by: {}", field)),
        };

        let entry = groups.entry(key).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += row.size;
    }

    let mut result: Vec<GroupRow> = groups.into_iter()
        .map(|(key, (count, total_size))| GroupRow { key, count, total_size })
        .collect();

    // Default sort by count descending
    result.sort_by(|a, b| b.count.cmp(&a.count));

    Ok(result)
}

/// Sum a field across files
fn sum_files(rows: &[FileRow], field: &str) -> Result<u64, String> {
    match field.to_lowercase().as_str() {
        "size" => Ok(rows.iter().map(|r| r.size).sum()),
        "count" => Ok(rows.len() as u64),
        _ => Err(format!("Cannot sum field: {}", field)),
    }
}

/// Get path string from a file row
fn get_path(row: &FileRow, path_buffer: &[u8]) -> String {
    let start = row.path_offset as usize;
    let end = start + row.path_len as usize;
    String::from_utf8_lossy(&path_buffer[start..end]).to_string()
}

/// Search files for a pattern (simple implementation)
fn search_files(path: &str, pattern: &str) -> Result<Vec<SearchHit>, String> {
    use std::fs;
    use std::io::{BufRead, BufReader};

    let mut hits = Vec::new();
    let pattern_lower = pattern.to_lowercase();

    fn search_dir(
        dir: &std::path::Path,
        pattern: &str,
        hits: &mut Vec<SearchHit>,
    ) -> std::io::Result<()> {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if !path.to_string_lossy().contains("/.") {
                        let _ = search_dir(&path, pattern, hits);
                    }
                } else if path.is_file() {
                    // Skip binary files
                    let ext = path.extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .unwrap_or_default();

                    let text_exts = ["rs", "txt", "md", "js", "ts", "py", "go", "c", "h", "cpp", "hpp", "java", "json", "yaml", "yml", "toml", "xml", "html", "css", "sh", "sql"];

                    if text_exts.contains(&ext.as_str()) || ext.is_empty() {
                        if let Ok(file) = fs::File::open(&path) {
                            let reader = BufReader::new(file);
                            for (line_num, line) in reader.lines().enumerate() {
                                if let Ok(line) = line {
                                    if line.to_lowercase().contains(pattern) {
                                        hits.push(SearchHit {
                                            file_path: path.to_string_lossy().to_string(),
                                            line_number: (line_num + 1) as u32,
                                            content: line.trim().to_string(),
                                        });

                                        // Limit hits per file
                                        if hits.len() > 10000 {
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    let root = std::path::Path::new(path);
    search_dir(root, &pattern_lower, &mut hits)
        .map_err(|e| format!("Search failed: {}", e))?;

    Ok(hits)
}

/// Filter search hits by predicate
fn filter_search_hits(hits: &[SearchHit], pred: &Predicate) -> Result<Vec<SearchHit>, String> {
    let mut result = Vec::new();

    for hit in hits {
        let matches = match pred.field.to_lowercase().as_str() {
            "file" | "path" => {
                match (&pred.op, &pred.value) {
                    (PredicateOp::Contains, PredicateValue::String(s)) => {
                        hit.file_path.to_lowercase().contains(&s.to_lowercase())
                    }
                    (PredicateOp::Eq, PredicateValue::String(s)) => {
                        hit.file_path == *s
                    }
                    _ => true,
                }
            }
            "line" => {
                match (&pred.op, &pred.value) {
                    (PredicateOp::Contains, PredicateValue::String(s)) => {
                        hit.content.to_lowercase().contains(&s.to_lowercase())
                    }
                    _ => true,
                }
            }
            _ => true,
        };

        if matches {
            result.push(hit.clone());
        }
    }

    Ok(result)
}

const HELP_TEXT: &str = r#"GPU Shell Commands

DATA SOURCES
  files [path]           List files recursively (default: current directory)
  search <pattern> [path] Search file contents
  dups [path]            Find duplicate files

TRANSFORMS
  where <predicate>      Filter rows
  sort <field> [desc]    Sort by field
  select <fields>        Select columns
  group <field>          Group by field
  head <n>               First n rows (default: 10)
  tail <n>               Last n rows (default: 10)
  unique                 Remove duplicates

AGGREGATIONS
  count                  Count rows
  sum <field>            Sum numeric field

ACTIONS
  cat <file>             Display file contents
  help                   Show this help

PREDICATES
  field = value          Equality
  field != value         Not equal
  field > value          Greater than
  field < value          Less than
  field >= value         Greater or equal
  field <= value         Less or equal
  field ~= value         Contains
  field ^= value         Starts with
  field $= value         Ends with

FIELDS (for files)
  path, name, size, modified, ext, is_dir

SIZE VALUES
  1KB, 1MB, 1GB, 100B

DURATION VALUES
  yesterday, today, now
  1 hour ago, 2 days ago, 1 week ago

EXAMPLES
  files ~/code | where ext = "rs" | count
  files ~ | where size > 100MB | sort size desc | head 10
  search "TODO" ~/code | where file ~= test
  files . | group ext | sort count desc
"#;
