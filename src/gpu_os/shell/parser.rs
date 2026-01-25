//! Pipeline parser for GPU Shell
//!
//! Parses commands like:
//!   files ~/code | where ext = "rs" | sort size desc | head 10

use std::str::FromStr;

/// A complete pipeline of commands
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub commands: Vec<Command>,
}

/// Individual command in a pipeline
#[derive(Debug, Clone)]
pub enum Command {
    // Data sources
    Files { path: String },
    Search { pattern: String, path: Option<String> },
    Dups { path: String },

    // Transforms
    Where { predicate: Predicate },
    Sort { field: String, descending: bool },
    Select { fields: Vec<String> },
    Group { field: String },
    Head { n: usize },
    Tail { n: usize },
    Unique,

    // Aggregations
    Count,
    Sum { field: String },

    // Actions
    Cat { path: String },
    Help,
}

/// Predicate for filtering
#[derive(Debug, Clone)]
pub struct Predicate {
    pub field: String,
    pub op: PredicateOp,
    pub value: PredicateValue,
}

/// Predicate operator
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateOp {
    Eq,       // =
    NotEq,    // !=
    Gt,       // >
    Lt,       // <
    Gte,      // >=
    Lte,      // <=
    Contains, // ~=
    StartsWith, // ^=
    EndsWith,   // $=
}

/// Value in a predicate
#[derive(Debug, Clone)]
pub enum PredicateValue {
    String(String),
    Number(u64),
    Size(u64),      // 1KB, 1MB, 1GB
    Duration(u64),  // "yesterday", "1 hour ago"
    Bool(bool),
}

/// Parse error
#[derive(Debug, Clone)]
pub enum ParseError {
    EmptyInput,
    UnknownCommand(String),
    MissingArgument(String),
    InvalidPredicate(String),
    InvalidValue(String),
}

impl Pipeline {
    /// Parse a pipeline string
    pub fn parse(input: &str) -> Result<Pipeline, ParseError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::EmptyInput);
        }

        let parts: Vec<&str> = input.split('|').map(|s| s.trim()).collect();
        let mut commands = Vec::new();

        for part in parts {
            let cmd = Command::parse(part)?;
            commands.push(cmd);
        }

        Ok(Pipeline { commands })
    }
}

impl Command {
    /// Parse a single command
    pub fn parse(input: &str) -> Result<Command, ParseError> {
        let tokens = tokenize(input);
        if tokens.is_empty() {
            return Err(ParseError::EmptyInput);
        }

        let cmd_name = tokens[0].to_lowercase();
        let args = &tokens[1..];

        match cmd_name.as_str() {
            "files" | "ls" => {
                let path = args.first().map(|s| s.to_string()).unwrap_or_else(|| ".".to_string());
                Ok(Command::Files { path })
            }

            "search" | "grep" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("pattern".into()));
                }
                let pattern = args[0].to_string();
                let path = args.get(1).map(|s| s.to_string());
                Ok(Command::Search { pattern, path })
            }

            "dups" | "duplicates" => {
                let path = args.first().map(|s| s.to_string()).unwrap_or_else(|| ".".to_string());
                Ok(Command::Dups { path })
            }

            "where" | "filter" => {
                let pred_str = args.join(" ");
                let predicate = Predicate::parse(&pred_str)?;
                Ok(Command::Where { predicate })
            }

            "sort" | "order" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("field".into()));
                }
                let field = args[0].to_string();
                let descending = args.get(1).map(|s| s.to_lowercase() == "desc").unwrap_or(false);
                Ok(Command::Sort { field, descending })
            }

            "select" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("fields".into()));
                }
                let fields: Vec<String> = args.iter()
                    .flat_map(|s| s.split(','))
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                Ok(Command::Select { fields })
            }

            "group" | "groupby" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("field".into()));
                }
                let field = args[0].to_string();
                Ok(Command::Group { field })
            }

            "head" | "first" => {
                let n = args.first()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                Ok(Command::Head { n })
            }

            "tail" | "last" => {
                let n = args.first()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                Ok(Command::Tail { n })
            }

            "unique" | "distinct" => {
                Ok(Command::Unique)
            }

            "count" => Ok(Command::Count),

            "sum" => {
                let field = args.first()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "size".to_string());
                Ok(Command::Sum { field })
            }

            "cat" | "show" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("path".into()));
                }
                Ok(Command::Cat { path: args[0].to_string() })
            }

            "help" | "?" => Ok(Command::Help),

            _ => Err(ParseError::UnknownCommand(cmd_name)),
        }
    }
}

impl Predicate {
    /// Parse a predicate string like "size > 1MB"
    pub fn parse(input: &str) -> Result<Predicate, ParseError> {
        let input = input.trim();

        // Find operator
        let (field, op, value_str) = if let Some(idx) = input.find(">=") {
            (&input[..idx], PredicateOp::Gte, &input[idx + 2..])
        } else if let Some(idx) = input.find("<=") {
            (&input[..idx], PredicateOp::Lte, &input[idx + 2..])
        } else if let Some(idx) = input.find("!=") {
            (&input[..idx], PredicateOp::NotEq, &input[idx + 2..])
        } else if let Some(idx) = input.find("~=") {
            (&input[..idx], PredicateOp::Contains, &input[idx + 2..])
        } else if let Some(idx) = input.find("^=") {
            (&input[..idx], PredicateOp::StartsWith, &input[idx + 2..])
        } else if let Some(idx) = input.find("$=") {
            (&input[..idx], PredicateOp::EndsWith, &input[idx + 2..])
        } else if let Some(idx) = input.find('>') {
            (&input[..idx], PredicateOp::Gt, &input[idx + 1..])
        } else if let Some(idx) = input.find('<') {
            (&input[..idx], PredicateOp::Lt, &input[idx + 1..])
        } else if let Some(idx) = input.find('=') {
            (&input[..idx], PredicateOp::Eq, &input[idx + 1..])
        } else {
            return Err(ParseError::InvalidPredicate(input.to_string()));
        };

        let field = field.trim().to_string();
        let value_str = value_str.trim();

        let value = PredicateValue::parse(value_str)?;

        Ok(Predicate { field, op, value })
    }
}

impl PredicateValue {
    /// Parse a value string
    pub fn parse(input: &str) -> Result<PredicateValue, ParseError> {
        let input = input.trim();

        // Quoted string
        if (input.starts_with('"') && input.ends_with('"')) ||
           (input.starts_with('\'') && input.ends_with('\'')) {
            return Ok(PredicateValue::String(input[1..input.len()-1].to_string()));
        }

        // Boolean
        if input.eq_ignore_ascii_case("true") {
            return Ok(PredicateValue::Bool(true));
        }
        if input.eq_ignore_ascii_case("false") {
            return Ok(PredicateValue::Bool(false));
        }

        // Size (1KB, 1MB, 1GB)
        if let Some(size) = parse_size(input) {
            return Ok(PredicateValue::Size(size));
        }

        // Duration ("yesterday", "1 hour ago", etc.)
        if let Some(duration) = parse_duration(input) {
            return Ok(PredicateValue::Duration(duration));
        }

        // Plain number
        if let Ok(n) = input.parse::<u64>() {
            return Ok(PredicateValue::Number(n));
        }

        // Treat as unquoted string (for things like ext = rs)
        Ok(PredicateValue::String(input.to_string()))
    }
}

/// Tokenize input, respecting quoted strings
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = '"';

    for c in input.chars() {
        if in_quotes {
            if c == quote_char {
                in_quotes = false;
                tokens.push(current.clone());
                current.clear();
            } else {
                current.push(c);
            }
        } else if c == '"' || c == '\'' {
            in_quotes = true;
            quote_char = c;
        } else if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Parse size string like "1KB", "100MB", "1.5GB"
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim().to_uppercase();

    let (num_str, multiplier) = if s.ends_with("GB") {
        (&s[..s.len()-2], 1024 * 1024 * 1024)
    } else if s.ends_with("MB") {
        (&s[..s.len()-2], 1024 * 1024)
    } else if s.ends_with("KB") {
        (&s[..s.len()-2], 1024)
    } else if s.ends_with('G') {
        (&s[..s.len()-1], 1024 * 1024 * 1024)
    } else if s.ends_with('M') {
        (&s[..s.len()-1], 1024 * 1024)
    } else if s.ends_with('K') {
        (&s[..s.len()-1], 1024)
    } else if s.ends_with('B') {
        (&s[..s.len()-1], 1)
    } else {
        return None;
    };

    num_str.trim().parse::<f64>().ok()
        .map(|n| (n * multiplier as f64) as u64)
}

/// Parse duration string like "yesterday", "1 hour ago", "today"
fn parse_duration(s: &str) -> Option<u64> {
    use std::time::{SystemTime, UNIX_EPOCH, Duration};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();

    let s = s.trim().to_lowercase();

    // Named durations
    match s.as_str() {
        "now" => return Some(now),
        "today" => return Some(now - (now % 86400)), // Start of today
        "yesterday" => return Some(now - 86400),
        _ => {}
    }

    // Parse "N unit ago" patterns
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() >= 2 {
        if let Ok(n) = parts[0].parse::<u64>() {
            let unit = parts[1].trim_end_matches('s'); // Remove trailing 's'
            let seconds = match unit {
                "second" | "sec" => n,
                "minute" | "min" => n * 60,
                "hour" | "hr" => n * 3600,
                "day" => n * 86400,
                "week" => n * 86400 * 7,
                "month" => n * 86400 * 30,
                "year" => n * 86400 * 365,
                _ => return None,
            };
            return Some(now - seconds);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_pipeline() {
        let p = Pipeline::parse("files ~/code").unwrap();
        assert_eq!(p.commands.len(), 1);
        assert!(matches!(p.commands[0], Command::Files { .. }));
    }

    #[test]
    fn test_parse_chained_pipeline() {
        let p = Pipeline::parse("files ~ | where size > 1MB | sort size desc | head 10").unwrap();
        assert_eq!(p.commands.len(), 4);
    }

    #[test]
    fn test_parse_predicate() {
        let p = Predicate::parse("size > 1MB").unwrap();
        assert_eq!(p.field, "size");
        assert_eq!(p.op, PredicateOp::Gt);
        assert!(matches!(p.value, PredicateValue::Size(1048576)));
    }

    #[test]
    fn test_parse_string_predicate() {
        let p = Predicate::parse("ext = \"rs\"").unwrap();
        assert_eq!(p.field, "ext");
        assert_eq!(p.op, PredicateOp::Eq);
        assert!(matches!(p.value, PredicateValue::String(ref s) if s == "rs"));
    }

    #[test]
    fn test_parse_contains_predicate() {
        let p = Predicate::parse("name ~= test").unwrap();
        assert_eq!(p.field, "name");
        assert_eq!(p.op, PredicateOp::Contains);
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("1KB"), Some(1024));
        assert_eq!(parse_size("1MB"), Some(1024 * 1024));
        assert_eq!(parse_size("1.5GB"), Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64));
    }
}
