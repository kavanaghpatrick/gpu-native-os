//! Tests for GPU Shell (Issue #126)

use rust_experiment::gpu_os::shell::{
    GpuShell, Pipeline, Command, Predicate, PredicateOp, PredicateValue,
    Value, TableRenderer,
};
use std::time::Instant;

// ============================================================================
// Parser Tests
// ============================================================================

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
    assert!(matches!(p.commands[0], Command::Files { .. }));
    assert!(matches!(p.commands[1], Command::Where { .. }));
    assert!(matches!(p.commands[2], Command::Sort { descending: true, .. }));
    assert!(matches!(p.commands[3], Command::Head { n: 10 }));
}

#[test]
fn test_parse_search_command() {
    let p = Pipeline::parse("search \"TODO\" ~/code").unwrap();
    assert_eq!(p.commands.len(), 1);
    if let Command::Search { pattern, path } = &p.commands[0] {
        assert_eq!(pattern, "TODO");
        assert_eq!(path.as_deref(), Some("~/code"));
    } else {
        panic!("Expected Search command");
    }
}

#[test]
fn test_parse_group_command() {
    let p = Pipeline::parse("files . | group ext | sort count desc").unwrap();
    assert_eq!(p.commands.len(), 3);
    assert!(matches!(p.commands[1], Command::Group { ref field } if field == "ext"));
}

#[test]
fn test_parse_predicate_size() {
    let pred = Predicate::parse("size > 1MB").unwrap();
    assert_eq!(pred.field, "size");
    assert_eq!(pred.op, PredicateOp::Gt);
    if let PredicateValue::Size(n) = pred.value {
        assert_eq!(n, 1024 * 1024);
    } else {
        panic!("Expected Size value");
    }
}

#[test]
fn test_parse_predicate_string() {
    let pred = Predicate::parse("ext = \"rs\"").unwrap();
    assert_eq!(pred.field, "ext");
    assert_eq!(pred.op, PredicateOp::Eq);
    if let PredicateValue::String(s) = pred.value {
        assert_eq!(s, "rs");
    } else {
        panic!("Expected String value");
    }
}

#[test]
fn test_parse_predicate_contains() {
    let pred = Predicate::parse("name ~= test").unwrap();
    assert_eq!(pred.field, "name");
    assert_eq!(pred.op, PredicateOp::Contains);
}

#[test]
fn test_parse_predicate_operators() {
    assert_eq!(Predicate::parse("size >= 100").unwrap().op, PredicateOp::Gte);
    assert_eq!(Predicate::parse("size <= 100").unwrap().op, PredicateOp::Lte);
    assert_eq!(Predicate::parse("ext != \"tmp\"").unwrap().op, PredicateOp::NotEq);
    assert_eq!(Predicate::parse("name ^= \"test\"").unwrap().op, PredicateOp::StartsWith);
    assert_eq!(Predicate::parse("name $= \".rs\"").unwrap().op, PredicateOp::EndsWith);
}

// ============================================================================
// Execution Tests
// ============================================================================

#[test]
fn test_execute_files() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp").expect("Execution failed");
    assert!(matches!(result, Value::Files { .. }));
}

#[test]
fn test_execute_count() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | count").expect("Execution failed");
    assert!(matches!(result, Value::Count(_)));
}

#[test]
fn test_execute_where_filter() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    // Get total count
    let all = shell.execute("files /tmp | count").expect("Execution failed");
    let all_count = if let Value::Count(n) = all { n } else { panic!("Expected count") };

    // Filter to only directories - should be <= total
    let filtered = shell.execute("files /tmp | where is_dir = true | count").expect("Execution failed");
    let filtered_count = if let Value::Count(n) = filtered { n } else { panic!("Expected count") };

    assert!(filtered_count <= all_count);
}

#[test]
fn test_execute_sort() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | sort size desc | head 5").expect("Execution failed");

    if let Value::Files { rows, .. } = result {
        // Verify descending order
        for i in 1..rows.len() {
            assert!(rows[i-1].size >= rows[i].size, "Not sorted in descending order");
        }
    } else {
        panic!("Expected Files result");
    }
}

#[test]
fn test_execute_group() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | group is_dir").expect("Execution failed");

    if let Value::Groups { rows, group_field } = result {
        assert_eq!(group_field, "is_dir");
        // Should have at most 2 groups: "file" and "directory"
        assert!(rows.len() <= 2);
    } else {
        panic!("Expected Groups result");
    }
}

#[test]
fn test_execute_head_tail() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    let head = shell.execute("files /tmp | head 3").expect("Execution failed");
    if let Value::Files { rows, .. } = head {
        assert!(rows.len() <= 3);
    }

    let tail = shell.execute("files /tmp | tail 3").expect("Execution failed");
    if let Value::Files { rows, .. } = tail {
        assert!(rows.len() <= 3);
    }
}

#[test]
fn test_execute_help() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("help").expect("Execution failed");
    if let Value::Text(help) = result {
        assert!(help.contains("DATA SOURCES"));
        assert!(help.contains("files"));
        assert!(help.contains("where"));
    } else {
        panic!("Expected help text");
    }
}

// ============================================================================
// Cache Tests
// ============================================================================

#[test]
fn test_cache_warm_queries() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    // Cold query
    let start = Instant::now();
    shell.execute("files /tmp").expect("Execution failed");
    let cold_time = start.elapsed();

    // Warm query (should use cache)
    let start = Instant::now();
    shell.execute("files /tmp | where is_dir = true").expect("Execution failed");
    let warm_time = start.elapsed();

    // Warm query should be faster (cache hit)
    let (hits, misses, _, _) = shell.cache_stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);

    // Note: warm_time might not always be faster due to system variance
    // but cache should definitely have been hit
    println!("Cold: {:?}, Warm: {:?}", cold_time, warm_time);
}

#[test]
fn test_cache_clear() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    shell.execute("files /tmp").expect("Execution failed");
    let (_, _, cached_before, _) = shell.cache_stats();
    assert!(cached_before > 0);

    shell.clear_cache();
    let (_, _, cached_after, _) = shell.cache_stats();
    assert_eq!(cached_after, 0);
}

// ============================================================================
// Render Tests
// ============================================================================

#[test]
fn test_render_files() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | head 5").expect("Execution failed");

    let renderer = TableRenderer::default();
    let output = renderer.render(&result, None);

    // Should contain table elements
    assert!(output.contains("path"));
    assert!(output.contains("size"));
    assert!(output.contains("modified"));
}

#[test]
fn test_render_count() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | count").expect("Execution failed");

    let renderer = TableRenderer::default();
    let output = renderer.render(&result, None);

    // Should be a number
    assert!(output.parse::<u64>().is_ok() || output.contains("0"));
}

#[test]
fn test_render_groups() {
    let mut shell = GpuShell::new().expect("Failed to create shell");
    let result = shell.execute("files /tmp | group is_dir").expect("Execution failed");

    let renderer = TableRenderer::default();
    let output = renderer.render(&result, None);

    assert!(output.contains("count"));
    assert!(output.contains("total_size"));
}

// ============================================================================
// Benchmark Tests
// ============================================================================

#[test]
#[ignore] // Run with: cargo test benchmark --release -- --ignored
fn benchmark_file_listing() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    // Warm up
    shell.execute("files ~").expect("Execution failed");

    // Benchmark
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        shell.execute("files ~ | where ext = \"rs\" | count").expect("Execution failed");
    }
    let elapsed = start.elapsed();

    println!("\n=== GPU Shell Benchmark ===");
    println!("Query: files ~ | where ext = \"rs\" | count");
    println!("Iterations: {}", iterations);
    println!("Total time: {:?}", elapsed);
    println!("Average: {:.2}ms per query", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
}

#[test]
#[ignore]
fn benchmark_complex_pipeline() {
    let mut shell = GpuShell::new().expect("Failed to create shell");

    // Warm up
    shell.execute("files ~").expect("Execution failed");

    // Complex pipeline
    let query = "files ~ | where size > 1KB | where ext = \"rs\" | sort size desc | head 20";

    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        shell.execute(query).expect("Execution failed");
    }
    let elapsed = start.elapsed();

    println!("\n=== Complex Pipeline Benchmark ===");
    println!("Query: {}", query);
    println!("Iterations: {}", iterations);
    println!("Total time: {:?}", elapsed);
    println!("Average: {:.2}ms per query", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
}
