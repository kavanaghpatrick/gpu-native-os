# Issue #135: Shared GPU-Resident Filesystem Index

## Problem Statement

Currently, each GPU filesystem tool rebuilds its own index:
- `gpu_ripgrep` → `/tmp/gpu_ripgrep_index.bin`
- `filesystem_browser` → builds in-memory each time
- `filesystem_content_search` → scans directory each time

This is wasteful because:
1. **Redundant I/O**: Each tool scans the same filesystem
2. **Startup latency**: Index build takes 200ms-30s depending on directory size
3. **Memory waste**: Multiple copies of the same data
4. **No persistence**: Index lost on reboot

## Solution: Shared GPU-Resident Filesystem Index

A single, persistent filesystem index that:
1. Lives at `~/.gpu_os/index/`
2. Is shared by all GPU filesystem tools
3. Loads in 0.3ms via mmap (GPU-resident)
4. Can be updated incrementally via FSEvents (macOS) or inotify (Linux)

## Architecture

```
~/.gpu_os/
├── index/
│   ├── home.idx          # ~/  (user files)
│   ├── system.idx        # /usr, /opt, etc.
│   └── manifest.json     # Index metadata
├── config.toml           # User preferences
└── cache/                # Temporary data
```

### Index Manifest (manifest.json)
```json
{
  "version": 1,
  "indexes": [
    {
      "name": "home",
      "path": "~/.gpu_os/index/home.idx",
      "root": "/Users/username",
      "entry_count": 150000,
      "size_bytes": 36864000,
      "built_at": "2024-01-25T10:30:00Z",
      "exclude_patterns": [".git", "node_modules", "target", ".cache"]
    }
  ],
  "last_updated": "2024-01-25T10:30:00Z"
}
```

## API Design

### Rust API

```rust
/// Shared filesystem index manager
pub struct GpuFilesystemIndex {
    device: Device,
    indexes: HashMap<String, GpuResidentIndex>,
    manifest: IndexManifest,
}

impl GpuFilesystemIndex {
    /// Load all indexes from ~/.gpu_os/index/
    /// Creates directory structure if it doesn't exist
    pub fn load_or_create(device: &Device) -> Result<Self, IndexError>;

    /// Get the home directory index (most common)
    pub fn home(&self) -> Option<&GpuResidentIndex>;

    /// Get a specific index by name
    pub fn get(&self, name: &str) -> Option<&GpuResidentIndex>;

    /// Search across all indexes
    pub fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult>;

    /// Rebuild a specific index
    pub fn rebuild(&mut self, name: &str) -> Result<(), IndexError>;

    /// Rebuild all indexes
    pub fn rebuild_all(&mut self) -> Result<(), IndexError>;

    /// Check if indexes are stale (older than threshold)
    pub fn is_stale(&self, max_age: Duration) -> bool;

    /// Get total entry count across all indexes
    pub fn total_entries(&self) -> usize;

    /// Get memory usage
    pub fn memory_usage(&self) -> usize;
}

/// Index manifest tracking all indexes
pub struct IndexManifest {
    pub version: u32,
    pub indexes: Vec<IndexInfo>,
    pub last_updated: DateTime<Utc>,
}

pub struct IndexInfo {
    pub name: String,
    pub path: PathBuf,
    pub root: PathBuf,
    pub entry_count: u32,
    pub size_bytes: u64,
    pub built_at: DateTime<Utc>,
    pub exclude_patterns: Vec<String>,
}
```

### CLI Commands

```bash
# Build/rebuild the shared index
gpu-index build              # Build home index
gpu-index build --all        # Build all indexes
gpu-index build --root /opt  # Build custom root

# Query the index
gpu-index status             # Show index stats
gpu-index search "pattern"   # Search across indexes

# Maintenance
gpu-index clean              # Remove stale indexes
gpu-index watch              # Start FSEvents watcher (daemon)
```

## Pseudocode

### Index Loading (Hot Path)

```rust
fn load_or_create(device: &Device) -> Result<GpuFilesystemIndex> {
    let gpu_os_dir = home_dir().join(".gpu_os");
    let index_dir = gpu_os_dir.join("index");

    // Create directory structure if needed
    if !index_dir.exists() {
        create_dir_all(&index_dir)?;

        // First run: build home index
        let home_index_path = index_dir.join("home.idx");
        GpuResidentIndex::build_and_save(home_dir(), &home_index_path, None)?;

        // Create manifest
        let manifest = IndexManifest::new(vec![
            IndexInfo {
                name: "home".into(),
                path: home_index_path,
                root: home_dir(),
                // ... other fields
            }
        ]);
        manifest.save(&index_dir.join("manifest.json"))?;
    }

    // Load manifest
    let manifest = IndexManifest::load(&index_dir.join("manifest.json"))?;

    // Load all indexes via mmap (instant, ~0.3ms each)
    let mut indexes = HashMap::new();
    for info in &manifest.indexes {
        let index = GpuResidentIndex::load_smart(device, &info.path)?;
        indexes.insert(info.name.clone(), index);
    }

    Ok(GpuFilesystemIndex { device, indexes, manifest })
}
```

### Unified Search

```rust
fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
    let mut all_results = Vec::new();

    // Search each index in parallel (GPU already parallelizes internally)
    for (name, index) in &self.indexes {
        let results = index.search(query, max_results);
        for (idx, score) in results {
            if let Some(path) = index.get_path(idx) {
                all_results.push(SearchResult {
                    index_name: name.clone(),
                    path: path.to_string(),
                    score,
                });
            }
        }
    }

    // Sort by score, take top N
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    all_results.truncate(max_results);
    all_results
}
```

### Integration with gpu_ripgrep

```rust
// Before (current)
let index_path = temp_dir().join("gpu_ripgrep_index.bin");
if rebuild || !index_path.exists() {
    GpuResidentIndex::build_and_save(&dir, &index_path, None)?;
}
let index = GpuResidentIndex::load_smart(&device, &index_path)?;

// After (shared index)
let fs_index = GpuFilesystemIndex::load_or_create(&device)?;

// Check if we need to rebuild (stale or missing)
if rebuild || fs_index.is_stale(Duration::from_secs(3600)) {
    fs_index.rebuild("home")?;
}

// Use the shared index
let index = fs_index.home().expect("Home index should exist");
```

## Default Exclude Patterns

```rust
const DEFAULT_EXCLUDES: &[&str] = &[
    // Version control
    ".git", ".hg", ".svn",

    // Build artifacts
    "target", "build", "dist", "out", "bin", "obj",

    // Dependencies
    "node_modules", "vendor", ".cargo", "venv", ".venv",

    // Caches
    ".cache", "__pycache__", ".pytest_cache", ".mypy_cache",

    // IDE
    ".idea", ".vscode", ".vs",

    // macOS
    ".DS_Store", ".Spotlight-V100", ".Trashes",

    // Large media (optional)
    // "*.mp4", "*.mov", "*.avi", "*.mkv",
];
```

## Tests

### Unit Tests

```rust
#[test]
fn test_index_directory_creation() {
    let temp = tempdir().unwrap();
    std::env::set_var("HOME", temp.path());

    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    assert!(temp.path().join(".gpu_os/index").exists());
    assert!(temp.path().join(".gpu_os/index/manifest.json").exists());
}

#[test]
fn test_shared_index_load_time() {
    let device = Device::system_default().unwrap();

    // First load (may build)
    let _ = GpuFilesystemIndex::load_or_create(&device).unwrap();

    // Second load (should be instant)
    let start = Instant::now();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();
    let load_time = start.elapsed();

    // Should load in under 10ms (mmap is instant)
    assert!(load_time < Duration::from_millis(10),
        "Index load took {:?}, expected < 10ms", load_time);
}

#[test]
fn test_search_across_indexes() {
    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    let results = fs_index.search(".rs", 100);

    // Should find Rust files
    assert!(!results.is_empty(), "Should find .rs files");
    assert!(results.iter().all(|r| r.path.ends_with(".rs")));
}

#[test]
fn test_stale_detection() {
    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    // Just built, shouldn't be stale
    assert!(!fs_index.is_stale(Duration::from_secs(60)));

    // With 0 duration, everything is stale
    assert!(fs_index.is_stale(Duration::from_secs(0)));
}

#[test]
fn test_exclude_patterns() {
    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    let results = fs_index.search("node_modules", 100);

    // Should NOT find node_modules (excluded by default)
    assert!(results.is_empty() ||
            !results.iter().any(|r| r.path.contains("/node_modules/")));
}

#[test]
fn test_memory_usage_reasonable() {
    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    let memory = fs_index.memory_usage();
    let entries = fs_index.total_entries();

    // Should be ~256 bytes per entry (path + metadata)
    let bytes_per_entry = memory as f64 / entries as f64;
    assert!(bytes_per_entry < 512.0,
        "Memory usage too high: {} bytes/entry", bytes_per_entry);
}
```

### Integration Tests

```rust
#[test]
fn test_gpu_ripgrep_uses_shared_index() {
    // Build shared index
    let device = Device::system_default().unwrap();
    let mut fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();
    fs_index.rebuild("home").unwrap();

    // Run gpu_ripgrep - should use shared index
    let output = Command::new("./target/release/examples/gpu_ripgrep")
        .args(&["fn", ".", "-m", "5"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show "Loaded cached index" not "Building index"
    assert!(stdout.contains("Loaded cached index") ||
            stdout.contains("shared index"),
            "Should use shared index");
}

#[test]
fn test_multiple_tools_share_index() {
    let device = Device::system_default().unwrap();

    // Load from tool 1
    let fs_index1 = GpuFilesystemIndex::load_or_create(&device).unwrap();
    let entries1 = fs_index1.total_entries();

    // Load from tool 2 (simulated)
    let fs_index2 = GpuFilesystemIndex::load_or_create(&device).unwrap();
    let entries2 = fs_index2.total_entries();

    // Should have same entry count (same index)
    assert_eq!(entries1, entries2);
}
```

### Benchmarks

```rust
#[test]
fn benchmark_shared_vs_per_tool_index() {
    let device = Device::system_default().unwrap();
    let iterations = 5;

    println!("\n=== Shared Index vs Per-Tool Index ===\n");

    // Benchmark: Load shared index
    let mut shared_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = GpuFilesystemIndex::load_or_create(&device).unwrap();
        shared_times.push(start.elapsed());
    }

    // Benchmark: Build per-tool index (simulates current behavior)
    let mut build_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let temp_path = tempfile::NamedTempFile::new().unwrap();
        GpuResidentIndex::build_and_save(".", temp_path.path(), None).unwrap();
        build_times.push(start.elapsed());
    }

    let shared_avg = shared_times.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                     / iterations as f64 * 1000.0;
    let build_avg = build_times.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                    / iterations as f64 * 1000.0;

    println!("Shared index load: {:.1}ms (average)", shared_avg);
    println!("Per-tool build:    {:.1}ms (average)", build_avg);
    println!("Speedup:           {:.0}x", build_avg / shared_avg);

    // Shared should be at least 10x faster
    assert!(build_avg / shared_avg > 10.0,
        "Expected >10x speedup, got {:.1}x", build_avg / shared_avg);
}

#[test]
fn benchmark_search_latency() {
    let device = Device::system_default().unwrap();
    let fs_index = GpuFilesystemIndex::load_or_create(&device).unwrap();

    let queries = ["fn", ".rs", "test", "config", "main"];
    let iterations = 100;

    println!("\n=== Search Latency ===\n");

    for query in queries {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fs_index.search(query, 100);
        }
        let total = start.elapsed();
        let per_search = total.as_secs_f64() * 1000.0 / iterations as f64;

        println!("  \"{}\": {:.2}ms per search", query, per_search);
    }
}
```

## Migration Plan

1. **Phase 1**: Create `GpuFilesystemIndex` module
2. **Phase 2**: Update `gpu_ripgrep` to use shared index
3. **Phase 3**: Update `filesystem_browser` to use shared index
4. **Phase 4**: Update `filesystem_content_search` to use shared index
5. **Phase 5**: Add CLI tool `gpu-index` for management

## Success Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First tool startup | 200-500ms (build) | 200-500ms (build) | Same |
| Second tool startup | 200-500ms (rebuild) | <10ms (load) | 50x |
| Memory (3 tools) | 3x index size | 1x index size | 3x |
| Index freshness | Per-session | Persistent | ∞ |

## Future Enhancements

1. **FSEvents Daemon**: Auto-update index when files change
2. **Incremental Updates**: Only reindex changed directories
3. **Index Compression**: LZ4 compression for smaller disk footprint
4. **Remote Index**: Sync index across machines via iCloud/Dropbox
5. **Index Sharding**: Split large indexes for faster parallel search
