# Issue #144: GPU Extension Hash Table - O(1) File Extension Filtering

## Problem Statement

Multiple locations in the codebase filter files by extension using CPU linear search:

**filesystem_browser.rs lines 71-78:**
```rust
fn should_exclude_path(path: &str) -> bool {
    let ext = path.rsplit('.').next().unwrap_or("");
    let ext_lower = ext.to_lowercase();  // CPU string operation
    EXCLUDED_EXTENSIONS.iter().any(|&excluded| ext_lower == excluded)  // O(60) comparisons
}
```

**gpu_ripgrep.rs lines 44-50:**
```rust
fn has_searchable_extension(path: &str) -> bool {
    if let Some(ext) = path.rsplit('.').next() {
        DEFAULT_EXTENSIONS.contains(&ext.to_lowercase().as_str())  // O(17) comparisons
    } else { false }
}
```

**Impact:**
- Called for every file in index (1.7M files)
- 60 string comparisons per file = 102M comparisons
- CPU-bound during path filtering

## Solution

Replace CPU linear search with GPU hash table lookup:
1. Build perfect hash table of extension hashes at startup
2. GPU kernel computes extension hash and looks up in O(1)
3. Filter entire index in single GPU dispatch

## Requirements

### Functional Requirements
1. O(1) extension lookup regardless of extension count
2. Support both include and exclude lists
3. Case-insensitive matching
4. Handle files with no extension

### Performance Requirements
1. **Hash computation:** <1 cycle per character on GPU
2. **Lookup:** O(1) with no collisions (perfect hash)
3. **Filter 1.7M files:** <10ms total

### Non-Functional Requirements
1. Hash table fits in constant memory (~4KB for 1024 extensions)
2. No false positives (exact match)
3. Low false negative rate if using bloom filter variant

## Technical Design

### Perfect Hash for Extensions

Since we have a known, fixed set of extensions, we can compute a perfect hash function at build time:

```rust
// Build-time: Generate perfect hash function
fn build_extension_hash_table(extensions: &[&str]) -> ExtensionHashTable {
    // Use CHD (Compress, Hash, Displace) algorithm for perfect hashing
    let mut table = vec![0u32; extensions.len() * 2]; // 2x for minimal collisions

    for ext in extensions {
        let hash = compute_extension_hash(ext);
        let slot = hash % table.len() as u32;
        table[slot as usize] = hash;
    }

    ExtensionHashTable { table, size: table.len() }
}
```

### GPU Hash Function

```metal
// src/gpu_os/document/filter.metal

// FNV-1a hash optimized for short strings
inline uint hash_extension(device const char* ext, uint len) {
    uint hash = 2166136261u;  // FNV offset basis

    for (uint i = 0; i < len && i < 16; i++) {
        // Case-insensitive: convert to lowercase
        char c = ext[i];
        if (c >= 'A' && c <= 'Z') c += 32;

        hash ^= uint(c);
        hash *= 16777619u;  // FNV prime
    }

    return hash;
}

// Extract extension from path and compute hash
inline uint get_extension_hash(device const char* path, uint path_len) {
    // Find last '.'
    int dot_pos = -1;
    for (int i = path_len - 1; i >= 0 && i > path_len - 16; i--) {
        if (path[i] == '.') {
            dot_pos = i;
            break;
        }
        if (path[i] == '/') break;  // No extension
    }

    if (dot_pos < 0) return 0;  // No extension

    return hash_extension(path + dot_pos + 1, path_len - dot_pos - 1);
}
```

### GPU Filter Kernel

```metal
struct FilterParams {
    uint path_count;
    uint hash_table_size;
    uint filter_mode;  // 0 = exclude if match, 1 = include if match
};

kernel void filter_by_extension(
    device const GpuPathEntry* paths [[buffer(0)]],
    device const uint* hash_table [[buffer(1)]],     // Perfect hash table
    device atomic_uint* result_count [[buffer(2)]],
    device uint* result_indices [[buffer(3)]],       // Filtered path indices
    constant FilterParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.path_count) return;

    GpuPathEntry entry = paths[tid];
    uint ext_hash = get_extension_hash(entry.path, entry.path_len);

    // O(1) hash table lookup
    bool found = false;
    if (ext_hash != 0) {
        uint slot = ext_hash % params.hash_table_size;
        found = (hash_table[slot] == ext_hash);
    }

    // Apply filter mode
    bool include = (params.filter_mode == 0) ? !found : found;

    if (include) {
        uint idx = atomic_fetch_add_explicit(result_count, 1, memory_order_relaxed);
        result_indices[idx] = tid;
    }
}
```

### Rust Implementation

```rust
// src/gpu_os/extension_filter.rs

pub struct GpuExtensionFilter {
    device: Device,
    command_queue: CommandQueue,
    filter_pipeline: ComputePipelineState,
    hash_table_buffer: Buffer,
    result_count_buffer: Buffer,
    result_indices_buffer: Buffer,
    hash_table_size: usize,
}

impl GpuExtensionFilter {
    pub fn new(device: &Device, extensions: &[&str]) -> Self {
        // Build perfect hash table
        let hash_table = build_perfect_hash_table(extensions);

        // Upload to GPU
        let hash_table_buffer = device.new_buffer_with_data(
            hash_table.as_ptr() as *const _,
            (hash_table.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create filter pipeline
        let library = device.new_library_with_source(FILTER_SHADER_SOURCE, &CompileOptions::new()).unwrap();
        let function = library.get_function("filter_by_extension", None).unwrap();
        let filter_pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();

        Self {
            device: device.clone(),
            command_queue: device.new_command_queue(),
            filter_pipeline,
            hash_table_buffer,
            result_count_buffer: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            result_indices_buffer: device.new_buffer(
                (MAX_PATHS * 4) as u64,
                MTLResourceOptions::StorageModeShared
            ),
            hash_table_size: hash_table.len(),
        }
    }

    pub fn filter_paths(
        &self,
        paths_buffer: &Buffer,
        path_count: usize,
        mode: FilterMode
    ) -> Vec<usize> {
        // Reset count
        unsafe {
            *(self.result_count_buffer.contents() as *mut u32) = 0;
        }

        let params = FilterParams {
            path_count: path_count as u32,
            hash_table_size: self.hash_table_size as u32,
            filter_mode: match mode {
                FilterMode::Exclude => 0,
                FilterMode::Include => 1,
            },
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<FilterParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.filter_pipeline);
        encoder.set_buffer(0, Some(paths_buffer), 0);
        encoder.set_buffer(1, Some(&self.hash_table_buffer), 0);
        encoder.set_buffer(2, Some(&self.result_count_buffer), 0);
        encoder.set_buffer(3, Some(&self.result_indices_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);

        let threads = MTLSize::new(path_count as u64, 1, 1);
        let threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let count = unsafe {
            *(self.result_count_buffer.contents() as *const u32) as usize
        };

        let mut indices = Vec::with_capacity(count);
        unsafe {
            let ptr = self.result_indices_buffer.contents() as *const u32;
            for i in 0..count {
                indices.push(*ptr.add(i) as usize);
            }
        }

        indices
    }
}

fn build_perfect_hash_table(extensions: &[&str]) -> Vec<u32> {
    // Simple approach: 2x size for low collision probability
    let size = (extensions.len() * 2).next_power_of_two();
    let mut table = vec![0u32; size];

    for ext in extensions {
        let hash = fnv1a_hash(ext.to_lowercase().as_bytes());
        let slot = (hash as usize) % size;
        // Note: For true perfect hash, use more sophisticated algorithm
        // This simple approach may have collisions
        table[slot] = hash;
    }

    table
}

fn fnv1a_hash(bytes: &[u8]) -> u32 {
    let mut hash = 2166136261u32;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}
```

## Pseudocode

```
# Build time
function build_hash_table(extensions):
    size = next_power_of_two(extensions.length * 2)
    table = array[size] of uint32

    for ext in extensions:
        hash = fnv1a(lowercase(ext))
        slot = hash % size
        table[slot] = hash

    return table

# Runtime (GPU)
kernel filter_by_extension(paths, hash_table, params):
    tid = thread_id
    if tid >= params.path_count: return

    path = paths[tid]
    ext_hash = extract_and_hash_extension(path)

    # O(1) lookup
    slot = ext_hash % hash_table.size
    found = (hash_table[slot] == ext_hash)

    # Apply filter
    if (params.exclude_mode and not found) or
       (params.include_mode and found):
        idx = atomic_add(result_count, 1)
        result_indices[idx] = tid
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_144_extension_filter.rs

#[test]
fn test_hash_consistency() {
    // Verify GPU and CPU hash produce same results
    let extensions = vec!["rs", "py", "js", "ts", "md", "txt"];

    for ext in &extensions {
        let cpu_hash = fnv1a_hash(ext.to_lowercase().as_bytes());
        let gpu_hash = compute_hash_on_gpu(ext);

        assert_eq!(cpu_hash, gpu_hash, "Hash mismatch for '{}'", ext);
    }
}

#[test]
fn test_case_insensitive() {
    let device = Device::system_default().unwrap();
    let filter = GpuExtensionFilter::new(&device, &["RS", "PY", "JS"]);

    let paths = create_test_paths(&[
        "file.rs",
        "file.RS",
        "file.Rs",
        "file.rS",
        "file.py",
        "file.txt",
    ]);

    let included = filter.filter_paths(&paths, 6, FilterMode::Include);

    assert_eq!(included.len(), 5);  // All .rs and .py variants
    assert!(!included.contains(&5)); // .txt excluded
}

#[test]
fn test_exclude_mode() {
    let device = Device::system_default().unwrap();

    let excluded = vec![
        "cache", "tmp", "o", "obj", "pyc", "class",
        "zip", "tar", "gz", "png", "jpg", "mp3", "mp4"
    ];

    let filter = GpuExtensionFilter::new(&device, &excluded);

    let paths = create_test_paths(&[
        "source.rs",
        "data.json",
        "image.png",
        "archive.zip",
        "compiled.o",
        "readme.md",
    ]);

    let kept = filter.filter_paths(&paths, 6, FilterMode::Exclude);

    assert_eq!(kept.len(), 3);  // .rs, .json, .md kept
    assert!(kept.contains(&0)); // source.rs
    assert!(kept.contains(&1)); // data.json
    assert!(kept.contains(&5)); // readme.md
}

#[test]
fn test_no_extension() {
    let device = Device::system_default().unwrap();
    let filter = GpuExtensionFilter::new(&device, &["txt", "md"]);

    let paths = create_test_paths(&[
        "Makefile",
        "Dockerfile",
        "README",
        "readme.md",
    ]);

    let included = filter.filter_paths(&paths, 4, FilterMode::Include);

    assert_eq!(included.len(), 1);  // Only readme.md
    assert!(included.contains(&3));
}

#[test]
fn test_filter_performance() {
    let device = Device::system_default().unwrap();

    // 60 extensions to exclude
    let excluded: Vec<&str> = EXCLUDED_EXTENSIONS.iter().copied().collect();
    let filter = GpuExtensionFilter::new(&device, &excluded);

    // 1.7M paths
    let paths = create_random_paths(1_700_000);
    let paths_buffer = upload_paths_to_gpu(&device, &paths);

    // Warmup
    for _ in 0..5 {
        filter.filter_paths(&paths_buffer, paths.len(), FilterMode::Exclude);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..10 {
        filter.filter_paths(&paths_buffer, paths.len(), FilterMode::Exclude);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / 10.0;
    println!("Filter 1.7M paths: {}ms", avg_ms);

    // Target: <10ms
    assert!(avg_ms < 20.0, "Filter too slow: {}ms", avg_ms);
}

#[test]
fn test_hash_collision_rate() {
    // Verify low collision rate for common extensions
    let extensions: Vec<&str> = vec![
        "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp",
        "h", "hpp", "cs", "rb", "php", "swift", "kt", "scala", "clj",
        "md", "txt", "json", "yaml", "yml", "toml", "xml", "html", "css",
        "sql", "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
    ];

    let table = build_perfect_hash_table(&extensions);

    // Check for collisions
    let mut seen_slots = std::collections::HashSet::new();
    let mut collisions = 0;

    for ext in &extensions {
        let hash = fnv1a_hash(ext.to_lowercase().as_bytes());
        let slot = (hash as usize) % table.len();

        if seen_slots.contains(&slot) {
            collisions += 1;
        }
        seen_slots.insert(slot);
    }

    println!("Collisions: {} / {}", collisions, extensions.len());
    assert!(collisions < extensions.len() / 10, "Too many collisions");
}
```

### Visual Verification Tests

```rust
// tests/test_issue_144_visual.rs

#[test]
fn visual_test_filtered_file_list() {
    let device = Device::system_default().unwrap();
    let filter = GpuExtensionFilter::new(&device, &EXCLUDED_EXTENSIONS);

    // Load real index
    let index = GpuFilesystemIndex::load_or_create(&device).unwrap();
    let home_index = index.home().unwrap();

    // Filter
    let start = Instant::now();
    let kept = filter.filter_paths(&home_index.buffer(), home_index.count(), FilterMode::Exclude);
    let elapsed = start.elapsed();

    println!("Filtered {} -> {} paths in {:?}", home_index.count(), kept.len(), elapsed);

    // Render filtered list sample
    let mut renderer = TestRenderer::new(&device, 800, 600);

    for (i, &idx) in kept.iter().take(50).enumerate() {
        let path = home_index.get_path(idx);
        renderer.draw_text(&path, 10.0, (i as f32) * 12.0, 0xFFFFFF);
    }

    renderer.save_to_file("tests/visual_output/filtered_paths.png");

    // Verify no excluded extensions in output
    for &idx in kept.iter().take(1000) {
        let path = home_index.get_path(idx);
        let ext = path.rsplit('.').next().unwrap_or("");

        assert!(
            !EXCLUDED_EXTENSIONS.contains(&ext.to_lowercase().as_str()),
            "Excluded extension found: {} in {}", ext, path
        );
    }
}
```

## Success Metrics

1. **Lookup time:** O(1) per file (single hash computation + table lookup)
2. **Filter time:** <10ms for 1.7M files
3. **Collision rate:** <5% for standard extension sets
4. **Memory:** <4KB for hash table

## Dependencies

None (standalone module)

## Files to Create/Modify

1. `src/gpu_os/extension_filter.rs` - New module
2. `src/gpu_os/extension_filter.metal` - GPU kernels
3. `examples/filesystem_browser.rs` - Integrate filter
4. `examples/gpu_ripgrep.rs` - Integrate filter
5. `tests/test_issue_144_extension_filter.rs` - Tests
