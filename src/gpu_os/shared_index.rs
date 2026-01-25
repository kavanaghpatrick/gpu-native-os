//! Issue #135: Shared GPU-Resident Filesystem Index
//!
//! A unified filesystem index that all GPU tools share.
//! Lives at `~/.gpu_os/index/` and loads in <10ms via mmap.
//!
//! # Architecture
//!
//! ```text
//! ~/.gpu_os/
//! ├── index/
//! │   ├── home.idx          # ~/  (user files)
//! │   └── manifest.json     # Index metadata
//! └── config.toml           # User preferences (future)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rust_experiment::gpu_os::shared_index::GpuFilesystemIndex;
//!
//! let device = Device::system_default().unwrap();
//! let fs_index = GpuFilesystemIndex::load_or_create(&device)?;
//!
//! // Search across all indexes
//! let results = fs_index.search(".rs", 100);
//!
//! // Get home index directly
//! if let Some(home) = fs_index.home() {
//!     println!("Home index has {} entries", home.entry_count());
//! }
//! ```

use crate::gpu_os::gpu_index::{GpuResidentIndex, IndexError};
use crate::gpu_os::batch_io::GpuBatchLoader;
use chrono::{DateTime, Utc};
use metal::Device;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

/// Default patterns to exclude from indexing
pub const DEFAULT_EXCLUDES: &[&str] = &[
    // Version control
    ".git",
    ".hg",
    ".svn",
    // Build artifacts
    "target",
    "build",
    "dist",
    "out",
    "bin",
    "obj",
    // Dependencies
    "node_modules",
    "vendor",
    ".cargo",
    "venv",
    ".venv",
    "env",
    // Caches
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    // IDE
    ".idea",
    ".vscode",
    ".vs",
    // macOS
    ".DS_Store",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    // Temporary
    "tmp",
    "temp",
    ".tmp",
];

/// Index manifest version
const MANIFEST_VERSION: u32 = 1;

/// Search result from unified search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Which index this result came from
    pub index_name: String,
    /// Full path to the file/directory
    pub path: String,
    /// Search score (higher = better match)
    pub score: f32,
}

/// Information about a single index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// Name of this index (e.g., "home", "system")
    pub name: String,
    /// Path to the index file
    pub path: PathBuf,
    /// Root directory that was indexed
    pub root: PathBuf,
    /// Number of entries in the index
    pub entry_count: u32,
    /// Size of the index file in bytes
    pub size_bytes: u64,
    /// When the index was built
    pub built_at: DateTime<Utc>,
    /// Patterns that were excluded
    pub exclude_patterns: Vec<String>,
}

/// Manifest tracking all indexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    /// Manifest format version
    pub version: u32,
    /// List of all indexes
    pub indexes: Vec<IndexInfo>,
    /// When the manifest was last updated
    pub last_updated: DateTime<Utc>,
}

impl IndexManifest {
    /// Create a new manifest with the given indexes
    pub fn new(indexes: Vec<IndexInfo>) -> Self {
        Self {
            version: MANIFEST_VERSION,
            indexes,
            last_updated: Utc::now(),
        }
    }

    /// Load manifest from a file
    pub fn load(path: &Path) -> Result<Self, IndexError> {
        let content = fs::read_to_string(path)
            .map_err(IndexError::IoError)?;
        serde_json::from_str(&content)
            .map_err(|e| IndexError::InvalidFormat(e.to_string()))
    }

    /// Save manifest to a file
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| IndexError::InvalidFormat(e.to_string()))?;
        fs::write(path, content)
            .map_err(IndexError::IoError)
    }
}

/// Shared GPU-resident filesystem index
///
/// Manages multiple indexes (home, system, etc.) that are shared
/// across all GPU filesystem tools.
pub struct GpuFilesystemIndex {
    /// Metal device
    device: Device,
    /// Loaded indexes by name
    indexes: HashMap<String, GpuResidentIndex>,
    /// Index manifest
    manifest: IndexManifest,
    /// Base directory (~/.gpu_os)
    base_dir: PathBuf,
}

impl GpuFilesystemIndex {
    /// Load or create the shared index at the default location (~/.gpu_os)
    pub fn load_or_create(device: &Device) -> Result<Self, IndexError> {
        let home = dirs::home_dir()
            .ok_or_else(|| IndexError::InvalidFormat("Could not find home directory".into()))?;
        let base_dir = home.join(".gpu_os");
        Self::load_or_create_at(device, &base_dir)
    }

    /// Load or create the shared index at a specific location
    pub fn load_or_create_at(device: &Device, base_dir: &Path) -> Result<Self, IndexError> {
        let index_dir = base_dir.join("index");
        let manifest_path = index_dir.join("manifest.json");

        // Create directory structure if needed
        if !index_dir.exists() {
            fs::create_dir_all(&index_dir)
                .map_err(IndexError::IoError)?;
        }

        // Load or create manifest
        let manifest = if manifest_path.exists() {
            IndexManifest::load(&manifest_path)?
        } else {
            // First run: create default manifest
            // If base_dir is ~/.gpu_os, index home directory
            // Otherwise, index the base_dir's parent (for custom locations/tests)
            let default_home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let default_gpu_os = default_home.join(".gpu_os");

            let (index_root, index_name) = if base_dir == default_gpu_os {
                // Standard location - index home directory
                (default_home.clone(), "home".to_string())
            } else {
                // Custom location (e.g., tests) - index the base_dir itself
                (base_dir.to_path_buf(), "local".to_string())
            };

            let index_file_path = index_dir.join(format!("{}.idx", index_name));

            // Build initial index
            eprintln!("[shared_index] Building index for {} -> {}", index_root.display(), index_file_path.display());
            Self::build_index_internal(&index_root, &index_file_path)?;

            let size_bytes = fs::metadata(&index_file_path)
                .map(|m| m.len())
                .unwrap_or(0);

            // Load to get entry count
            let temp_index = GpuResidentIndex::load(device, &index_file_path)?;
            let entry_count = temp_index.entry_count() as u32;

            eprintln!("[shared_index] Index built: {} entries, {:.1} MB",
                entry_count, size_bytes as f64 / (1024.0 * 1024.0));

            let manifest = IndexManifest::new(vec![IndexInfo {
                name: index_name,
                path: index_file_path,
                root: index_root,
                entry_count,
                size_bytes,
                built_at: Utc::now(),
                exclude_patterns: DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect(),
            }]);

            manifest.save(&manifest_path)?;
            manifest
        };

        // Load all indexes
        let mut indexes = HashMap::new();
        for info in &manifest.indexes {
            if info.path.exists() {
                match GpuResidentIndex::load_smart(device, &info.path) {
                    Ok(index) => {
                        indexes.insert(info.name.clone(), index);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load index '{}': {}", info.name, e);
                    }
                }
            }
        }

        Ok(Self {
            device: device.clone(),
            indexes,
            manifest,
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Build an index from a directory (internal helper)
    fn build_index_internal(root: &Path, output: &Path) -> Result<(), IndexError> {
        // Use fast `find` + GPU approach for large directories
        build_index_fast(root, output, DEFAULT_EXCLUDES)
    }

    /// Get the home directory index
    pub fn home(&self) -> Option<&GpuResidentIndex> {
        self.indexes.get("home")
    }

    /// Get a specific index by name
    pub fn get(&self, name: &str) -> Option<&GpuResidentIndex> {
        self.indexes.get(name)
    }

    /// Get the GPU buffer for a specific index
    ///
    /// Returns the Metal buffer containing path entries for GPU search.
    /// Use with GpuPathSearch or GpuContentSearch for GPU-accelerated search.
    pub fn get_buffer(&self, name: &str) -> Option<&metal::Buffer> {
        self.indexes.get(name).map(|idx| idx.entries_buffer())
    }

    /// Get all GPU buffers for parallel search across indexes
    pub fn all_buffers(&self) -> Vec<(&str, &metal::Buffer, u32)> {
        self.indexes.iter()
            .map(|(name, idx)| (name.as_str(), idx.entries_buffer(), idx.entry_count()))
            .collect()
    }

    /// Rebuild a specific index
    pub fn rebuild(&mut self, name: &str) -> Result<(), IndexError> {
        let info = self.manifest.indexes.iter()
            .find(|i| i.name == name)
            .ok_or_else(|| IndexError::InvalidFormat(format!("Index '{}' not found", name)))?
            .clone();

        // Rebuild the index file
        Self::build_index_internal(&info.root, &info.path)?;

        // Reload
        let index = GpuResidentIndex::load_smart(&self.device, &info.path)?;

        // Update manifest
        let size_bytes = fs::metadata(&info.path).map(|m| m.len()).unwrap_or(0);
        let entry_count = index.entry_count() as u32;

        if let Some(manifest_info) = self.manifest.indexes.iter_mut().find(|i| i.name == name) {
            manifest_info.entry_count = entry_count;
            manifest_info.size_bytes = size_bytes;
            manifest_info.built_at = Utc::now();
        }
        self.manifest.last_updated = Utc::now();

        // Save manifest
        let manifest_path = self.base_dir.join("index/manifest.json");
        self.manifest.save(&manifest_path)?;

        // Update loaded index
        self.indexes.insert(name.to_string(), index);

        Ok(())
    }

    /// Rebuild all indexes
    pub fn rebuild_all(&mut self) -> Result<(), IndexError> {
        let names: Vec<String> = self.manifest.indexes.iter()
            .map(|i| i.name.clone())
            .collect();

        for name in names {
            self.rebuild(&name)?;
        }

        Ok(())
    }

    /// Check if any index is stale (older than threshold)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        let now = Utc::now();

        for info in &self.manifest.indexes {
            let age = now.signed_duration_since(info.built_at);
            if age.to_std().unwrap_or(Duration::MAX) > max_age {
                return true;
            }
        }

        false
    }

    /// Get total entry count across all indexes
    pub fn total_entries(&self) -> usize {
        self.indexes.values().map(|i| i.entry_count() as usize).sum()
    }

    /// Get total memory usage across all indexes
    pub fn memory_usage(&self) -> usize {
        self.indexes.values().map(|i| i.memory_usage()).sum()
    }

    /// Get the manifest
    pub fn manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Add a new index for a directory
    pub fn add_index(&mut self, name: &str, root: &Path) -> Result<(), IndexError> {
        let index_dir = self.base_dir.join("index");
        let index_path = index_dir.join(format!("{}.idx", name));

        // Build the index
        Self::build_index_internal(root, &index_path)?;

        // Load it
        let index = GpuResidentIndex::load_smart(&self.device, &index_path)?;

        let size_bytes = fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
        let entry_count = index.entry_count() as u32;

        // Add to manifest
        self.manifest.indexes.push(IndexInfo {
            name: name.to_string(),
            path: index_path,
            root: root.to_path_buf(),
            entry_count,
            size_bytes,
            built_at: Utc::now(),
            exclude_patterns: DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect(),
        });
        self.manifest.last_updated = Utc::now();

        // Save manifest
        let manifest_path = self.base_dir.join("index/manifest.json");
        self.manifest.save(&manifest_path)?;

        // Store index
        self.indexes.insert(name.to_string(), index);

        Ok(())
    }

    /// Remove an index
    pub fn remove_index(&mut self, name: &str) -> Result<(), IndexError> {
        // Remove from indexes
        self.indexes.remove(name);

        // Find and remove from manifest
        if let Some(pos) = self.manifest.indexes.iter().position(|i| i.name == name) {
            let info = self.manifest.indexes.remove(pos);

            // Delete the index file
            if info.path.exists() {
                let _ = fs::remove_file(&info.path);
            }
        }

        self.manifest.last_updated = Utc::now();

        // Save manifest
        let manifest_path = self.base_dir.join("index/manifest.json");
        self.manifest.save(&manifest_path)
    }
}

// Add the build_and_save_with_excludes method to GpuResidentIndex
// This is a trait extension to avoid modifying the original file too much
impl GpuResidentIndex {
    /// Build and save an index with custom exclude patterns
    pub fn build_and_save_with_excludes(
        root: impl AsRef<Path>,
        output: impl AsRef<Path>,
        progress: Option<&dyn Fn(usize)>,
        excludes: &[&str],
    ) -> Result<(), IndexError> {
        Self::build_and_save_with_excludes_verbose(root, output, progress, excludes, false)
    }

    /// Build and save an index with verbose progress output
    pub fn build_and_save_with_excludes_verbose(
        root: impl AsRef<Path>,
        output: impl AsRef<Path>,
        progress: Option<&dyn Fn(usize)>,
        excludes: &[&str],
        verbose: bool,
    ) -> Result<(), IndexError> {
        use std::io::Write;

        let root = root.as_ref();
        let output = output.as_ref();

        eprintln!("[index] Starting scan of {}", root.display());
        eprintln!("[index] Excludes: {:?}", &excludes[..excludes.len().min(5)]);
        let scan_start = std::time::Instant::now();

        // Use parallel scanning for large directories
        let paths = collect_paths_parallel(root, excludes, 50);

        let scan_time = scan_start.elapsed();
        eprintln!("[index] Scan complete: {} paths in {:.2}s ({:.0} paths/sec)",
            paths.len(), scan_time.as_secs_f64(),
            paths.len() as f64 / scan_time.as_secs_f64());

        if let Some(cb) = progress {
            cb(paths.len());
        }

        // Write index file
        eprintln!("[index] Writing to {}", output.display());
        let write_start = std::time::Instant::now();

        let mut file = fs::File::create(output)
            .map_err(IndexError::IoError)?;

        // Write header (4KB page)
        let mut header = vec![0u8; 4096];
        let magic = 0x47505549u32; // "GPUI"
        let version = 1u32;  // Must match INDEX_VERSION in gpu_index.rs
        let entry_count = paths.len() as u32;

        header[0..4].copy_from_slice(&magic.to_le_bytes());
        header[4..8].copy_from_slice(&version.to_le_bytes());
        header[8..12].copy_from_slice(&entry_count.to_le_bytes());

        file.write_all(&header)
            .map_err(IndexError::IoError)?;

        // Write entries (256 bytes each in GpuPathEntry format)
        const GPU_PATH_MAX_LEN: usize = 224;
        const FLAG_IS_DIR: u16 = 1;

        for (i, path) in paths.iter().enumerate() {
            let mut entry = vec![0u8; 256];
            let path_bytes = path.as_bytes();
            let path_len = path_bytes.len().min(GPU_PATH_MAX_LEN);

            // path: [u8; 224] at offset 0
            entry[..path_len].copy_from_slice(&path_bytes[..path_len]);

            // path_len: u16 at offset 224
            entry[224..226].copy_from_slice(&(path_len as u16).to_le_bytes());

            // flags: u16 at offset 226
            let is_dir = Path::new(path).is_dir();
            let flags: u16 = if is_dir { FLAG_IS_DIR } else { 0 };
            entry[226..228].copy_from_slice(&flags.to_le_bytes());

            // parent_idx: u32 at offset 228
            entry[228..232].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());

            file.write_all(&entry)
                .map_err(IndexError::IoError)?;

            // Progress for large writes
            if verbose && i > 0 && i % 50000 == 0 {
                eprintln!("[index] Written {}/{} entries...", i, paths.len());
            }
        }

        let write_time = write_start.elapsed();
        let total_size = 4096 + paths.len() * 256;
        eprintln!("[index] Write complete: {:.1} MB in {:.2}s",
            total_size as f64 / (1024.0 * 1024.0),
            write_time.as_secs_f64());

        Ok(())
    }
}

/// Progress tracker for index building
pub struct IndexBuildProgress {
    pub files_found: usize,
    pub dirs_scanned: usize,
    pub dirs_skipped: usize,
    pub current_dir: String,
    pub start_time: std::time::Instant,
}

impl IndexBuildProgress {
    pub fn new() -> Self {
        Self {
            files_found: 0,
            dirs_scanned: 0,
            dirs_skipped: 0,
            current_dir: String::new(),
            start_time: std::time::Instant::now(),
        }
    }

    pub fn print_status(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.files_found as f64 / elapsed;
        eprintln!("[{:.1}s] {} files, {} dirs scanned, {} skipped ({:.0} files/sec) - {}",
            elapsed,
            self.files_found,
            self.dirs_scanned,
            self.dirs_skipped,
            rate,
            if self.current_dir.len() > 50 {
                format!("...{}", &self.current_dir[self.current_dir.len()-47..])
            } else {
                self.current_dir.clone()
            }
        );
    }
}

/// Recursively collect paths, respecting exclude patterns (with progress)
fn collect_paths_recursive(
    dir: &Path,
    paths: &mut Vec<String>,
    excludes: &[&str],
    depth: usize,
    max_depth: usize,
) {
    collect_paths_with_progress(dir, paths, excludes, depth, max_depth, &mut None);
}

/// Recursively collect paths with optional progress tracking (single-threaded)
pub fn collect_paths_with_progress(
    dir: &Path,
    paths: &mut Vec<String>,
    excludes: &[&str],
    depth: usize,
    max_depth: usize,
    progress: &mut Option<IndexBuildProgress>,
) {
    if depth > max_depth {
        return;
    }

    // Update progress
    if let Some(ref mut p) = progress {
        p.dirs_scanned += 1;
        p.current_dir = dir.to_string_lossy().to_string();

        // Print every 1000 dirs
        if p.dirs_scanned % 1000 == 0 {
            p.print_status();
        }
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            if let Some(ref mut p) = progress {
                if p.dirs_scanned < 100 {
                    eprintln!("  [skip] {}: {}", dir.display(), e);
                }
            }
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip excluded patterns
        if excludes.iter().any(|ex| name == *ex) || name.starts_with('.') {
            if let Some(ref mut p) = progress {
                if path.is_dir() {
                    p.dirs_skipped += 1;
                }
            }
            continue;
        }

        let path_str = path.to_string_lossy().to_string();
        paths.push(path_str);

        if let Some(ref mut p) = progress {
            p.files_found += 1;
        }

        if path.is_dir() {
            collect_paths_with_progress(&path, paths, excludes, depth + 1, max_depth, progress);
        }
    }
}

/// PARALLEL directory scanning using Rayon
/// Scans top-level directories in parallel for massive speedup
pub fn collect_paths_parallel(
    root: &Path,
    excludes: &[&str],
    max_depth: usize,
) -> Vec<String> {
    let files_found = AtomicUsize::new(0);
    let dirs_scanned = AtomicUsize::new(0);
    let start = std::time::Instant::now();

    // Get top-level entries
    let top_entries: Vec<PathBuf> = match fs::read_dir(root) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                !excludes.iter().any(|ex| name == *ex) && !name.starts_with('.')
            })
            .collect(),
        Err(e) => {
            eprintln!("[parallel] Cannot read {}: {}", root.display(), e);
            return Vec::new();
        }
    };

    eprintln!("[parallel] Scanning {} top-level entries with {} threads...",
        top_entries.len(), rayon::current_num_threads());

    // Scan each top-level entry in parallel
    let all_paths: Vec<Vec<String>> = top_entries
        .par_iter()
        .map(|entry| {
            let mut local_paths = Vec::new();
            let path_str = entry.to_string_lossy().to_string();
            local_paths.push(path_str);

            if entry.is_dir() {
                collect_paths_recursive_fast(entry, &mut local_paths, excludes, 1, max_depth, &files_found, &dirs_scanned);
            }

            // Progress update every ~5000 files
            let count = files_found.load(Ordering::Relaxed);
            if count % 5000 == 0 && count > 0 {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("[parallel] {:.1}s: {} files, {} dirs ({:.0} files/sec)",
                    elapsed, count, dirs_scanned.load(Ordering::Relaxed),
                    count as f64 / elapsed);
            }

            local_paths
        })
        .collect();

    // Flatten results
    let paths: Vec<String> = all_paths.into_iter().flatten().collect();

    let elapsed = start.elapsed();
    eprintln!("[parallel] Done: {} files in {:.2}s ({:.0} files/sec)",
        paths.len(), elapsed.as_secs_f64(),
        paths.len() as f64 / elapsed.as_secs_f64());

    paths
}

/// Build index using fast `find` command + parallel processing
/// This is MUCH faster than recursive read_dir for large directories
pub fn build_index_fast(root: &Path, output: &Path, excludes: &[&str]) -> Result<(), IndexError> {
    let start = std::time::Instant::now();
    eprintln!("[fast-index] Building index for {} using optimized find...", root.display());

    // Build find command with excludes
    // find is highly optimized for directory traversal
    let mut cmd = Command::new("find");
    cmd.arg(root);

    // Add exclude patterns
    for (i, exclude) in excludes.iter().enumerate() {
        if i > 0 {
            cmd.arg("-o");
        }
        cmd.arg("-name").arg(exclude).arg("-prune");
    }
    cmd.arg("-o").arg("-print");

    // Also exclude hidden files
    cmd.arg("!").arg("-name").arg(".*");

    let find_start = std::time::Instant::now();
    eprintln!("[fast-index] Running find command...");

    let output_result = Command::new("find")
        .arg(root)
        .arg("-not").arg("-path").arg("*/\\.*")  // Skip hidden
        .arg("-not").arg("-path").arg("*/node_modules/*")
        .arg("-not").arg("-path").arg("*/target/*")
        .arg("-not").arg("-path").arg("*/.git/*")
        .arg("-not").arg("-path").arg("*/.cache/*")
        .arg("-not").arg("-path").arg("*/venv/*")
        .arg("-not").arg("-path").arg("*/__pycache__/*")
        .output()
        .map_err(|e| IndexError::IoError(e))?;

    let find_time = find_start.elapsed();

    if !output_result.status.success() {
        eprintln!("[fast-index] find command had errors (continuing with partial results)");
    }

    let paths_data = String::from_utf8_lossy(&output_result.stdout);
    let paths: Vec<&str> = paths_data.lines().collect();

    eprintln!("[fast-index] find complete: {} paths in {:.2}s ({:.0} paths/sec)",
        paths.len(), find_time.as_secs_f64(),
        paths.len() as f64 / find_time.as_secs_f64());

    // Write index file (this part is fast - just sequential writes)
    let write_start = std::time::Instant::now();
    eprintln!("[fast-index] Writing index to {}...", output.display());

    let mut file = fs::File::create(output)
        .map_err(IndexError::IoError)?;

    // Write header (4KB page)
    let mut header = vec![0u8; 4096];
    let magic = 0x47505549u32; // "GPUI"
    let version = 1u32;
    let entry_count = paths.len() as u32;

    header[0..4].copy_from_slice(&magic.to_le_bytes());
    header[4..8].copy_from_slice(&version.to_le_bytes());
    header[8..12].copy_from_slice(&entry_count.to_le_bytes());

    file.write_all(&header).map_err(IndexError::IoError)?;

    // Write entries in parallel chunks for speed
    // GpuPathEntry format (256 bytes total):
    //   path: [u8; 224]      - 224 bytes
    //   path_len: u16        - 2 bytes
    //   flags: u16           - 2 bytes (FLAG_IS_DIR = 1)
    //   parent_idx: u32      - 4 bytes (use 0xFFFFFFFF for all)
    //   size: u64            - 8 bytes
    //   mtime: u64           - 8 bytes
    //   _reserved: [u8; 8]   - 8 bytes
    const GPU_PATH_MAX_LEN: usize = 224;
    const FLAG_IS_DIR: u16 = 1;

    let entries_size = paths.len() * 256;
    let mut entries_buffer = vec![0u8; entries_size];

    // Fill entries in parallel using rayon
    entries_buffer.par_chunks_mut(256)
        .zip(paths.par_iter())
        .for_each(|(entry, path)| {
            let path_bytes = path.as_bytes();
            let path_len = path_bytes.len().min(GPU_PATH_MAX_LEN);

            // path: [u8; 224] at offset 0
            entry[..path_len].copy_from_slice(&path_bytes[..path_len]);

            // path_len: u16 at offset 224
            entry[224..226].copy_from_slice(&(path_len as u16).to_le_bytes());

            // flags: u16 at offset 226
            let is_dir = Path::new(path).is_dir();
            let flags: u16 = if is_dir { FLAG_IS_DIR } else { 0 };
            entry[226..228].copy_from_slice(&flags.to_le_bytes());

            // parent_idx: u32 at offset 228 (use 0xFFFFFFFF = no parent)
            entry[228..232].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());

            // size, mtime, reserved: all zeros (already zeroed)
        });

    // Write all entries at once
    file.write_all(&entries_buffer).map_err(IndexError::IoError)?;

    let write_time = write_start.elapsed();
    let total_size = 4096 + entries_size;

    eprintln!("[fast-index] Write complete: {:.1} MB in {:.2}s ({:.0} MB/s)",
        total_size as f64 / (1024.0 * 1024.0),
        write_time.as_secs_f64(),
        (total_size as f64 / (1024.0 * 1024.0)) / write_time.as_secs_f64());

    let total_time = start.elapsed();
    eprintln!("[fast-index] Total: {} paths indexed in {:.2}s", paths.len(), total_time.as_secs_f64());

    Ok(())
}

/// Fast recursive collection (no progress struct, just atomics)
fn collect_paths_recursive_fast(
    dir: &Path,
    paths: &mut Vec<String>,
    excludes: &[&str],
    depth: usize,
    max_depth: usize,
    files_found: &AtomicUsize,
    dirs_scanned: &AtomicUsize,
) {
    if depth > max_depth {
        return;
    }

    dirs_scanned.fetch_add(1, Ordering::Relaxed);

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip excluded patterns
        if excludes.iter().any(|ex| name == *ex) || name.starts_with('.') {
            continue;
        }

        let path_str = path.to_string_lossy().to_string();
        paths.push(path_str);
        files_found.fetch_add(1, Ordering::Relaxed);

        if path.is_dir() {
            collect_paths_recursive_fast(&path, paths, excludes, depth + 1, max_depth, files_found, dirs_scanned);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = IndexManifest::new(vec![
            IndexInfo {
                name: "test".to_string(),
                path: PathBuf::from("/tmp/test.idx"),
                root: PathBuf::from("/tmp"),
                entry_count: 100,
                size_bytes: 25600,
                built_at: Utc::now(),
                exclude_patterns: vec![".git".to_string()],
            }
        ]);

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: IndexManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, manifest.version);
        assert_eq!(parsed.indexes.len(), 1);
        assert_eq!(parsed.indexes[0].name, "test");
    }

    #[test]
    fn test_default_excludes() {
        assert!(DEFAULT_EXCLUDES.contains(&".git"));
        assert!(DEFAULT_EXCLUDES.contains(&"node_modules"));
        assert!(DEFAULT_EXCLUDES.contains(&"target"));
    }
}
