//! GPU Shell - PowerShell-style command line with GPU pipeline execution
//!
//! Commands pipe GPU buffers instead of text. Each pipeline stage is a GPU kernel,
//! data stays on GPU throughout execution.
//!
//! Example:
//! ```
//! gpu> files ~/code | where ext = "rs" | where size > 10KB | sort size desc | head 10
//! ```

pub mod value;
pub mod parser;
pub mod executor;
pub mod render;

use metal::Device;
use std::collections::HashMap;
use std::time::Instant;

use crate::gpu_os::shared_index::GpuFilesystemIndex;
use crate::gpu_os::gpu_index::GpuResidentIndex;

pub use value::{Value, FileRow, Schema, Column};
pub use parser::{Pipeline, Command, Predicate, PredicateOp, PredicateValue, ParseError};
pub use executor::ExecError;
pub use render::TableRenderer;

/// File metadata cache for warm queries
pub struct FileCache {
    pub path: String,
    pub entries: Vec<FileRow>,
    pub path_buffer: Vec<u8>,
    pub loaded_at: Instant,
    pub from_gpu_index: bool,
}

/// GPU Shell - interactive command line with GPU-accelerated pipelines
pub struct GpuShell {
    device: Device,

    // Cached data sources
    file_cache: HashMap<String, FileCache>,

    // GPU filesystem index (pre-built, fast loading)
    gpu_index: Option<GpuFilesystemIndex>,

    // Statistics
    pub last_query_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub index_hits: u64,
}

impl GpuShell {
    /// Create a new GPU Shell instance
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        // Try to load the GPU filesystem index
        let gpu_index = match GpuFilesystemIndex::load_or_create(&device) {
            Ok(idx) => {
                println!("Loaded GPU index: {} entries", idx.total_entries());
                Some(idx)
            }
            Err(e) => {
                eprintln!("Warning: Could not load GPU index: {:?}", e);
                eprintln!("Falling back to filesystem scan (slower)");
                None
            }
        };

        Ok(Self {
            device,
            file_cache: HashMap::new(),
            gpu_index,
            last_query_time_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            index_hits: 0,
        })
    }

    /// Execute a shell command string
    pub fn execute(&mut self, input: &str) -> Result<Value, String> {
        let start = Instant::now();

        // Parse the pipeline
        let pipeline = Pipeline::parse(input)
            .map_err(|e| format!("Parse error: {:?}", e))?;

        // Execute the pipeline
        let result = executor::execute(self, &pipeline)?;

        self.last_query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Get or load file cache for a path
    pub fn get_or_load_files(&mut self, path: &str) -> Result<&FileCache, String> {
        let canonical = self.canonicalize_path(path);

        if !self.file_cache.contains_key(&canonical) {
            self.cache_misses += 1;

            // Try GPU index first for home directory
            let cache = if self.is_home_path(&canonical) {
                if let Some(ref gpu_index) = self.gpu_index {
                    if let Some(home_idx) = gpu_index.home() {
                        self.index_hits += 1;
                        self.load_from_gpu_index(home_idx, &canonical)?
                    } else {
                        self.load_files(&canonical)?
                    }
                } else {
                    self.load_files(&canonical)?
                }
            } else {
                self.load_files(&canonical)?
            };

            self.file_cache.insert(canonical.clone(), cache);
        } else {
            self.cache_hits += 1;
        }

        Ok(self.file_cache.get(&canonical).unwrap())
    }

    /// Check if path is under home directory
    fn is_home_path(&self, path: &str) -> bool {
        if let Some(home) = dirs::home_dir() {
            path.starts_with(&home.to_string_lossy().to_string())
        } else {
            false
        }
    }

    /// Load files from GPU index (fast path)
    fn load_from_gpu_index(&self, index: &GpuResidentIndex, filter_path: &str) -> Result<FileCache, String> {
        let mut entries = Vec::new();
        let mut path_buffer = Vec::new();

        let start = Instant::now();

        for entry in index.iter() {
            let entry_path = entry.path_str();

            // Filter to requested path
            if !entry_path.starts_with(filter_path) {
                continue;
            }

            let path_offset = path_buffer.len() as u32;
            let path_bytes = entry_path.as_bytes();
            path_buffer.extend_from_slice(path_bytes);

            // Extract extension
            let ext_str = std::path::Path::new(entry_path)
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let mut ext_bytes = [0u8; 8];
            for (i, b) in ext_str.bytes().take(8).enumerate() {
                ext_bytes[i] = b;
            }

            let ext_hash = {
                let mut hash = 0u32;
                for b in ext_str.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(b as u32);
                }
                hash
            };

            entries.push(FileRow {
                path_offset,
                path_len: path_bytes.len() as u16,
                size: entry.size,
                modified: entry.mtime,
                is_dir: entry.is_dir(),
                ext_hash,
                ext_bytes,
            });
        }

        let elapsed = start.elapsed();
        println!("GPU index load: {} entries in {:.1}ms", entries.len(), elapsed.as_secs_f64() * 1000.0);

        Ok(FileCache {
            path: filter_path.to_string(),
            entries,
            path_buffer,
            loaded_at: Instant::now(),
            from_gpu_index: true,
        })
    }

    /// Load files from filesystem into cache
    fn load_files(&mut self, path: &str) -> Result<FileCache, String> {
        use std::fs;
        use std::os::unix::fs::MetadataExt;

        let mut entries = Vec::new();
        let mut path_buffer = Vec::new();

        fn walk_dir(
            dir: &std::path::Path,
            entries: &mut Vec<FileRow>,
            path_buffer: &mut Vec<u8>,
        ) -> std::io::Result<()> {
            if let Ok(read_dir) = fs::read_dir(dir) {
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    if let Ok(meta) = entry.metadata() {
                        let path_str = path.to_string_lossy();
                        let path_offset = path_buffer.len() as u32;
                        let path_bytes = path_str.as_bytes();
                        path_buffer.extend_from_slice(path_bytes);

                        // Extract extension hash
                        let ext_hash = path.extension()
                            .map(|e| {
                                let ext = e.to_string_lossy().to_lowercase();
                                let mut hash = 0u32;
                                for b in ext.bytes() {
                                    hash = hash.wrapping_mul(31).wrapping_add(b as u32);
                                }
                                hash
                            })
                            .unwrap_or(0);

                        // Extract extension string for display
                        let ext_str = path.extension()
                            .map(|e| e.to_string_lossy().to_lowercase())
                            .unwrap_or_default();
                        let mut ext_bytes = [0u8; 8];
                        for (i, b) in ext_str.bytes().take(8).enumerate() {
                            ext_bytes[i] = b;
                        }

                        entries.push(FileRow {
                            path_offset,
                            path_len: path_bytes.len() as u16,
                            size: meta.len(),
                            modified: meta.mtime() as u64,
                            is_dir: meta.is_dir(),
                            ext_hash,
                            ext_bytes,
                        });

                        if meta.is_dir() && !path.to_string_lossy().contains("/.") {
                            let _ = walk_dir(&path, entries, path_buffer);
                        }
                    }
                }
            }
            Ok(())
        }

        let root = std::path::Path::new(path);
        walk_dir(root, &mut entries, &mut path_buffer)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        Ok(FileCache {
            path: path.to_string(),
            entries,
            path_buffer,
            loaded_at: Instant::now(),
            from_gpu_index: false,
        })
    }

    /// Canonicalize a path (expand ~, resolve relative paths)
    fn canonicalize_path(&self, path: &str) -> String {
        if path.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                return path.replacen('~', &home.to_string_lossy(), 1);
            }
        }

        if path.starts_with('/') {
            path.to_string()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path).to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string())
        }
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.file_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64, usize, u64) {
        (self.cache_hits, self.cache_misses, self.file_cache.len(), self.index_hits)
    }

    /// Check if GPU index is available
    pub fn has_gpu_index(&self) -> bool {
        self.gpu_index.is_some()
    }

    /// Get GPU index info
    pub fn gpu_index_info(&self) -> Option<(usize, usize)> {
        self.gpu_index.as_ref().map(|idx| (idx.total_entries(), idx.memory_usage()))
    }

    /// Rebuild the GPU index
    pub fn rebuild_index(&mut self) -> Result<(), String> {
        if let Some(ref mut idx) = self.gpu_index {
            idx.rebuild_all().map_err(|e| format!("Rebuild failed: {:?}", e))?;
            println!("Index rebuilt: {} entries", idx.total_entries());
        } else {
            return Err("No GPU index loaded".into());
        }
        Ok(())
    }
}
