// Issue #77: GPU-Resident Filesystem Index
//
// THE GPU IS THE COMPUTER. Filesystem index lives permanently in GPU memory.
// CPU never touches path data after initial load.
//
// Traditional: CPU scans → copies paths → GPU searches → repeat every search
// GPU-Resident: CPU scans ONCE → mmap index → GPU owns data forever
//
// Loading methods:
// 1. mmap (current) - OS maps file pages, GPU accesses via unified memory
// 2. GPU-direct - MTLIOCommandQueue loads directly into GPU buffer (bypasses CPU entirely)
//
// Uses #82 (MmapBuffer) for zero-copy file-to-GPU access.
// Uses #125 (GpuIOQueue) for GPU-direct file loading.

use metal::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::mem;

use super::mmap_buffer::{MmapBuffer, MmapError, PAGE_SIZE, align_to_page};
use super::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

// ============================================================================
// Constants
// ============================================================================

/// Maximum path length (matches existing STREAM_MAX_PATH_LEN)
pub const GPU_PATH_MAX_LEN: usize = 224;

/// Size of each index entry (cache-line aligned)
pub const GPU_ENTRY_SIZE: usize = 256;

/// Magic number for index file validation
pub const INDEX_MAGIC: u32 = 0x47505549; // "GPUI" in little-endian

/// Index file version
pub const INDEX_VERSION: u32 = 1;

// ============================================================================
// GPU Path Entry (256 bytes, cache-aligned)
// ============================================================================

/// A single filesystem entry stored in GPU-resident memory.
///
/// Fixed-width for efficient GPU access. Total size: 256 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPathEntry {
    /// Fixed-width path bytes (null-padded)
    pub path: [u8; GPU_PATH_MAX_LEN], // 224 bytes
    /// Actual path length
    pub path_len: u16,                 // 2 bytes
    /// Flags: is_dir, is_symlink, is_hidden, etc.
    pub flags: u16,                    // 2 bytes
    /// Index of parent directory entry (-1 for root)
    pub parent_idx: u32,               // 4 bytes
    /// File size in bytes
    pub size: u64,                     // 8 bytes
    /// Modification time (Unix timestamp)
    pub mtime: u64,                    // 8 bytes
    /// Reserved for future use (alignment padding)
    pub _reserved: [u8; 8],            // 8 bytes
}
// Total: 224 + 2 + 2 + 4 + 8 + 8 + 8 = 256 bytes

/// Entry flags
pub const FLAG_IS_DIR: u16 = 1 << 0;
pub const FLAG_IS_SYMLINK: u16 = 1 << 1;
pub const FLAG_IS_HIDDEN: u16 = 1 << 2;
pub const FLAG_IS_EXECUTABLE: u16 = 1 << 3;

impl GpuPathEntry {
    /// Create a new entry from a path string and metadata.
    pub fn new(path_str: &str, is_dir: bool, size: u64, mtime: u64) -> Self {
        let bytes = path_str.as_bytes();
        let len = bytes.len().min(GPU_PATH_MAX_LEN);

        let mut path = [0u8; GPU_PATH_MAX_LEN];
        path[..len].copy_from_slice(&bytes[..len]);

        let mut flags = 0u16;
        if is_dir {
            flags |= FLAG_IS_DIR;
        }
        // Check for hidden files (starts with . after last /)
        if let Some(name_start) = path_str.rfind('/') {
            if path_str.as_bytes().get(name_start + 1) == Some(&b'.') {
                flags |= FLAG_IS_HIDDEN;
            }
        } else if path_str.starts_with('.') {
            flags |= FLAG_IS_HIDDEN;
        }

        Self {
            path,
            path_len: len as u16,
            flags,
            parent_idx: u32::MAX, // Will be set during indexing
            size,
            mtime,
            _reserved: [0; 8],
        }
    }

    /// Get the path as a string slice.
    pub fn path_str(&self) -> &str {
        let len = self.path_len as usize;
        std::str::from_utf8(&self.path[..len]).unwrap_or("")
    }

    /// Check if this entry is a directory.
    #[inline]
    pub fn is_dir(&self) -> bool {
        self.flags & FLAG_IS_DIR != 0
    }

    /// Check if this entry is hidden.
    #[inline]
    pub fn is_hidden(&self) -> bool {
        self.flags & FLAG_IS_HIDDEN != 0
    }

    /// Convert to raw bytes for writing.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                GPU_ENTRY_SIZE,
            )
        }
    }
}

impl Default for GpuPathEntry {
    fn default() -> Self {
        Self {
            path: [0; GPU_PATH_MAX_LEN],
            path_len: 0,
            flags: 0,
            parent_idx: u32::MAX,
            size: 0,
            mtime: 0,
            _reserved: [0; 8],
        }
    }
}

// Verify size at compile time
const _: () = assert!(mem::size_of::<GpuPathEntry>() == GPU_ENTRY_SIZE);

// ============================================================================
// Index File Header
// ============================================================================

/// Header for the index file (page-aligned for mmap).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Version number
    pub version: u32,
    /// Number of entries
    pub entry_count: u32,
    /// Reserved flags
    pub flags: u32,
    /// Timestamp when index was built
    pub build_time: u64,
    /// Padding to align to 4096 bytes (page size)
    pub _padding: [u8; PAGE_SIZE - 24],
}

impl Default for IndexHeader {
    fn default() -> Self {
        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            entry_count: 0,
            flags: 0,
            build_time: 0,
            _padding: [0; PAGE_SIZE - 24],
        }
    }
}

const _: () = assert!(mem::size_of::<IndexHeader>() == PAGE_SIZE);

// ============================================================================
// Index Storage - Either mmap or GPU-direct loaded
// ============================================================================

/// Storage backend for the GPU-resident index.
///
/// - Mmap: CPU mmaps file, GPU accesses via unified memory (traditional)
/// - GpuDirect: MTLIOCommandQueue loads directly into GPU buffer (bypasses CPU)
enum IndexStorage {
    /// Traditional mmap-based storage (zero-copy via unified memory)
    Mmap(MmapBuffer),
    /// GPU-direct I/O storage (MTLIOCommandQueue, bypasses CPU entirely)
    GpuDirect {
        buffer: Buffer,
        file_size: usize,
    },
}

impl IndexStorage {
    /// Get the Metal buffer
    fn metal_buffer(&self) -> &Buffer {
        match self {
            IndexStorage::Mmap(mmap) => mmap.metal_buffer(),
            IndexStorage::GpuDirect { buffer, .. } => buffer,
        }
    }

    /// Get raw pointer to data
    fn as_ptr(&self) -> *const u8 {
        match self {
            IndexStorage::Mmap(mmap) => mmap.as_ptr(),
            IndexStorage::GpuDirect { buffer, .. } => buffer.contents() as *const u8,
        }
    }

    /// Get file size
    fn file_size(&self) -> usize {
        match self {
            IndexStorage::Mmap(mmap) => mmap.file_size(),
            IndexStorage::GpuDirect { file_size, .. } => *file_size,
        }
    }

    /// Get aligned buffer size
    fn aligned_size(&self) -> usize {
        match self {
            IndexStorage::Mmap(mmap) => mmap.aligned_size(),
            IndexStorage::GpuDirect { buffer, .. } => buffer.length() as usize,
        }
    }
}

// ============================================================================
// GPU-Resident Index
// ============================================================================

/// A filesystem index that lives permanently in GPU memory.
///
/// Supports two loading methods:
/// - `load()` - mmap + newBufferWithBytesNoCopy (zero-copy via unified memory)
/// - `load_gpu_direct()` - MTLIOCommandQueue (GPU loads file directly, CPU never touches bytes)
///
/// Once loaded, the GPU owns this data - no CPU copies ever.
///
/// # Example
/// ```ignore
/// // Build index once (slow, ~30s for 3M files)
/// GpuResidentIndex::build_and_save("/", "index.bin")?;
///
/// // Load via mmap (traditional zero-copy)
/// let index = GpuResidentIndex::load(&device, "index.bin")?;
///
/// // OR: Load via GPU-direct I/O (CPU never touches bytes)
/// let index = GpuResidentIndex::load_gpu_direct(&device, "index.bin")?;
///
/// // GPU now owns the data - search without copies
/// let results = search.search_with_index(&index, "query");
/// ```
pub struct GpuResidentIndex {
    /// Storage backend (mmap or GPU-direct)
    storage: IndexStorage,

    /// Number of entries in the index
    entry_count: u32,

    /// Offset to first entry (after header)
    entries_offset: usize,
}

impl GpuResidentIndex {
    /// Load an index from disk using zero-copy mmap.
    ///
    /// This is nearly instant regardless of index size - no data is copied.
    /// The GPU can immediately access all paths via the Metal buffer.
    pub fn load(device: &Device, path: impl AsRef<Path>) -> Result<Self, IndexError> {
        let mmap = MmapBuffer::from_file(device, path.as_ref())
            .map_err(IndexError::MmapError)?;

        // Validate header
        if mmap.file_size() < PAGE_SIZE {
            return Err(IndexError::InvalidFormat("File too small for header".into()));
        }

        let header = unsafe { &*(mmap.as_ptr() as *const IndexHeader) };

        if header.magic != INDEX_MAGIC {
            return Err(IndexError::InvalidFormat("Invalid magic number".into()));
        }

        if header.version != INDEX_VERSION {
            return Err(IndexError::InvalidFormat(format!(
                "Version mismatch: expected {}, got {}",
                INDEX_VERSION, header.version
            )));
        }

        let expected_size = PAGE_SIZE + (header.entry_count as usize * GPU_ENTRY_SIZE);
        if mmap.file_size() < expected_size {
            return Err(IndexError::InvalidFormat("File truncated".into()));
        }

        // Advise kernel we'll need this data
        mmap.advise_willneed();

        Ok(Self {
            storage: IndexStorage::Mmap(mmap),
            entry_count: header.entry_count,
            entries_offset: PAGE_SIZE,
        })
    }

    /// Load an index using GPU-direct I/O (MTLIOCommandQueue).
    ///
    /// **THE GPU IS THE COMPUTER**: This method bypasses CPU entirely.
    /// - Data flows: Disk → SSD Controller → GPU Memory
    /// - CPU never touches the bytes
    /// - Uses Metal 3 MTLIOCommandQueue for async file loading
    ///
    /// Best for: Large index files, cold cache scenarios, bandwidth-bound workloads.
    /// mmap may be faster for frequently-accessed hot data due to page caching.
    pub fn load_gpu_direct(device: &Device, path: impl AsRef<Path>) -> Result<Self, IndexError> {
        let path = path.as_ref();

        // Get file size
        let file_size = std::fs::metadata(path)
            .map_err(IndexError::IoError)?
            .len() as usize;

        if file_size < PAGE_SIZE {
            return Err(IndexError::InvalidFormat("File too small for header".into()));
        }

        // Create IO queue for GPU-direct loading
        let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent)
            .ok_or_else(|| IndexError::InvalidFormat("Failed to create GPU I/O queue".into()))?;

        // Open file for GPU I/O
        let file_handle = GpuIOFileHandle::open(device, path)
            .ok_or_else(|| IndexError::InvalidFormat("Failed to open file for GPU I/O".into()))?;

        // Allocate GPU buffer (page-aligned)
        let aligned_size = align_to_page(file_size);
        let buffer = device.new_buffer(
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create I/O command buffer
        let cmd_buffer = io_queue.command_buffer()
            .ok_or_else(|| IndexError::InvalidFormat("Failed to create I/O command buffer".into()))?;

        // Issue GPU-direct load command
        cmd_buffer.load_buffer(
            &buffer,
            0,               // Destination offset in buffer
            file_size as u64, // Size to load
            &file_handle,
            0,               // Source offset in file
        );

        // Commit and wait for GPU to complete load
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Validate header
        let header = unsafe { &*(buffer.contents() as *const IndexHeader) };

        if header.magic != INDEX_MAGIC {
            return Err(IndexError::InvalidFormat("Invalid magic number".into()));
        }

        if header.version != INDEX_VERSION {
            return Err(IndexError::InvalidFormat(format!(
                "Version mismatch: expected {}, got {}",
                INDEX_VERSION, header.version
            )));
        }

        let expected_size = PAGE_SIZE + (header.entry_count as usize * GPU_ENTRY_SIZE);
        if file_size < expected_size {
            return Err(IndexError::InvalidFormat("File truncated".into()));
        }

        Ok(Self {
            storage: IndexStorage::GpuDirect { buffer, file_size },
            entry_count: header.entry_count,
            entries_offset: PAGE_SIZE,
        })
    }

    /// Returns true if this index was loaded via GPU-direct I/O.
    pub fn is_gpu_direct(&self) -> bool {
        matches!(self.storage, IndexStorage::GpuDirect { .. })
    }

    /// Build an index from a directory tree and save to disk.
    ///
    /// This scans the filesystem (slow) and writes entries in GPU-friendly format.
    /// After building, use `load()` for instant mmap access.
    pub fn build_and_save(
        root: impl AsRef<Path>,
        output: impl AsRef<Path>,
        progress: Option<&dyn Fn(usize)>,
    ) -> Result<u32, IndexError> {
        let root = root.as_ref();
        let output = output.as_ref();

        // Collect all paths
        let mut entries = Vec::new();
        Self::scan_recursive(root, &mut entries, progress)?;

        let entry_count = entries.len() as u32;

        // Write to file
        let file = File::create(output).map_err(IndexError::IoError)?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header = IndexHeader {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            entry_count,
            flags: 0,
            build_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            _padding: [0; PAGE_SIZE - 24],
        };

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                PAGE_SIZE,
            )
        };
        writer.write_all(header_bytes).map_err(IndexError::IoError)?;

        // Write entries
        for entry in &entries {
            writer.write_all(entry.as_bytes()).map_err(IndexError::IoError)?;
        }

        // Pad to page boundary
        let total_size = PAGE_SIZE + entries.len() * GPU_ENTRY_SIZE;
        let aligned_size = align_to_page(total_size);
        let padding = aligned_size - total_size;
        if padding > 0 {
            writer.write_all(&vec![0u8; padding]).map_err(IndexError::IoError)?;
        }

        writer.flush().map_err(IndexError::IoError)?;

        Ok(entry_count)
    }

    /// Recursively scan a directory tree.
    fn scan_recursive(
        path: &Path,
        entries: &mut Vec<GpuPathEntry>,
        progress: Option<&dyn Fn(usize)>,
    ) -> Result<(), IndexError> {
        let dir = std::fs::read_dir(path).map_err(IndexError::IoError)?;

        for entry_result in dir {
            let entry = match entry_result {
                Ok(e) => e,
                Err(_) => continue, // Skip entries we can't read
            };

            let entry_path = entry.path();
            let path_str = entry_path.to_string_lossy();

            // Skip paths that are too long
            if path_str.len() > GPU_PATH_MAX_LEN {
                continue;
            }

            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue, // Skip entries we can't stat
            };

            let is_dir = metadata.is_dir();
            let size = if is_dir { 0 } else { metadata.len() };
            let mtime = metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            entries.push(GpuPathEntry::new(&path_str, is_dir, size, mtime));

            // Report progress
            if let Some(cb) = progress {
                if entries.len() % 10000 == 0 {
                    cb(entries.len());
                }
            }

            // Recurse into directories
            if is_dir {
                // Skip common uninteresting directories
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.')
                    && name_str != "node_modules"
                    && name_str != "target"
                    && name_str != "__pycache__"
                {
                    let _ = Self::scan_recursive(&entry_path, entries, progress);
                }
            }
        }

        Ok(())
    }

    /// Get the Metal buffer containing all path entries.
    ///
    /// Use this with GPU compute shaders for searching.
    #[inline]
    pub fn entries_buffer(&self) -> &Buffer {
        self.storage.metal_buffer()
    }

    /// Get the number of entries in the index.
    #[inline]
    pub fn entry_count(&self) -> u32 {
        self.entry_count
    }

    /// Get the offset (in bytes) to the first entry.
    #[inline]
    pub fn entries_offset(&self) -> usize {
        self.entries_offset
    }

    /// Get an entry by index (CPU-side access for debugging).
    pub fn get_entry(&self, index: usize) -> Option<&GpuPathEntry> {
        if index >= self.entry_count as usize {
            return None;
        }

        let offset = self.entries_offset + index * GPU_ENTRY_SIZE;
        if offset + GPU_ENTRY_SIZE > self.storage.file_size() {
            return None;
        }

        unsafe {
            let ptr = self.storage.as_ptr().add(offset) as *const GpuPathEntry;
            Some(&*ptr)
        }
    }

    /// Iterate over all entries (CPU-side access for debugging).
    pub fn iter(&self) -> impl Iterator<Item = &GpuPathEntry> {
        (0..self.entry_count as usize).filter_map(|i| self.get_entry(i))
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.storage.aligned_size()
    }

    // ========================================================================
    // GPU-Direct I/O Integration
    // ========================================================================
    //
    // These methods bridge the GPU-resident index to GPU batch loading.
    // Path resolution happens in GPU memory, then files load via MTLIOCommandQueue.

    /// Get file paths matching an extension filter (for GPU batch loading).
    ///
    /// This reads from the GPU-resident index (fast) and returns PathBufs
    /// that can be passed directly to GpuBatchLoader.
    ///
    /// # Example
    /// ```ignore
    /// let index = GpuResidentIndex::load(&device, "index.bin")?;
    /// let rust_files = index.files_with_extension("rs");
    /// let batch_result = loader.load_batch(&rust_files);
    /// ```
    pub fn files_with_extension(&self, ext: &str) -> Vec<std::path::PathBuf> {
        let ext_with_dot = format!(".{}", ext);
        self.iter()
            .filter(|e| !e.is_dir())
            .filter(|e| e.path_str().ends_with(&ext_with_dot))
            .map(|e| std::path::PathBuf::from(e.path_str()))
            .collect()
    }

    /// Get file paths matching any of the given extensions.
    pub fn files_with_extensions(&self, extensions: &[&str]) -> Vec<std::path::PathBuf> {
        self.iter()
            .filter(|e| !e.is_dir())
            .filter(|e| {
                let path = e.path_str();
                extensions.iter().any(|ext| path.ends_with(&format!(".{}", ext)))
            })
            .map(|e| std::path::PathBuf::from(e.path_str()))
            .collect()
    }

    /// Get all file paths (non-directories) from the index.
    pub fn all_files(&self) -> Vec<std::path::PathBuf> {
        self.iter()
            .filter(|e| !e.is_dir())
            .map(|e| std::path::PathBuf::from(e.path_str()))
            .collect()
    }

    /// Get files in a specific directory (non-recursive).
    pub fn files_in_directory(&self, dir: &str) -> Vec<std::path::PathBuf> {
        let dir_prefix = if dir.ends_with('/') {
            dir.to_string()
        } else {
            format!("{}/", dir)
        };

        self.iter()
            .filter(|e| !e.is_dir())
            .filter(|e| {
                let path = e.path_str();
                if !path.starts_with(&dir_prefix) {
                    return false;
                }
                // Check it's directly in this dir, not a subdirectory
                let remainder = &path[dir_prefix.len()..];
                !remainder.contains('/')
            })
            .map(|e| std::path::PathBuf::from(e.path_str()))
            .collect()
    }

    /// Get total size of files matching a filter (useful for progress estimation).
    pub fn total_size_of_files(&self, paths: &[std::path::PathBuf]) -> u64 {
        let path_set: std::collections::HashSet<&str> = paths
            .iter()
            .filter_map(|p| p.to_str())
            .collect();

        self.iter()
            .filter(|e| path_set.contains(e.path_str()))
            .map(|e| e.size)
            .sum()
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum IndexError {
    IoError(std::io::Error),
    MmapError(MmapError),
    InvalidFormat(String),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::IoError(e) => write!(f, "I/O error: {}", e),
            IndexError::MmapError(e) => write!(f, "mmap error: {}", e),
            IndexError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for IndexError {}

// ============================================================================
// GPU Shader Constants (for use in Metal shaders)
// ============================================================================

/// Metal shader header with GpuPathEntry struct definition.
pub const GPU_INDEX_SHADER_HEADER: &str = r#"
// GPU-Resident Index Entry (matches Rust GpuPathEntry)
struct GpuPathEntry {
    char path[224];       // Fixed-width path
    uint16_t path_len;    // Actual length
    uint16_t flags;       // is_dir, is_hidden, etc.
    uint32_t parent_idx;  // Parent directory index
    uint64_t size;        // File size
    uint64_t mtime;       // Modification time
    uint8_t _reserved[8]; // Padding to 256 bytes
};

// Entry flags
constant uint16_t FLAG_IS_DIR = 1 << 0;
constant uint16_t FLAG_IS_SYMLINK = 1 << 1;
constant uint16_t FLAG_IS_HIDDEN = 1 << 2;
constant uint16_t FLAG_IS_EXECUTABLE = 1 << 3;

// Index header offset (entries start after this)
constant uint32_t INDEX_HEADER_SIZE = 4096;
"#;
