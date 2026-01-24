// Issue #82: Zero-Copy File Access via mmap + newBufferWithBytesNoCopy
//
// THE GPU IS THE COMPUTER. No CPU copies allowed.
//
// Traditional: File → CPU Read → CPU Buffer → GPU Copy → GPU
// Zero-Copy:   File → mmap → newBufferWithBytesNoCopy → GPU (same memory!)
//
// With Apple Silicon unified memory, mmap'd data flows:
//   Disk → Unified Memory → GPU (same physical memory, no copies!)

use metal::*;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;

/// Page size for alignment (4KB on most systems)
pub const PAGE_SIZE: usize = 4096;

/// A memory-mapped file exposed as a Metal buffer with zero copies.
///
/// # Example
/// ```ignore
/// let buffer = MmapBuffer::from_file(&device, "index.bin")?;
/// // GPU can now read file data directly - no copies!
/// encoder.set_buffer(0, Some(buffer.metal_buffer()), 0);
/// ```
///
/// # Safety
/// The mmap and Metal buffer share the same physical memory.
/// The buffer remains valid as long as MmapBuffer is alive.
pub struct MmapBuffer {
    /// Raw pointer to mmap'd memory (kept for munmap on drop)
    mmap_ptr: *mut libc::c_void,
    /// Size of mmap'd region (page-aligned)
    mmap_len: usize,
    /// Original file size (not page-aligned)
    file_size: usize,
    /// Keep file open to maintain mmap
    _file: File,
    /// Metal buffer pointing to same memory - NO COPY
    buffer: Buffer,
}

// SAFETY: The mmap and buffer point to the same memory,
// which is valid as long as MmapBuffer is alive.
// The mmap is read-only (MAP_PRIVATE + PROT_READ).
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a zero-copy Metal buffer from a file.
    ///
    /// The file is memory-mapped and a Metal buffer is created pointing
    /// to the same physical memory. No data is copied.
    ///
    /// # Arguments
    /// * `device` - Metal device to create buffer on
    /// * `path` - Path to file to map
    ///
    /// # Returns
    /// * `Ok(MmapBuffer)` - Zero-copy buffer ready for GPU access
    /// * `Err(MmapError)` - If file doesn't exist, is empty, or mmap fails
    pub fn from_file(device: &Device, path: impl AsRef<Path>) -> Result<Self, MmapError> {
        let path = path.as_ref();

        let file = File::open(path)
            .map_err(MmapError::IoError)?;

        let file_size = file.metadata()
            .map_err(MmapError::IoError)?
            .len() as usize;

        if file_size == 0 {
            return Err(MmapError::EmptyFile);
        }

        // Round up to page boundary for mmap alignment
        let aligned_size = (file_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // Memory-map the file (read-only, private)
        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(MmapError::MmapFailed(std::io::Error::last_os_error()));
        }

        // Create Metal buffer pointing to same memory - ZERO COPY!
        // StorageModeShared is required for unified memory access
        let buffer = device.new_buffer_with_bytes_no_copy(
            mmap_ptr,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None, // No deallocator - we manage the mmap ourselves
        );

        Ok(Self {
            mmap_ptr,
            mmap_len: aligned_size,
            file_size,
            _file: file,
            buffer,
        })
    }

    /// Create a zero-copy Metal buffer from raw bytes (for testing).
    ///
    /// Allocates page-aligned memory and creates a Metal buffer pointing to it.
    /// Useful for creating test buffers without actual files.
    pub fn from_bytes(device: &Device, data: &[u8]) -> Result<Self, MmapError> {
        if data.is_empty() {
            return Err(MmapError::EmptyFile);
        }

        let file_size = data.len();
        let aligned_size = (file_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // Allocate page-aligned memory using mmap (anonymous)
        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1, // No file descriptor for anonymous mapping
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(MmapError::MmapFailed(std::io::Error::last_os_error()));
        }

        // Copy data into mmap'd region
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), mmap_ptr as *mut u8, file_size);
        }

        // Make it read-only
        unsafe {
            libc::mprotect(mmap_ptr, aligned_size, libc::PROT_READ);
        }

        // Create Metal buffer - ZERO COPY from this point on
        let buffer = device.new_buffer_with_bytes_no_copy(
            mmap_ptr,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Create a dummy file handle (we don't actually have a file)
        // This is a bit of a hack, but from_bytes is mainly for testing
        let dummy_file = File::open("/dev/null").map_err(MmapError::IoError)?;

        Ok(Self {
            mmap_ptr,
            mmap_len: aligned_size,
            file_size,
            _file: dummy_file,
            buffer,
        })
    }

    /// Get the Metal buffer for GPU access.
    ///
    /// Use this with encoder.set_buffer() to give GPU access to the data.
    #[inline]
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the original file size (not page-aligned).
    #[inline]
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Get the aligned buffer size (multiple of PAGE_SIZE).
    #[inline]
    pub fn aligned_size(&self) -> usize {
        self.mmap_len
    }

    /// Get raw pointer to mmap'd data (for CPU-side inspection).
    ///
    /// # Safety
    /// The pointer is valid only while MmapBuffer is alive.
    /// Data beyond file_size() may be uninitialized.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap_ptr as *const u8
    }

    /// Advise the kernel about sequential access pattern.
    ///
    /// Call this if you'll read the data from start to end.
    /// Helps kernel optimize page prefetching.
    pub fn advise_sequential(&self) {
        unsafe {
            libc::madvise(
                self.mmap_ptr,
                self.mmap_len,
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    /// Advise the kernel that we'll need this data soon.
    ///
    /// Call this before GPU access to prefetch pages into memory.
    /// Reduces page fault latency during GPU reads.
    pub fn advise_willneed(&self) {
        unsafe {
            libc::madvise(
                self.mmap_ptr,
                self.mmap_len,
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Advise the kernel about random access pattern.
    ///
    /// Call this if you'll access data in random order.
    pub fn advise_random(&self) {
        unsafe {
            libc::madvise(
                self.mmap_ptr,
                self.mmap_len,
                libc::MADV_RANDOM,
            );
        }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        // Unmap the memory region
        unsafe {
            libc::munmap(self.mmap_ptr, self.mmap_len);
        }
        // File is closed automatically when _file is dropped
    }
}

/// Errors that can occur when creating an MmapBuffer
#[derive(Debug)]
pub enum MmapError {
    /// I/O error (file not found, permission denied, etc.)
    IoError(std::io::Error),
    /// mmap system call failed
    MmapFailed(std::io::Error),
    /// Cannot mmap an empty file
    EmptyFile,
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::IoError(e) => write!(f, "I/O error: {}", e),
            MmapError::MmapFailed(e) => write!(f, "mmap failed: {}", e),
            MmapError::EmptyFile => write!(f, "Cannot mmap empty file"),
        }
    }
}

impl std::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MmapError::IoError(e) | MmapError::MmapFailed(e) => Some(e),
            MmapError::EmptyFile => None,
        }
    }
}

// ============================================================================
// Helper: Create a page-aligned buffer for writing (useful for creating index files)
// ============================================================================

/// Create a page-aligned Vec for data that will be memory-mapped.
///
/// When writing index files that will later be mmap'd, use this to ensure
/// proper alignment.
pub fn create_aligned_vec(capacity: usize) -> Vec<u8> {
    let aligned_capacity = (capacity + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let mut vec = Vec::with_capacity(aligned_capacity);

    // Pre-fill to aligned size to ensure allocation is aligned
    // (Vec doesn't guarantee page alignment, but this increases likelihood)
    vec.resize(aligned_capacity, 0);
    vec.truncate(0);

    vec
}

/// Round a size up to the nearest page boundary.
#[inline]
pub fn align_to_page(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

// ============================================================================
// WritableMmapBuffer: GPU-Direct File Writing via mmap
// ============================================================================

/// A writable memory-mapped file exposed as a Metal buffer.
///
/// GPU writes to this buffer are automatically persisted to disk via mmap.
/// This enables true GPU-direct file editing without CPU involvement.
///
/// # How It Works
/// 1. File is opened with read/write permissions
/// 2. mmap with MAP_SHARED makes GPU writes visible to the file
/// 3. Metal buffer points to mmap'd memory (unified memory on Apple Silicon)
/// 4. GPU writes → mmap → kernel flushes to disk
///
/// # Example
/// ```ignore
/// let mut buffer = WritableMmapBuffer::open(&device, "data.bin")?;
/// // GPU kernel modifies the buffer...
/// buffer.sync(); // Ensure changes are flushed to disk
/// ```
pub struct WritableMmapBuffer {
    mmap_ptr: *mut libc::c_void,
    mmap_len: usize,
    file_size: usize,
    file: File,
    buffer: Buffer,
}

unsafe impl Send for WritableMmapBuffer {}
unsafe impl Sync for WritableMmapBuffer {}

impl WritableMmapBuffer {
    /// Open an existing file for GPU-direct read/write access.
    pub fn open(device: &Device, path: impl AsRef<Path>) -> Result<Self, MmapError> {
        use std::fs::OpenOptions;

        let path = path.as_ref();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(MmapError::IoError)?;

        let file_size = file.metadata()
            .map_err(MmapError::IoError)?
            .len() as usize;

        if file_size == 0 {
            return Err(MmapError::EmptyFile);
        }

        let aligned_size = (file_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // MAP_SHARED: changes are written back to file
        // PROT_READ | PROT_WRITE: GPU can read and write
        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(MmapError::MmapFailed(std::io::Error::last_os_error()));
        }

        // Create Metal buffer - GPU writes go directly to mmap'd file
        let buffer = device.new_buffer_with_bytes_no_copy(
            mmap_ptr,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Ok(Self {
            mmap_ptr,
            mmap_len: aligned_size,
            file_size,
            file,
            buffer,
        })
    }

    /// Create a new file with the specified size for GPU-direct access.
    pub fn create(device: &Device, path: impl AsRef<Path>, size: usize) -> Result<Self, MmapError> {
        use std::fs::OpenOptions;

        if size == 0 {
            return Err(MmapError::EmptyFile);
        }

        let path = path.as_ref();
        let aligned_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // Create file with specified size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(MmapError::IoError)?;

        // Extend file to required size
        file.set_len(aligned_size as u64)
            .map_err(MmapError::IoError)?;

        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(MmapError::MmapFailed(std::io::Error::last_os_error()));
        }

        let buffer = device.new_buffer_with_bytes_no_copy(
            mmap_ptr,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Ok(Self {
            mmap_ptr,
            mmap_len: aligned_size,
            file_size: size,
            file,
            buffer,
        })
    }

    /// Get the Metal buffer for GPU access.
    #[inline]
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the file size.
    #[inline]
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Sync changes to disk.
    ///
    /// GPU writes are automatically visible to the mmap, but the kernel
    /// may not have flushed them to disk yet. Call this to ensure durability.
    pub fn sync(&self) -> Result<(), MmapError> {
        let result = unsafe {
            libc::msync(self.mmap_ptr, self.mmap_len, libc::MS_SYNC)
        };

        if result != 0 {
            return Err(MmapError::IoError(std::io::Error::last_os_error()));
        }

        Ok(())
    }

    /// Async sync - returns immediately, sync happens in background.
    pub fn sync_async(&self) {
        unsafe {
            libc::msync(self.mmap_ptr, self.mmap_len, libc::MS_ASYNC);
        }
    }

    /// Resize the file and remapping.
    ///
    /// Note: This invalidates any existing GPU references to the buffer.
    pub fn resize(&mut self, device: &Device, new_size: usize) -> Result<(), MmapError> {
        if new_size == 0 {
            return Err(MmapError::EmptyFile);
        }

        // Sync before resize
        self.sync()?;

        // Unmap old region
        unsafe {
            libc::munmap(self.mmap_ptr, self.mmap_len);
        }

        let aligned_size = (new_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // Resize file
        self.file.set_len(aligned_size as u64)
            .map_err(MmapError::IoError)?;

        // Remap
        let mmap_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.file.as_raw_fd(),
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(MmapError::MmapFailed(std::io::Error::last_os_error()));
        }

        // Create new Metal buffer
        self.buffer = device.new_buffer_with_bytes_no_copy(
            mmap_ptr,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        self.mmap_ptr = mmap_ptr;
        self.mmap_len = aligned_size;
        self.file_size = new_size;

        Ok(())
    }

    /// Get raw pointer for CPU-side access.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap_ptr as *const u8
    }

    /// Get mutable raw pointer for CPU-side access.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mmap_ptr as *mut u8
    }
}

impl Drop for WritableMmapBuffer {
    fn drop(&mut self) {
        // Sync changes before unmapping
        let _ = self.sync();

        unsafe {
            libc::munmap(self.mmap_ptr, self.mmap_len);
        }
    }
}
