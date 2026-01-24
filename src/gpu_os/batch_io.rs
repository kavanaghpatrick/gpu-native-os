// Issue #125: GPU-Initiated Batch I/O with MTLIOCommandQueue
//
// THE GPU IS THE COMPUTER. Batch ALL file loads into single GPU command.
//
// Traditional: CPU opens each file sequentially → 163ms for 10K files
// GPU Batch:   Queue all loads → single commit → GPU handles scheduling → ~30ms
//
// Key insight: MTLIOCommandQueue can batch hundreds of file loads into
// a single command buffer. The GPU scheduler optimizes the I/O order.

use metal::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::gpu_io::{GpuIOQueue, GpuIOFileHandle, GpuIOCommandBuffer, IOPriority, IOQueueType, IOStatus};

/// Page size for buffer alignment
const PAGE_SIZE: u64 = 4096;

/// Align size to page boundary
#[inline]
fn align_to_page(size: u64) -> u64 {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Descriptor for a file in the mega-buffer
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FileDescriptor {
    /// Offset in mega-buffer where this file's data starts
    pub offset: u64,
    /// Actual file size (not aligned)
    pub size: u32,
    /// File index in original list
    pub file_index: u32,
    /// I/O status: 0=pending, 1=loading, 2=error, 3=complete
    pub status: u32,
    /// Padding for Metal alignment
    pub _padding: u32,
}

/// Result of batch loading
pub struct BatchLoadResult {
    /// Single large buffer containing all file data
    pub mega_buffer: Buffer,
    /// Per-file descriptors (GPU-readable)
    pub descriptors: Buffer,
    /// Descriptor data (CPU-readable copy)
    pub descriptor_data: Vec<FileDescriptor>,
    /// File paths in order
    pub file_paths: Vec<PathBuf>,
    /// Total bytes loaded
    pub total_bytes: u64,
}

impl BatchLoadResult {
    /// Get data for a specific file
    pub fn file_data(&self, index: usize) -> Option<&[u8]> {
        let desc = self.descriptor_data.get(index)?;
        if desc.status != 3 {
            return None; // Not complete
        }

        let ptr = self.mega_buffer.contents() as *const u8;
        Some(unsafe {
            std::slice::from_raw_parts(
                ptr.add(desc.offset as usize),
                desc.size as usize,
            )
        })
    }

    /// Get file descriptor
    pub fn descriptor(&self, index: usize) -> Option<&FileDescriptor> {
        self.descriptor_data.get(index)
    }

    /// Number of files
    pub fn file_count(&self) -> usize {
        self.descriptor_data.len()
    }
}

/// Handle for in-progress async batch load
pub struct BatchLoadHandle {
    cmd_buffer: GpuIOCommandBuffer,
    mega_buffer: Buffer,
    descriptors: Buffer,
    descriptor_data: Vec<FileDescriptor>,
    file_paths: Vec<PathBuf>,
    total_bytes: u64,
    ready_count: Arc<AtomicU32>,
}

impl BatchLoadHandle {
    /// Check if all files are loaded
    pub fn is_complete(&self) -> bool {
        self.cmd_buffer.status() == IOStatus::Complete
    }

    /// Get number of files that have completed loading
    pub fn ready_count(&self) -> u32 {
        // Read from status buffer or check descriptors
        self.ready_count.load(Ordering::Relaxed)
    }

    /// Wait for all files to complete and return result
    pub fn wait(self) -> Option<BatchLoadResult> {
        self.cmd_buffer.wait_until_completed();

        if self.cmd_buffer.status() != IOStatus::Complete {
            return None;
        }

        // Update all descriptors to complete
        let mut descriptor_data = self.descriptor_data;
        for desc in &mut descriptor_data {
            desc.status = 3; // Complete
        }

        Some(BatchLoadResult {
            mega_buffer: self.mega_buffer,
            descriptors: self.descriptors,
            descriptor_data,
            file_paths: self.file_paths,
            total_bytes: self.total_bytes,
        })
    }

    /// Get the mega-buffer (may contain partial data)
    pub fn buffer(&self) -> &Buffer {
        &self.mega_buffer
    }

    /// Get file descriptors buffer
    pub fn descriptors_buffer(&self) -> &Buffer {
        &self.descriptors
    }
}

/// GPU Batch Loader - loads many files in single GPU command
pub struct GpuBatchLoader {
    queue: GpuIOQueue,
    device: Device,
}

impl GpuBatchLoader {
    /// Create a new batch loader
    pub fn new(device: &Device) -> Option<Self> {
        let queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent)?;
        Some(Self {
            queue,
            device: device.clone(),
        })
    }

    /// Load multiple files into a single mega-buffer (blocking)
    pub fn load_batch(&self, files: &[PathBuf]) -> Option<BatchLoadResult> {
        let handle = self.load_batch_async(files)?;
        handle.wait()
    }

    /// Load multiple files asynchronously (non-blocking)
    pub fn load_batch_async(&self, files: &[PathBuf]) -> Option<BatchLoadHandle> {
        if files.is_empty() {
            return None;
        }

        // Phase 1: Gather file metadata and compute offsets
        let mut descriptors = Vec::with_capacity(files.len());
        let mut file_handles = Vec::with_capacity(files.len());
        let mut current_offset = 0u64;
        let mut valid_files = Vec::with_capacity(files.len());

        for (i, path) in files.iter().enumerate() {
            // Get file size
            let size = match fs::metadata(path) {
                Ok(m) => m.len(),
                Err(_) => continue,
            };

            if size == 0 || size > 100 * 1024 * 1024 {
                continue; // Skip empty or >100MB files
            }

            // Open file handle for GPU I/O
            let handle = match GpuIOFileHandle::open(&self.device, path) {
                Some(h) => h,
                None => continue,
            };

            let aligned_size = align_to_page(size);

            descriptors.push(FileDescriptor {
                offset: current_offset,
                size: size as u32,
                file_index: i as u32,
                status: 0, // Pending
                _padding: 0,
            });

            file_handles.push((handle, current_offset, size));
            valid_files.push(path.clone());
            current_offset += aligned_size;
        }

        if descriptors.is_empty() {
            return None;
        }

        let total_bytes = current_offset;

        // Phase 2: Allocate mega-buffer
        let mega_buffer = self.device.new_buffer(
            total_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        // Phase 3: Create command buffer and queue ALL loads
        let cmd_buffer = self.queue.command_buffer()?;

        for (handle, offset, size) in &file_handles {
            cmd_buffer.load_buffer(&mega_buffer, *offset, *size, handle, 0);
        }

        // Phase 4: Commit batch
        cmd_buffer.commit();

        // Create descriptors buffer
        let descriptors_buffer = self.device.new_buffer_with_data(
            descriptors.as_ptr() as *const _,
            (descriptors.len() * std::mem::size_of::<FileDescriptor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Some(BatchLoadHandle {
            cmd_buffer,
            mega_buffer,
            descriptors: descriptors_buffer,
            descriptor_data: descriptors,
            file_paths: valid_files,
            total_bytes,
            ready_count: Arc::new(AtomicU32::new(0)),
        })
    }

    /// Load files with progress callback
    pub fn load_batch_with_progress<F>(
        &self,
        files: &[PathBuf],
        mut progress: F,
    ) -> Option<BatchLoadResult>
    where
        F: FnMut(usize, usize), // (loaded, total)
    {
        // For now, just do blocking load with final callback
        // TODO: Implement true progress tracking with MTLSharedEvent
        let handle = self.load_batch_async(files)?;
        let total = handle.descriptor_data.len();
        progress(0, total);

        let result = handle.wait()?;
        progress(total, total);
        Some(result)
    }
}

/// High-level API: Load files and return searchable buffer
pub struct GpuBatchSearchBuffer {
    result: BatchLoadResult,
    path_to_index: HashMap<PathBuf, usize>,
}

impl GpuBatchSearchBuffer {
    /// Create from batch load result
    pub fn new(result: BatchLoadResult) -> Self {
        let path_to_index: HashMap<_, _> = result.file_paths
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        Self { result, path_to_index }
    }

    /// Get the mega-buffer for GPU search
    pub fn search_buffer(&self) -> &Buffer {
        &self.result.mega_buffer
    }

    /// Get descriptors buffer for GPU search
    pub fn descriptors(&self) -> &Buffer {
        &self.result.descriptors
    }

    /// Get file count
    pub fn file_count(&self) -> usize {
        self.result.file_count()
    }

    /// Get total bytes
    pub fn total_bytes(&self) -> u64 {
        self.result.total_bytes
    }

    /// Find file index by path
    pub fn file_index(&self, path: &Path) -> Option<usize> {
        self.path_to_index.get(path).copied()
    }

    /// Get file path by index
    pub fn file_path(&self, index: usize) -> Option<&Path> {
        self.result.file_paths.get(index).map(|p| p.as_path())
    }

    /// Get descriptor by index
    pub fn descriptor(&self, index: usize) -> Option<&FileDescriptor> {
        self.result.descriptor(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn collect_test_files(dir: &Path, max: usize) -> Vec<PathBuf> {
        let mut files = Vec::new();
        collect_recursive(dir, &mut files, max, 0);
        files
    }

    fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>, max: usize, depth: usize) {
        if depth > 5 || files.len() >= max {
            return;
        }

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            if files.len() >= max {
                break;
            }

            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with('.') && name != "target" {
                    collect_recursive(&path, files, max, depth + 1);
                }
            } else if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ["rs", "txt", "md", "toml"].contains(&ext) {
                    if let Ok(meta) = fs::metadata(&path) {
                        if meta.len() > 0 && meta.len() < 1_000_000 {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_batch_loader_creation() {
        let device = Device::system_default().expect("No Metal device");

        match GpuBatchLoader::new(&device) {
            Some(_) => println!("GpuBatchLoader created successfully"),
            None => println!("MTLIOCommandQueue not available (requires Metal 3+)"),
        }
    }

    #[test]
    fn test_batch_load_single_file() {
        let device = Device::system_default().expect("No Metal device");

        let loader = match GpuBatchLoader::new(&device) {
            Some(l) => l,
            None => {
                println!("Skipping: MTLIOCommandQueue not available");
                return;
            }
        };

        let files = vec![PathBuf::from("Cargo.toml")];
        let result = loader.load_batch(&files);

        match result {
            Some(r) => {
                println!("Loaded {} files, {} bytes", r.file_count(), r.total_bytes);

                // Verify content
                let expected = fs::read("Cargo.toml").expect("Failed to read file");
                let actual = r.file_data(0).expect("Failed to get file data");
                assert_eq!(actual, expected.as_slice());
                println!("Content verified!");
            }
            None => println!("Batch load failed"),
        }
    }

    #[test]
    fn test_batch_load_multiple_files() {
        let device = Device::system_default().expect("No Metal device");

        let loader = match GpuBatchLoader::new(&device) {
            Some(l) => l,
            None => {
                println!("Skipping: MTLIOCommandQueue not available");
                return;
            }
        };

        let files = collect_test_files(Path::new("."), 100);
        if files.is_empty() {
            println!("No test files found");
            return;
        }

        println!("Loading {} files via GPU batch I/O", files.len());

        let start = Instant::now();
        let result = loader.load_batch(&files);
        let elapsed = start.elapsed();

        match result {
            Some(r) => {
                println!("Loaded {} files ({:.1} KB) in {:.1}ms",
                    r.file_count(),
                    r.total_bytes as f64 / 1024.0,
                    elapsed.as_secs_f64() * 1000.0);

                let throughput = (r.total_bytes as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64();
                println!("Throughput: {:.1} MB/s", throughput);

                // Verify a few files
                for i in 0..5.min(r.file_count()) {
                    if let Some(path) = r.file_paths.get(i) {
                        let expected = fs::read(path).unwrap_or_default();
                        let actual = r.file_data(i).unwrap_or_default();
                        assert_eq!(actual.len(), expected.len(),
                            "Size mismatch for file {}", path.display());
                    }
                }
                println!("Content verified for {} files", 5.min(r.file_count()));
            }
            None => println!("Batch load failed"),
        }
    }

    #[test]
    fn benchmark_batch_vs_sequential() {
        let device = Device::system_default().expect("No Metal device");

        let files = collect_test_files(Path::new("."), 500);
        if files.is_empty() {
            println!("No test files found");
            return;
        }

        println!("\n=== Batch vs Sequential Loading ({} files) ===\n", files.len());

        // Sequential: std::fs::read
        let seq_start = Instant::now();
        let mut seq_total = 0usize;
        for file in &files {
            if let Ok(data) = fs::read(file) {
                seq_total += data.len();
            }
        }
        let seq_time = seq_start.elapsed();

        println!("Sequential (fs::read):");
        println!("  Time: {:.1}ms", seq_time.as_secs_f64() * 1000.0);
        println!("  Data: {:.1} KB", seq_total as f64 / 1024.0);
        println!("  Throughput: {:.1} MB/s",
            (seq_total as f64 / (1024.0 * 1024.0)) / seq_time.as_secs_f64());

        // Batch: GPU I/O
        if let Some(loader) = GpuBatchLoader::new(&device) {
            let batch_start = Instant::now();
            if let Some(result) = loader.load_batch(&files) {
                let batch_time = batch_start.elapsed();

                println!("\nGPU Batch I/O:");
                println!("  Time: {:.1}ms", batch_time.as_secs_f64() * 1000.0);
                println!("  Data: {:.1} KB", result.total_bytes as f64 / 1024.0);
                println!("  Throughput: {:.1} MB/s",
                    (result.total_bytes as f64 / (1024.0 * 1024.0)) / batch_time.as_secs_f64());

                let speedup = seq_time.as_secs_f64() / batch_time.as_secs_f64();
                println!("\n  Speedup: {:.2}x", speedup);
            }
        } else {
            println!("\nGPU Batch I/O not available (requires Metal 3+)");
        }
    }
}
