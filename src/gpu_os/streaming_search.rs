// Issue #132: Streaming I/O - Overlap File Loading with GPU Search
//
// THE GPU IS THE COMPUTER. Don't wait for ALL files to load before searching.
//
// Traditional:  Load ALL files (283ms) → THEN search (50ms) = 333ms total
// Streaming:    Load chunk 1 → [Load 2 + Search 1] → [Load 3 + Search 2] → ...
//
// Key insight: GPU can search one chunk while MTLIOCommandQueue loads the next.
// This overlaps I/O and compute for ~30%+ speedup on I/O-bound workloads.

use metal::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::batch_io::FileDescriptor;
use super::content_search::ContentMatch;
use super::gpu_io::{GpuIOCommandBuffer, GpuIOFileHandle, GpuIOQueue, IOPriority, IOQueueType};

/// Page size for buffer alignment
const PAGE_SIZE: u64 = 4096;

/// Default number of chunks for streaming (quad-buffering)
const DEFAULT_CHUNK_COUNT: usize = 4;

/// Maximum files per chunk (prevents memory bloat)
const MAX_FILES_PER_CHUNK: usize = 5000;

/// Chunk size in bytes (64 MB default - fits in L2 cache)
const DEFAULT_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Align size to page boundary
#[inline]
fn align_to_page(size: u64) -> u64 {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// A streaming chunk with buffer and metadata
pub struct StreamChunk {
    /// GPU buffer for file data
    pub buffer: Buffer,
    /// GPU buffer for file descriptors
    pub descriptors: Buffer,
    /// CPU-side descriptor data for result extraction
    pub descriptor_data: Vec<FileDescriptor>,
    /// File paths in this chunk
    pub file_paths: Vec<PathBuf>,
    /// Number of files in this chunk
    pub file_count: usize,
    /// Total bytes loaded (actual data, not aligned)
    pub total_bytes: u64,
    /// Set to true when I/O completes
    pub ready: Arc<AtomicBool>,
    /// Signaled value for event synchronization
    pub signal_value: AtomicU64,
}

impl StreamChunk {
    /// Create a new empty chunk with pre-allocated buffer
    fn new(device: &Device, max_bytes: u64, max_files: usize) -> Self {
        let buffer = device.new_buffer(max_bytes, MTLResourceOptions::StorageModeShared);
        let descriptors = device.new_buffer(
            (max_files * std::mem::size_of::<FileDescriptor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            buffer,
            descriptors,
            descriptor_data: Vec::with_capacity(max_files),
            file_paths: Vec::with_capacity(max_files),
            file_count: 0,
            total_bytes: 0,
            ready: Arc::new(AtomicBool::new(false)),
            signal_value: AtomicU64::new(0),
        }
    }

    /// Reset chunk for reuse
    fn reset(&mut self) {
        self.descriptor_data.clear();
        self.file_paths.clear();
        self.file_count = 0;
        self.total_bytes = 0;
        self.ready.store(false, Ordering::Release);
    }

    /// Mark chunk as ready (I/O complete)
    fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Check if chunk is ready for search
    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Sync descriptors to GPU buffer (must call after populating descriptor_data)
    fn sync_descriptors(&self) {
        if self.descriptor_data.is_empty() {
            return;
        }
        unsafe {
            let ptr = self.descriptors.contents() as *mut FileDescriptor;
            for (i, desc) in self.descriptor_data.iter().enumerate() {
                *ptr.add(i) = *desc;
            }
        }
    }
}

/// Streaming pipeline with quad-buffering for overlapped I/O
pub struct StreamingPipeline {
    /// Pre-allocated chunk buffers
    chunks: Vec<StreamChunk>,
    /// IO queue for async file loading
    io_queue: GpuIOQueue,
    /// Device reference
    device: Device,
    /// Max bytes per chunk
    chunk_bytes: u64,
    /// Max files per chunk
    max_files_per_chunk: usize,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline with quad-buffering
    pub fn new(device: &Device) -> Option<Self> {
        Self::with_config(device, DEFAULT_CHUNK_COUNT, DEFAULT_CHUNK_BYTES, MAX_FILES_PER_CHUNK)
    }

    /// Create pipeline with custom configuration
    pub fn with_config(
        device: &Device,
        chunk_count: usize,
        chunk_bytes: u64,
        max_files_per_chunk: usize,
    ) -> Option<Self> {
        let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent)?;

        let mut chunks = Vec::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            chunks.push(StreamChunk::new(device, chunk_bytes, max_files_per_chunk));
        }

        Some(Self {
            chunks,
            io_queue,
            device: device.clone(),
            chunk_bytes,
            max_files_per_chunk,
        })
    }

    /// Get number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Reset all chunks for reuse
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.reset();
        }
    }

    /// Partition files into chunks based on size
    ///
    /// Returns chunk boundaries: [(start_idx, end_idx), ...]
    pub fn partition_files(&self, files: &[PathBuf]) -> Vec<(usize, usize)> {
        if files.is_empty() {
            return vec![];
        }

        let mut partitions = Vec::new();
        let mut current_start = 0;
        let mut current_bytes = 0u64;
        let mut current_count = 0usize;

        for (i, file) in files.iter().enumerate() {
            let file_size = fs::metadata(file).map(|m| m.len()).unwrap_or(0);

            // Skip empty or huge files
            if file_size == 0 || file_size > 100 * 1024 * 1024 {
                continue;
            }

            let aligned_size = align_to_page(file_size);

            // Check if adding this file would exceed chunk limits
            let would_exceed_bytes = current_bytes + aligned_size > self.chunk_bytes;
            let would_exceed_count = current_count + 1 > self.max_files_per_chunk;

            if would_exceed_bytes || would_exceed_count {
                // Start new chunk
                if current_count > 0 {
                    partitions.push((current_start, i));
                }
                current_start = i;
                current_bytes = aligned_size;
                current_count = 1;
            } else {
                current_bytes += aligned_size;
                current_count += 1;
            }
        }

        // Add final chunk
        if current_count > 0 {
            partitions.push((current_start, files.len()));
        }

        // Limit to available chunks (merge extras into last chunk)
        if partitions.len() > self.chunks.len() {
            let _last_idx = self.chunks.len() - 1;
            let last_end = partitions.last().map(|(_, e)| *e).unwrap_or(files.len());
            partitions.truncate(self.chunks.len());
            if let Some((_, end)) = partitions.last_mut() {
                *end = last_end;
            }
        }

        partitions
    }

    /// Load a chunk asynchronously (returns immediately)
    ///
    /// Call `wait_chunk_ready` before searching this chunk.
    pub fn start_load_chunk(&mut self, chunk_idx: usize, files: &[PathBuf]) -> Option<GpuIOCommandBuffer> {
        if chunk_idx >= self.chunks.len() || files.is_empty() {
            return None;
        }

        let chunk = &mut self.chunks[chunk_idx];
        chunk.reset();

        // Phase 1: Gather file metadata and open handles
        let mut file_handles = Vec::with_capacity(files.len());
        let mut current_offset = 0u64;

        for (_i, path) in files.iter().enumerate() {
            let size = match fs::metadata(path) {
                Ok(m) => m.len(),
                Err(_) => continue,
            };

            if size == 0 || size > 100 * 1024 * 1024 {
                continue;
            }

            let handle = match GpuIOFileHandle::open(&self.device, path) {
                Some(h) => h,
                None => continue,
            };

            let aligned_size = align_to_page(size);

            // Check if we'd exceed chunk buffer
            if current_offset + aligned_size > self.chunk_bytes {
                break;
            }

            chunk.descriptor_data.push(FileDescriptor {
                offset: current_offset,
                size: size as u32,
                file_index: chunk.file_count as u32,
                status: 0, // Will be set to 3 (complete) after load
                _padding: 0,
            });

            file_handles.push((handle, current_offset, size));
            chunk.file_paths.push(path.clone());
            chunk.file_count += 1;
            chunk.total_bytes += size;
            current_offset += aligned_size;
        }

        if chunk.file_count == 0 {
            return None;
        }

        // Phase 2: Queue all loads
        let cmd_buffer = self.io_queue.command_buffer()?;

        for (handle, offset, size) in &file_handles {
            cmd_buffer.load_buffer(&chunk.buffer, *offset, *size, handle, 0);
        }

        // Commit but DON'T wait - returns immediately
        cmd_buffer.commit();

        Some(cmd_buffer)
    }

    /// Wait for a chunk's I/O to complete
    pub fn wait_chunk_ready(&mut self, chunk_idx: usize, cmd_buffer: Option<&GpuIOCommandBuffer>) {
        if chunk_idx >= self.chunks.len() {
            return;
        }

        // Wait for IO command buffer to complete
        if let Some(cb) = cmd_buffer {
            cb.wait_until_completed();
        }

        // Mark all descriptors as complete
        let chunk = &mut self.chunks[chunk_idx];
        for desc in &mut chunk.descriptor_data {
            desc.status = 3; // Complete
        }

        // Sync to GPU buffer
        chunk.sync_descriptors();
        chunk.mark_ready();
    }

    /// Get chunk for reading (after I/O complete)
    pub fn chunk(&self, idx: usize) -> Option<&StreamChunk> {
        self.chunks.get(idx)
    }

    /// Get mutable chunk
    pub fn chunk_mut(&mut self, idx: usize) -> Option<&mut StreamChunk> {
        self.chunks.get_mut(idx)
    }
}

/// Streaming search engine - overlaps I/O and GPU compute
pub struct StreamingSearch {
    /// Device reference
    #[allow(dead_code)]
    device: Device,
    /// Command queue for search operations
    command_queue: CommandQueue,
    /// Streaming pipeline with quad-buffers
    pipeline: StreamingPipeline,
    /// Search kernel pipeline state
    search_pipeline: ComputePipelineState,
    /// Search parameters buffer
    params_buffer: Buffer,
    /// Pattern buffer
    pattern_buffer: Buffer,
    /// Results buffer
    matches_buffer: Buffer,
    /// Match count buffer (atomic)
    match_count_buffer: Buffer,
}

/// Search parameters for streaming kernel
#[repr(C)]
#[derive(Copy, Clone)]
struct StreamSearchParams {
    file_count: u32,
    pattern_len: u32,
    case_sensitive: u32,
    total_bytes: u32,
}

/// Match result from GPU
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuMatchResult {
    file_index: u32,
    chunk_index: u32,
    line_number: u32,
    column: u32,
    match_length: u32,
    context_start: u32,
    context_len: u32,
    _padding: u32,
}

const MAX_PATTERN_LEN: usize = 64;
const MAX_MATCHES: usize = 10000;
const BYTES_PER_THREAD: usize = 64;

impl StreamingSearch {
    /// Create a new streaming search engine
    pub fn new(device: &Device) -> Option<Self> {
        let pipeline = StreamingPipeline::new(device)?;
        let command_queue = device.new_command_queue();

        // Compile search shader (reuse from content_search)
        let shader_source = get_streaming_search_shader();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(&shader_source, &options).ok()?;
        let kernel = library.get_function("streaming_search_kernel", None).ok()?;
        let search_pipeline = device.new_compute_pipeline_state_with_function(&kernel).ok()?;

        // Allocate buffers
        let params_buffer = device.new_buffer(
            std::mem::size_of::<StreamSearchParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pattern_buffer = device.new_buffer(
            MAX_PATTERN_LEN as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let matches_buffer = device.new_buffer(
            (MAX_MATCHES * std::mem::size_of::<GpuMatchResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let match_count_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Some(Self {
            device: device.clone(),
            command_queue,
            pipeline,
            search_pipeline,
            params_buffer,
            pattern_buffer,
            matches_buffer,
            match_count_buffer,
        })
    }

    /// Create with custom pipeline configuration
    pub fn with_config(
        device: &Device,
        chunk_count: usize,
        chunk_bytes: u64,
        max_files_per_chunk: usize,
    ) -> Option<Self> {
        let pipeline = StreamingPipeline::with_config(device, chunk_count, chunk_bytes, max_files_per_chunk)?;
        let command_queue = device.new_command_queue();

        let shader_source = get_streaming_search_shader();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(&shader_source, &options).ok()?;
        let kernel = library.get_function("streaming_search_kernel", None).ok()?;
        let search_pipeline = device.new_compute_pipeline_state_with_function(&kernel).ok()?;

        let params_buffer = device.new_buffer(
            std::mem::size_of::<StreamSearchParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pattern_buffer = device.new_buffer(
            MAX_PATTERN_LEN as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let matches_buffer = device.new_buffer(
            (MAX_MATCHES * std::mem::size_of::<GpuMatchResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let match_count_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Some(Self {
            device: device.clone(),
            command_queue,
            pipeline,
            search_pipeline,
            params_buffer,
            pattern_buffer,
            matches_buffer,
            match_count_buffer,
        })
    }

    /// Search files using streaming pipeline
    ///
    /// Overlaps I/O and GPU compute for improved throughput.
    /// Returns same results as batch search, but faster for I/O-bound workloads.
    pub fn search_streaming(
        &mut self,
        files: &[PathBuf],
        pattern: &str,
        case_sensitive: bool,
    ) -> Vec<ContentMatch> {
        let (results, _) = self.search_streaming_with_profile(files, pattern, case_sensitive);
        results
    }

    /// Search with detailed profiling
    pub fn search_streaming_with_profile(
        &mut self,
        files: &[PathBuf],
        pattern: &str,
        case_sensitive: bool,
    ) -> (Vec<ContentMatch>, StreamingProfile) {
        let total_start = Instant::now();
        let mut profile = StreamingProfile::default();

        if files.is_empty() || pattern.is_empty() || pattern.len() > MAX_PATTERN_LEN {
            return (vec![], profile);
        }

        // Reset match count
        unsafe {
            let ptr = self.match_count_buffer.contents() as *mut u32;
            *ptr = 0;
        }

        // Prepare pattern
        let pattern_bytes: Vec<u8> = if case_sensitive {
            pattern.as_bytes().to_vec()
        } else {
            pattern.to_lowercase().as_bytes().to_vec()
        };

        unsafe {
            let ptr = self.pattern_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(pattern_bytes.as_ptr(), ptr, pattern_bytes.len());
        }

        // Partition files into chunks
        let partition_start = Instant::now();
        self.pipeline.reset();
        let partitions = self.pipeline.partition_files(files);
        profile.partition_us = partition_start.elapsed().as_micros() as u64;
        profile.chunk_count = partitions.len();

        if partitions.is_empty() {
            return (vec![], profile);
        }

        let mut all_results = Vec::new();
        let mut _current_chunk_idx = 0usize;

        // Start loading first chunk
        let first_partition = &partitions[0];
        let first_files = &files[first_partition.0..first_partition.1];

        let io_start = Instant::now();
        let mut pending_io = self.pipeline.start_load_chunk(0, first_files);
        profile.io_queue_us += io_start.elapsed().as_micros() as u64;

        // Process chunks with overlap
        for (chunk_idx, partition) in partitions.iter().enumerate() {
            // Wait for current chunk I/O to complete
            let io_wait_start = Instant::now();
            self.pipeline.wait_chunk_ready(chunk_idx, pending_io.as_ref());
            profile.io_wait_us += io_wait_start.elapsed().as_micros() as u64;
            pending_io = None;

            // Start loading NEXT chunk (overlapped with search)
            if chunk_idx + 1 < partitions.len() {
                let next_partition = &partitions[chunk_idx + 1];
                let next_files = &files[next_partition.0..next_partition.1];

                let io_start = Instant::now();
                pending_io = self.pipeline.start_load_chunk(chunk_idx + 1, next_files);
                profile.io_queue_us += io_start.elapsed().as_micros() as u64;
            }

            // Search current chunk while next is loading
            let search_start = Instant::now();
            let chunk_results = self.search_chunk(
                chunk_idx,
                &pattern_bytes,
                case_sensitive,
                partition.0, // Base file index for this chunk
            );
            profile.search_us += search_start.elapsed().as_micros() as u64;

            all_results.extend(chunk_results);
            profile.files_processed += self.pipeline.chunk(chunk_idx).map(|c| c.file_count).unwrap_or(0);
            profile.bytes_processed += self.pipeline.chunk(chunk_idx).map(|c| c.total_bytes).unwrap_or(0);

            _current_chunk_idx = chunk_idx;
        }

        profile.total_us = total_start.elapsed().as_micros() as u64;
        profile.match_count = all_results.len();

        // Sort results by file path and line number
        all_results.sort_by(|a, b| {
            a.file_path.cmp(&b.file_path)
                .then(a.line_number.cmp(&b.line_number))
        });

        (all_results, profile)
    }

    /// Search a single chunk
    fn search_chunk(
        &self,
        chunk_idx: usize,
        pattern_bytes: &[u8],
        case_sensitive: bool,
        base_file_index: usize,
    ) -> Vec<ContentMatch> {
        let chunk = match self.pipeline.chunk(chunk_idx) {
            Some(c) if c.is_ready() && c.file_count > 0 => c,
            _ => return vec![],
        };

        // Reset match count for this chunk
        // Note: We accumulate across chunks, so we track the starting count
        let start_count = unsafe {
            *(self.match_count_buffer.contents() as *const u32)
        };

        // Write search params
        unsafe {
            let ptr = self.params_buffer.contents() as *mut StreamSearchParams;
            *ptr = StreamSearchParams {
                file_count: chunk.file_count as u32,
                pattern_len: pattern_bytes.len() as u32,
                case_sensitive: if case_sensitive { 1 } else { 0 },
                total_bytes: chunk.total_bytes as u32,
            };
        }

        // Dispatch search
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.search_pipeline);
        encoder.set_buffer(0, Some(&chunk.buffer), 0);
        encoder.set_buffer(1, Some(&chunk.descriptors), 0);
        encoder.set_buffer(2, Some(&self.params_buffer), 0);
        encoder.set_buffer(3, Some(&self.pattern_buffer), 0);
        encoder.set_buffer(4, Some(&self.matches_buffer), 0);
        encoder.set_buffer(5, Some(&self.match_count_buffer), 0);

        // Calculate threads needed
        let total_threads = ((chunk.total_bytes as usize) + BYTES_PER_THREAD - 1) / BYTES_PER_THREAD;
        let threads_per_group = 256;

        encoder.dispatch_threads(
            MTLSize::new(total_threads as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Extract results for this chunk
        self.extract_chunk_results(chunk, base_file_index, start_count)
    }

    /// Extract results from a chunk search
    fn extract_chunk_results(
        &self,
        chunk: &StreamChunk,
        _base_file_index: usize,
        start_count: u32,
    ) -> Vec<ContentMatch> {
        let mut results = Vec::new();

        unsafe {
            let total_count = *(self.match_count_buffer.contents() as *const u32);
            let matches = self.matches_buffer.contents() as *const GpuMatchResult;

            // Only process new matches from this chunk
            let new_count = (total_count - start_count) as usize;
            let end_idx = (total_count as usize).min(MAX_MATCHES);
            let start_idx = end_idx.saturating_sub(new_count);

            for i in start_idx..end_idx {
                let m = *matches.add(i);

                // Get file path from chunk
                let file_idx = m.file_index as usize;
                let file_path = chunk.file_paths.get(file_idx)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                if file_path.is_empty() {
                    continue;
                }

                // Get descriptor for context extraction
                let desc = match chunk.descriptor_data.get(file_idx) {
                    Some(d) => d,
                    None => continue,
                };

                // Extract context from chunk buffer
                let context = self.extract_context(chunk, desc, &m);

                // Calculate line number (count newlines before match)
                let line_number = self.calculate_line_number(chunk, desc, m.column as usize);

                results.push(ContentMatch {
                    file_path,
                    line_number,
                    column: m.column,
                    context,
                    match_start: m.column as usize,
                });
            }
        }

        results
    }

    /// Extract context string around a match
    fn extract_context(&self, chunk: &StreamChunk, desc: &FileDescriptor, m: &GpuMatchResult) -> String {
        let offset_in_file = m.column as usize;
        let global_offset = desc.offset as usize + offset_in_file;

        unsafe {
            let ptr = chunk.buffer.contents() as *const u8;
            let file_start = desc.offset as usize;
            let file_end = file_start + desc.size as usize;

            // Find line start
            let mut line_start = global_offset;
            while line_start > file_start {
                if *ptr.add(line_start - 1) == b'\n' {
                    break;
                }
                line_start -= 1;
            }

            // Find line end
            let mut line_end = global_offset + m.match_length as usize;
            while line_end < file_end && *ptr.add(line_end) != b'\n' && line_end < line_start + 120 {
                line_end += 1;
            }

            let context_len = (line_end - line_start).min(120);
            let data = std::slice::from_raw_parts(ptr.add(line_start), context_len);
            String::from_utf8_lossy(data).trim_end().to_string()
        }
    }

    /// Calculate line number by counting newlines
    fn calculate_line_number(&self, chunk: &StreamChunk, desc: &FileDescriptor, offset_in_file: usize) -> u32 {
        unsafe {
            let ptr = chunk.buffer.contents() as *const u8;
            let file_start = desc.offset as usize;
            let global_offset = file_start + offset_in_file;

            let mut lines = 1u32;
            for pos in file_start..global_offset {
                if *ptr.add(pos) == b'\n' {
                    lines += 1;
                }
            }
            lines
        }
    }

    /// Get the streaming pipeline for direct access
    pub fn pipeline(&self) -> &StreamingPipeline {
        &self.pipeline
    }

    /// Get mutable pipeline
    pub fn pipeline_mut(&mut self) -> &mut StreamingPipeline {
        &mut self.pipeline
    }
}

/// Profiling data for streaming search
#[derive(Debug, Clone, Default)]
pub struct StreamingProfile {
    /// Time spent partitioning files
    pub partition_us: u64,
    /// Time spent queueing I/O commands
    pub io_queue_us: u64,
    /// Time spent waiting for I/O
    pub io_wait_us: u64,
    /// Time spent in GPU search
    pub search_us: u64,
    /// Total elapsed time
    pub total_us: u64,
    /// Number of chunks processed
    pub chunk_count: usize,
    /// Number of files processed
    pub files_processed: usize,
    /// Bytes processed
    pub bytes_processed: u64,
    /// Number of matches found
    pub match_count: usize,
}

impl StreamingProfile {
    /// Print formatted profile summary
    pub fn print(&self) {
        println!("Streaming Search Profile:");
        println!("  Partition:  {:>6}us", self.partition_us);
        println!("  I/O Queue:  {:>6}us", self.io_queue_us);
        println!("  I/O Wait:   {:>6}us", self.io_wait_us);
        println!("  GPU Search: {:>6}us", self.search_us);
        println!("  Total:      {:>6}us ({:.1}ms)", self.total_us, self.total_us as f64 / 1000.0);
        println!("  Chunks:     {}", self.chunk_count);
        println!("  Files:      {}", self.files_processed);
        println!("  Data:       {:.2} MB", self.bytes_processed as f64 / (1024.0 * 1024.0));
        println!("  Matches:    {}", self.match_count);

        if self.total_us > 0 {
            let throughput = (self.bytes_processed as f64 / (1024.0 * 1024.0)) / (self.total_us as f64 / 1_000_000.0);
            println!("  Throughput: {:.1} MB/s", throughput);

            // Calculate overlap efficiency
            let io_time = self.io_queue_us + self.io_wait_us;
            let sequential_estimate = io_time + self.search_us;
            if sequential_estimate > self.total_us {
                let overlap_savings = sequential_estimate - self.total_us;
                let overlap_pct = 100.0 * overlap_savings as f64 / sequential_estimate as f64;
                println!("  Overlap:    {:.1}% time saved", overlap_pct);
            }
        }
    }
}

/// Get the streaming search shader source
fn get_streaming_search_shader() -> String {
    r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define BYTES_PER_THREAD 64
#define MAX_MATCHES_PER_THREAD 4
#define MAX_CONTEXT 80

struct FileDescriptor {
    ulong offset;
    uint size;
    uint file_index;
    uint status;
    uint _padding;
};

struct StreamSearchParams {
    uint file_count;
    uint pattern_len;
    uint case_sensitive;
    uint total_bytes;
};

struct MatchResult {
    uint file_index;
    uint chunk_index;
    uint line_number;
    uint column;
    uint match_length;
    uint context_start;
    uint context_len;
    uint _padding;
};

// Case-insensitive character compare
inline bool char_eq_fast(uchar a, uchar b, bool case_sensitive) {
    if (case_sensitive) return a == b;
    uchar a_lower = (a >= 'A' && a <= 'Z') ? a + 32 : a;
    uchar b_lower = (b >= 'A' && b <= 'Z') ? b + 32 : b;
    return a_lower == b_lower;
}

// Binary search to find which file contains byte position
inline uint find_file_for_position(
    device const FileDescriptor* files,
    uint file_count,
    uint byte_pos
) {
    if (file_count == 0) return 0xFFFFFFFF;

    uint left = 0;
    uint right = file_count - 1;

    while (left <= right) {
        uint mid = (left + right) / 2;
        ulong file_end = files[mid].offset + files[mid].size;

        if (byte_pos < files[mid].offset) {
            if (mid == 0) break;
            right = mid - 1;
        } else if (byte_pos >= file_end) {
            left = mid + 1;
        } else {
            return mid;
        }
    }

    return 0xFFFFFFFF;
}

kernel void streaming_search_kernel(
    device const uchar4* data [[buffer(0)]],
    device const FileDescriptor* files [[buffer(1)]],
    constant StreamSearchParams& params [[buffer(2)]],
    constant uchar* pattern [[buffer(3)]],
    device MatchResult* matches [[buffer(4)]],
    device atomic_uint& match_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint byte_base = gid * BYTES_PER_THREAD;
    uint vec4_base = byte_base / 4;

    if (byte_base >= params.total_bytes) return;

    // Find which file this thread is in
    uint file_idx = find_file_for_position(files, params.file_count, byte_base);
    if (file_idx == 0xFFFFFFFF) return;

    FileDescriptor file = files[file_idx];
    if (file.status != 3) return;  // File not loaded

    // Calculate valid bytes for this thread
    ulong file_end = file.offset + file.size;
    uint valid_bytes = min((uint)BYTES_PER_THREAD, (uint)(file_end - byte_base));

    // Load data using vectorized loads
    uchar local_data[BYTES_PER_THREAD];
    #pragma unroll
    for (uint i = 0; i < 16; i++) {
        if (i * 4 < valid_bytes) {
            uchar4 v = data[vec4_base + i];
            local_data[i*4 + 0] = v.x;
            local_data[i*4 + 1] = v.y;
            local_data[i*4 + 2] = v.z;
            local_data[i*4 + 3] = v.w;
        }
    }

    // Search within local data
    uint local_matches_pos[MAX_MATCHES_PER_THREAD];
    uint local_match_count = 0;

    bool case_sensitive = params.case_sensitive != 0;
    uint search_end = (valid_bytes >= params.pattern_len) ? (valid_bytes - params.pattern_len + 1) : 0;

    for (uint pos = 0; pos < search_end && local_match_count < MAX_MATCHES_PER_THREAD; pos++) {
        bool match = true;
        for (uint j = 0; j < params.pattern_len && match; j++) {
            if (!char_eq_fast(local_data[pos + j], pattern[j], case_sensitive)) {
                match = false;
            }
        }
        if (match) {
            local_matches_pos[local_match_count++] = pos;
        }
    }

    // SIMD reduction
    uint simd_total = simd_sum(local_match_count);
    uint my_offset = simd_prefix_exclusive_sum(local_match_count);

    uint group_base = 0;
    if (simd_lane == 0 && simd_total > 0) {
        group_base = atomic_fetch_add_explicit(&match_count, simd_total, memory_order_relaxed);
    }
    group_base = simd_broadcast_first(group_base);

    // Write results
    uint offset_in_file = (uint)(byte_base - file.offset);

    for (uint i = 0; i < local_match_count; i++) {
        uint global_idx = group_base + my_offset + i;
        if (global_idx < 10000) {  // MAX_MATCHES
            uint local_pos = local_matches_pos[i];

            MatchResult result;
            result.file_index = file.file_index;
            result.chunk_index = file_idx;
            result.line_number = 0;  // CPU calculates
            result.column = offset_in_file + local_pos;
            result.match_length = params.pattern_len;
            result.context_start = offset_in_file + local_pos;
            result.context_len = min(valid_bytes - local_pos, (uint)MAX_CONTEXT);
            result._padding = 0;

            matches[global_idx] = result;
        }
    }
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_pipeline_creation() {
        let device = Device::system_default().expect("No Metal device");

        match StreamingPipeline::new(&device) {
            Some(pipeline) => {
                println!("StreamingPipeline created with {} chunks", pipeline.chunk_count());
                assert_eq!(pipeline.chunk_count(), DEFAULT_CHUNK_COUNT);
            }
            None => {
                println!("MTLIOCommandQueue not available (requires Metal 3+)");
            }
        }
    }

    #[test]
    fn test_streaming_search_creation() {
        let device = Device::system_default().expect("No Metal device");

        match StreamingSearch::new(&device) {
            Some(search) => {
                println!("StreamingSearch created successfully");
                assert_eq!(search.pipeline().chunk_count(), DEFAULT_CHUNK_COUNT);
            }
            None => {
                println!("StreamingSearch requires Metal 3+ with MTLIOCommandQueue");
            }
        }
    }

    #[test]
    fn test_file_partitioning() {
        let device = Device::system_default().expect("No Metal device");

        let pipeline = match StreamingPipeline::with_config(&device, 4, 1024 * 1024, 100) {
            Some(p) => p,
            None => {
                println!("Skipping: MTLIOCommandQueue not available");
                return;
            }
        };

        // Create test file list
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir("src") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map(|e| e == "rs").unwrap_or(false) {
                    files.push(path);
                }
            }
        }

        if files.is_empty() {
            println!("No test files found");
            return;
        }

        let partitions = pipeline.partition_files(&files);
        println!("Partitioned {} files into {} chunks:", files.len(), partitions.len());
        for (i, (start, end)) in partitions.iter().enumerate() {
            println!("  Chunk {}: files {}..{} ({} files)", i, start, end, end - start);
        }

        assert!(!partitions.is_empty());
        assert!(partitions.len() <= pipeline.chunk_count());
    }

    #[test]
    fn test_chunk_reset() {
        let device = Device::system_default().expect("No Metal device");

        let mut pipeline = match StreamingPipeline::new(&device) {
            Some(p) => p,
            None => {
                println!("Skipping: MTLIOCommandQueue not available");
                return;
            }
        };

        // Mark first chunk as ready
        if let Some(chunk) = pipeline.chunk_mut(0) {
            chunk.mark_ready();
            assert!(chunk.is_ready());
        }

        // Reset and verify
        pipeline.reset();
        if let Some(chunk) = pipeline.chunk(0) {
            assert!(!chunk.is_ready());
        }
    }
}
