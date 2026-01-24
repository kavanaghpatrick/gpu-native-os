// GPU Content Search (Issue #50)
//
// grep-like search inside files using GPU parallel processing.
// Uses Boyer-Moore-Horspool algorithm for fast string matching.
// Each GPU thread processes one 4KB chunk.

use super::app::{AppBuilder, APP_SHADER_HEADER};
use metal::*;
use std::fs;
use std::io;
use std::mem;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

/// Chunk size (4KB - standard block size)
const CHUNK_SIZE: usize = 4096;

/// Maximum pattern length
const MAX_PATTERN_LEN: usize = 64;

/// Maximum matches per search
const MAX_MATCHES: usize = 10000;

/// Maximum files to index
const MAX_FILES: usize = 100000;

/// Maximum total chunks (100K files * avg 10 chunks = 1M chunks max)
const MAX_CHUNKS: usize = 1000000;

// ============================================================================
// GPU Structures
// ============================================================================

/// Metadata for each chunk
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ChunkMetadata {
    file_index: u32,      // Which file this chunk belongs to
    chunk_index: u32,     // Which chunk in the file (0-based)
    offset_in_file: u64,  // Byte offset in original file
    chunk_length: u32,    // Actual bytes in this chunk (may be < 4096)
    flags: u32,           // Bit 0: is_text, Bit 1: is_first, Bit 2: is_last
}

/// Search parameters passed to GPU
#[repr(C)]
#[derive(Copy, Clone)]
struct SearchParams {
    chunk_count: u32,
    pattern_len: u32,
    case_sensitive: u32,
    _padding: u32,
}

/// Match result from GPU
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuMatchResult {
    file_index: u32,
    chunk_index: u32,     // Which chunk in the buffer (for context extraction)
    line_number: u32,
    column: u32,
    match_length: u32,
    context_start: u32,   // Offset in chunk where context starts
    context_len: u32,     // Length of context to extract
    _padding: u32,
}

// ============================================================================
// Metal Shader
// ============================================================================

const CONTENT_SEARCH_SHADER: &str = r#"
{{APP_SHADER_HEADER}}

#define CHUNK_SIZE 4096
#define MAX_PATTERN_LEN 64
#define MAX_CONTEXT 80

struct ChunkMetadata {
    uint file_index;
    uint chunk_index;
    ulong offset_in_file;
    uint chunk_length;
    uint flags;  // Bit 0: is_text, Bit 1: is_first, Bit 2: is_last
};

struct SearchParams {
    uint chunk_count;
    uint pattern_len;
    uint case_sensitive;
    uint _padding;
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

// Check if byte is a binary indicator (null byte, control chars except common ones)
bool is_binary_byte(char c) {
    // Allow: tab (9), newline (10), carriage return (13), and printable ASCII (32-126)
    if (c == 0) return true;  // Null byte = binary
    if (c >= 1 && c <= 8) return true;   // Control chars
    if (c >= 14 && c <= 31) return true; // Control chars (except tab, newline, cr)
    if (c == 127) return true;  // DEL
    return false;
}

// Count newlines from start to position to get line number
uint count_lines(device const char* data, uint len, uint pos) {
    uint lines = 1;
    for (uint i = 0; i < pos && i < len; i++) {
        if (data[i] == '\n') lines++;
    }
    return lines;
}

// Find column (chars since last newline)
uint find_column(device const char* data, uint len, uint pos) {
    uint col = 0;
    for (uint i = pos; i > 0; i--) {
        if (data[i-1] == '\n') break;
        col++;
    }
    return col;
}

// Boyer-Moore-Horspool bad character shift table (simplified for GPU)
// Instead of full table, compute shift on-the-fly
uint get_shift(constant const char* pattern, uint pattern_len, char c) {
    // Search for character in pattern (from right, excluding last char)
    for (uint i = pattern_len - 1; i > 0; i--) {
        char p = pattern[i - 1];
        if (p == c) {
            return pattern_len - i;
        }
    }
    return pattern_len;  // Character not in pattern
}

// Case-insensitive character compare
bool char_eq(char a, char b, bool case_sensitive) {
    if (case_sensitive) return a == b;

    char a_lower = a;
    char b_lower = b;
    if (a >= 'A' && a <= 'Z') a_lower = a + 32;
    if (b >= 'A' && b <= 'Z') b_lower = b + 32;
    return a_lower == b_lower;
}

kernel void content_search_kernel(
    device const char* chunks [[buffer(0)]],           // All chunks packed (CHUNK_SIZE each)
    device const ChunkMetadata* metadata [[buffer(1)]], // Metadata for each chunk
    constant SearchParams& params [[buffer(2)]],        // Search parameters
    constant char* pattern [[buffer(3)]],               // Pattern to search (lowercase if case-insensitive)
    device MatchResult* matches [[buffer(4)]],          // Output matches
    device atomic_uint& match_count [[buffer(5)]],      // Number of matches found
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.chunk_count) return;

    // Get this chunk's data and metadata
    device const char* data = chunks + (gid * CHUNK_SIZE);
    ChunkMetadata meta = metadata[gid];
    uint chunk_len = meta.chunk_length;

    if (chunk_len == 0 || params.pattern_len == 0) return;
    if (params.pattern_len > chunk_len) return;

    // Quick binary file check (sample first 512 bytes)
    uint sample_len = min(chunk_len, 512u);
    uint binary_count = 0;
    for (uint i = 0; i < sample_len; i++) {
        if (is_binary_byte(data[i])) {
            binary_count++;
            if (binary_count > 5) return;  // Too many binary bytes, skip
        }
    }

    // Boyer-Moore-Horspool search
    uint pos = 0;
    bool case_sensitive = params.case_sensitive != 0;

    while (pos <= chunk_len - params.pattern_len) {
        // Try to match pattern at current position
        bool match = true;
        for (uint j = params.pattern_len; j > 0; j--) {
            if (!char_eq(data[pos + j - 1], pattern[j - 1], case_sensitive)) {
                match = false;
                // Compute shift based on mismatched character
                char bad_char = data[pos + params.pattern_len - 1];
                if (!case_sensitive && bad_char >= 'A' && bad_char <= 'Z') {
                    bad_char += 32;
                }
                uint shift = get_shift(pattern, params.pattern_len, bad_char);
                pos += shift;
                break;
            }
        }

        if (match) {
            // Found a match! Record it
            uint idx = atomic_fetch_add_explicit(&match_count, 1, memory_order_relaxed);
            if (idx < 10000) {  // MAX_MATCHES
                MatchResult result;
                result.file_index = meta.file_index;
                result.chunk_index = gid;  // GPU thread ID = chunk index (for zero-copy context extraction)
                result.line_number = count_lines(data, chunk_len, pos);
                result.column = find_column(data, chunk_len, pos);
                result.match_length = params.pattern_len;

                // Find context (start of line to ~40 chars after match or end of line)
                uint context_start = pos;
                for (uint i = pos; i > 0; i--) {
                    if (data[i-1] == '\n') break;
                    context_start = i - 1;
                }
                uint context_end = pos + params.pattern_len;
                for (uint i = context_end; i < chunk_len && i < pos + MAX_CONTEXT; i++) {
                    context_end = i + 1;
                    if (data[i] == '\n') break;
                }

                result.context_start = context_start;
                result.context_len = min(context_end - context_start, (uint)MAX_CONTEXT);
                result._padding = 0;

                matches[idx] = result;
            }
            pos++;  // Move past this match to find more
        }
    }
}
"#;

fn get_content_search_shader() -> String {
    CONTENT_SEARCH_SHADER.replace("{{APP_SHADER_HEADER}}", APP_SHADER_HEADER)
}

// ============================================================================
// Content Match Result (public API)
// ============================================================================

/// A match found in file contents
#[derive(Debug, Clone)]
pub struct ContentMatch {
    pub file_path: String,
    pub line_number: u32,
    pub column: u32,
    pub context: String,    // The line containing the match
    pub match_start: usize, // Position of match within context
}

/// Options for content search
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub case_sensitive: bool,
    pub max_results: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            max_results: 1000,
        }
    }
}

// ============================================================================
// GPU Content Search
// ============================================================================

/// GPU-accelerated content search (grep-like)
///
/// Searches file contents in parallel using Metal compute shaders.
/// Each GPU thread processes one 4KB chunk.
pub struct GpuContentSearch {
    device: Device,
    command_queue: CommandQueue,
    search_pipeline: ComputePipelineState,

    // Chunk storage
    chunks_buffer: Buffer,       // All chunks (packed, CHUNK_SIZE each)
    metadata_buffer: Buffer,     // ChunkMetadata for each chunk

    // Search parameters
    params_buffer: Buffer,       // SearchParams
    pattern_buffer: Buffer,      // Search pattern

    // Results
    matches_buffer: Buffer,      // MatchResult array
    match_count_buffer: Buffer,  // Atomic counter

    // State
    max_chunks: usize,
    current_chunk_count: usize,
    file_paths: Vec<String>,     // File paths for result lookup
    chunk_data: Vec<u8>,         // Chunk data (CPU copy for context extraction)
}

impl GpuContentSearch {
    /// Create a new GPU content search engine
    pub fn new(device: &Device, max_files: usize) -> Result<Self, String> {
        let max_chunks = max_files * 10; // Assume avg 10 chunks per file
        let max_chunks = max_chunks.min(MAX_CHUNKS);

        let builder = AppBuilder::new(device, "GpuContentSearch");
        let command_queue = device.new_command_queue();

        // Compile shader
        let library = builder.compile_library(&get_content_search_shader())?;
        let search_pipeline = builder.create_compute_pipeline(&library, "content_search_kernel")?;

        // Allocate buffers
        let chunks_buffer = builder.create_buffer(max_chunks * CHUNK_SIZE);
        let metadata_buffer = builder.create_buffer(max_chunks * mem::size_of::<ChunkMetadata>());
        let params_buffer = builder.create_buffer(mem::size_of::<SearchParams>());
        let pattern_buffer = builder.create_buffer(MAX_PATTERN_LEN);
        let matches_buffer = builder.create_buffer(MAX_MATCHES * mem::size_of::<GpuMatchResult>());
        let match_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        Ok(Self {
            device: device.clone(),
            command_queue,
            search_pipeline,
            chunks_buffer,
            metadata_buffer,
            params_buffer,
            pattern_buffer,
            matches_buffer,
            match_count_buffer,
            max_chunks,
            current_chunk_count: 0,
            file_paths: Vec::with_capacity(max_files.min(MAX_FILES)),
            chunk_data: Vec::new(),
        })
    }

    /// Load files into GPU buffers for searching
    ///
    /// Returns the number of chunks loaded
    pub fn load_files(&mut self, paths: &[&Path]) -> Result<usize, io::Error> {
        self.file_paths.clear();
        self.current_chunk_count = 0;
        self.chunk_data.clear();

        let mut total_chunks = 0;
        let mut metadata_vec: Vec<ChunkMetadata> = Vec::new();

        for path in paths.iter() {
            if total_chunks >= self.max_chunks {
                break;
            }

            // Read file
            let content = match fs::read(path) {
                Ok(c) => c,
                Err(_) => continue, // Skip unreadable files
            };

            // Skip empty files
            if content.is_empty() {
                continue;
            }

            // Skip very large files (> 10MB)
            if content.len() > 10 * 1024 * 1024 {
                continue;
            }

            // Store file path
            self.file_paths.push(path.to_string_lossy().to_string());
            let actual_file_index = self.file_paths.len() - 1;

            // Split into chunks
            let num_chunks = (content.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
            for chunk_index in 0..num_chunks {
                if total_chunks >= self.max_chunks {
                    break;
                }

                let offset = chunk_index * CHUNK_SIZE;
                let chunk_len = (content.len() - offset).min(CHUNK_SIZE);
                let chunk = &content[offset..offset + chunk_len];

                // Add to chunk data (padded to CHUNK_SIZE)
                let chunk_start = self.chunk_data.len();
                self.chunk_data.extend_from_slice(chunk);
                self.chunk_data.resize(chunk_start + CHUNK_SIZE, 0);

                // Create metadata
                let mut flags = 1u32; // is_text (assume text, GPU will verify)
                if chunk_index == 0 {
                    flags |= 2; // is_first
                }
                if chunk_index == num_chunks - 1 {
                    flags |= 4; // is_last
                }

                metadata_vec.push(ChunkMetadata {
                    file_index: actual_file_index as u32,
                    chunk_index: chunk_index as u32,
                    offset_in_file: offset as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                });

                total_chunks += 1;
            }
        }

        self.current_chunk_count = total_chunks;

        // Copy to GPU buffers
        unsafe {
            // Copy chunk data
            let chunk_ptr = self.chunks_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                self.chunk_data.as_ptr(),
                chunk_ptr,
                self.chunk_data.len(),
            );

            // Copy metadata
            let meta_ptr = self.metadata_buffer.contents() as *mut ChunkMetadata;
            for (i, meta) in metadata_vec.iter().enumerate() {
                *meta_ptr.add(i) = *meta;
            }
        }

        Ok(total_chunks)
    }

    /// Load files from pre-mapped buffers (ZERO CPU COPIES!)
    ///
    /// Takes mmap buffers that are already GPU-accessible and copies them
    /// to the chunks buffer using GPU blit (GPU-to-GPU, no CPU involvement).
    ///
    /// This is the GPU-native path: Disk → mmap → GPU blit → Search
    /// vs the CPU path: Disk → CPU read → CPU Vec → GPU copy → Search
    pub fn load_from_mmap(&mut self, buffers: &[(String, &super::mmap_buffer::MmapBuffer)]) -> Result<usize, String> {
        self.file_paths.clear();
        self.current_chunk_count = 0;
        self.chunk_data.clear();

        let mut total_chunks = 0;
        let mut metadata_vec: Vec<ChunkMetadata> = Vec::new();
        let mut gpu_offset = 0usize;

        // Create command buffer for GPU blits
        let command_buffer = self.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        for (path, mmap) in buffers.iter() {
            if total_chunks >= self.max_chunks {
                break;
            }

            let content_len = mmap.file_size();
            if content_len == 0 || content_len > 10 * 1024 * 1024 {
                continue;
            }

            // Store file path
            self.file_paths.push(path.clone());
            let actual_file_index = self.file_paths.len() - 1;

            // Calculate chunks needed
            let num_chunks = (content_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for chunk_index in 0..num_chunks {
                if total_chunks >= self.max_chunks {
                    break;
                }

                let offset_in_file = chunk_index * CHUNK_SIZE;
                let chunk_len = (content_len - offset_in_file).min(CHUNK_SIZE);

                // GPU blit from mmap buffer to chunks buffer
                // Source: mmap buffer at offset_in_file
                // Dest: chunks_buffer at gpu_offset
                blit_encoder.copy_from_buffer(
                    mmap.metal_buffer(),
                    offset_in_file as u64,
                    &self.chunks_buffer,
                    gpu_offset as u64,
                    chunk_len as u64,
                );

                // Create metadata
                let mut flags = 1u32; // is_text
                if chunk_index == 0 {
                    flags |= 2; // is_first
                }
                if chunk_index == num_chunks - 1 {
                    flags |= 4; // is_last
                }

                metadata_vec.push(ChunkMetadata {
                    file_index: actual_file_index as u32,
                    chunk_index: chunk_index as u32,
                    offset_in_file: offset_in_file as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                });

                gpu_offset += CHUNK_SIZE; // Padded chunks
                total_chunks += 1;
            }
        }

        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.current_chunk_count = total_chunks;

        // Copy metadata to GPU (still needed, but small)
        unsafe {
            let meta_ptr = self.metadata_buffer.contents() as *mut ChunkMetadata;
            for (i, meta) in metadata_vec.iter().enumerate() {
                *meta_ptr.add(i) = *meta;
            }
        }

        Ok(total_chunks)
    }

    /// Load files from GpuBatchLoader result (MTLIOCommandQueue - TRUE GPU-DIRECT I/O!)
    ///
    /// Takes a BatchLoadResult from GpuBatchLoader and copies to chunks buffer.
    /// This uses MTLIOCommandQueue which loads files directly to GPU without page faults.
    pub fn load_from_batch(&mut self, result: &super::batch_io::BatchLoadResult) -> Result<usize, String> {
        self.file_paths.clear();
        self.current_chunk_count = 0;
        self.chunk_data.clear();

        let mut total_chunks = 0;
        let mut metadata_vec: Vec<ChunkMetadata> = Vec::new();
        let mut gpu_offset = 0usize;

        // Create command buffer for GPU blits
        let command_buffer = self.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        for i in 0..result.file_count() {
            if total_chunks >= self.max_chunks {
                break;
            }

            let desc = match result.descriptor(i) {
                Some(d) => d,
                None => continue,
            };

            // Skip failed loads
            if desc.status != 3 {
                continue;
            }

            let content_len = desc.size as usize;
            if content_len == 0 || content_len > 10 * 1024 * 1024 {
                continue;
            }

            // Store file path
            let path = result.file_paths.get(i)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            self.file_paths.push(path);
            let actual_file_index = self.file_paths.len() - 1;

            // Calculate chunks needed
            let num_chunks = (content_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for chunk_index in 0..num_chunks {
                if total_chunks >= self.max_chunks {
                    break;
                }

                let offset_in_file = chunk_index * CHUNK_SIZE;
                let chunk_len = (content_len - offset_in_file).min(CHUNK_SIZE);

                // GPU blit from mega-buffer to chunks buffer
                blit_encoder.copy_from_buffer(
                    &result.mega_buffer,
                    (desc.offset as usize + offset_in_file) as u64,
                    &self.chunks_buffer,
                    gpu_offset as u64,
                    chunk_len as u64,
                );

                // Create metadata
                let mut flags = 1u32; // is_text
                if chunk_index == 0 {
                    flags |= 2; // is_first
                }
                if chunk_index == num_chunks - 1 {
                    flags |= 4; // is_last
                }

                metadata_vec.push(ChunkMetadata {
                    file_index: actual_file_index as u32,
                    chunk_index: chunk_index as u32,
                    offset_in_file: offset_in_file as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                });

                gpu_offset += CHUNK_SIZE; // Padded chunks
                total_chunks += 1;
            }
        }

        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.current_chunk_count = total_chunks;

        // Copy metadata to GPU
        unsafe {
            let meta_ptr = self.metadata_buffer.contents() as *mut ChunkMetadata;
            for (i, meta) in metadata_vec.iter().enumerate() {
                *meta_ptr.add(i) = *meta;
            }
        }

        Ok(total_chunks)
    }

    /// Search for pattern in loaded files
    pub fn search(&self, pattern: &str, options: &SearchOptions) -> Vec<ContentMatch> {
        if pattern.is_empty() || self.current_chunk_count == 0 {
            return vec![];
        }

        if pattern.len() > MAX_PATTERN_LEN {
            return vec![]; // Pattern too long
        }

        // Prepare pattern (lowercase if case-insensitive)
        let pattern_bytes: Vec<u8> = if options.case_sensitive {
            pattern.as_bytes().to_vec()
        } else {
            pattern.to_lowercase().as_bytes().to_vec()
        };

        // Write search parameters
        unsafe {
            // Reset match count
            let count_ptr = self.match_count_buffer.contents() as *mut u32;
            *count_ptr = 0;

            // Write params
            let params_ptr = self.params_buffer.contents() as *mut SearchParams;
            *params_ptr = SearchParams {
                chunk_count: self.current_chunk_count as u32,
                pattern_len: pattern_bytes.len() as u32,
                case_sensitive: if options.case_sensitive { 1 } else { 0 },
                _padding: 0,
            };

            // Write pattern
            let pattern_ptr = self.pattern_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                pattern_bytes.as_ptr(),
                pattern_ptr,
                pattern_bytes.len(),
            );
        }

        // Dispatch GPU search
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.search_pipeline);
        encoder.set_buffer(0, Some(&self.chunks_buffer), 0);
        encoder.set_buffer(1, Some(&self.metadata_buffer), 0);
        encoder.set_buffer(2, Some(&self.params_buffer), 0);
        encoder.set_buffer(3, Some(&self.pattern_buffer), 0);
        encoder.set_buffer(4, Some(&self.matches_buffer), 0);
        encoder.set_buffer(5, Some(&self.match_count_buffer), 0);

        // Dispatch: one thread per chunk
        let threads_per_group = 256;
        let thread_groups = (self.current_chunk_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let mut results: Vec<ContentMatch> = Vec::new();
        unsafe {
            let count = *(self.match_count_buffer.contents() as *const u32);
            let matches = self.matches_buffer.contents() as *const GpuMatchResult;

            let result_count = (count as usize).min(options.max_results).min(MAX_MATCHES);
            for i in 0..result_count {
                let m = *matches.add(i);

                if (m.file_index as usize) >= self.file_paths.len() {
                    continue;
                }

                // Extract context from CPU copy of chunk data
                let context = self.extract_context(&m);

                results.push(ContentMatch {
                    file_path: self.file_paths[m.file_index as usize].clone(),
                    line_number: m.line_number,
                    column: m.column,
                    context,
                    match_start: m.column as usize,
                });
            }
        }

        // Sort by file path, then line number
        results.sort_by(|a, b| {
            a.file_path.cmp(&b.file_path)
                .then(a.line_number.cmp(&b.line_number))
        });

        results
    }

    /// Extract context string from GPU buffer (ZERO CPU FILE READ!)
    ///
    /// Reads directly from the chunks_buffer which already has the data.
    /// The GPU shader provides chunk_index, context_start and context_len.
    fn extract_context(&self, match_result: &GpuMatchResult) -> String {
        let chunk_index = match_result.chunk_index as usize;
        let context_start = match_result.context_start as usize;
        let context_len = match_result.context_len as usize;

        if context_len == 0 || context_len > 256 || chunk_index >= self.current_chunk_count {
            return String::new();
        }

        // Read directly from GPU buffer - the GPU gave us exact coordinates!
        unsafe {
            let chunks_ptr = self.chunks_buffer.contents() as *const u8;
            let chunk_base = chunk_index * CHUNK_SIZE;

            // Bounds check
            if context_start + context_len > CHUNK_SIZE {
                return String::new();
            }

            let data = std::slice::from_raw_parts(
                chunks_ptr.add(chunk_base + context_start),
                context_len,
            );

            // Convert to UTF-8, replacing invalid sequences
            String::from_utf8_lossy(data).trim_end().to_string()
        }
    }

    /// Get number of loaded chunks
    pub fn chunk_count(&self) -> usize {
        self.current_chunk_count
    }

    /// Get number of loaded files
    pub fn file_count(&self) -> usize {
        self.file_paths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_metadata_size() {
        // Should be 24 bytes with padding
        assert!(mem::size_of::<ChunkMetadata>() <= 32);
    }

    #[test]
    fn test_search_params_size() {
        assert_eq!(mem::size_of::<SearchParams>(), 16);
    }

    #[test]
    fn test_match_result_size() {
        assert_eq!(mem::size_of::<GpuMatchResult>(), 32);
    }

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert!(!opts.case_sensitive);
        assert_eq!(opts.max_results, 1000);
    }
}
