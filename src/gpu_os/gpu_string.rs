// GPU String Processing (Issue #79)
//
// All string operations run on GPU - no CPU string processing in hot path.
//
// Operations:
// - Query tokenization: "foo BAR baz" → ["foo", "bar", "baz"]
// - Case conversion: uppercase → lowercase
// - Path parsing: extract filename, directory, extension
//
// Philosophy: The GPU is the computer. Strings are just byte arrays.
//             Every character can be processed in parallel.

use metal::*;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use super::app::AppBuilder;

// ============================================================================
// Constants
// ============================================================================

/// Maximum query length (bytes)
pub const MAX_QUERY_LEN: usize = 256;

/// Maximum words in a query
pub const MAX_QUERY_WORDS: usize = 8;

/// Maximum word length
pub const MAX_WORD_LEN: usize = 32;

/// Maximum path length for parsing
pub const MAX_PATH_LEN: usize = 256;

// ============================================================================
// Data Structures (GPU-compatible)
// ============================================================================

/// A parsed word from query tokenization
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuWord {
    pub chars: [u8; MAX_WORD_LEN],  // Lowercased word bytes
    pub len: u16,                    // Actual length
    pub start_offset: u16,           // Position in original query
}

/// Tokenization result
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct TokenizeResult {
    pub word_count: u32,
    pub _padding: [u32; 3],
}

/// Parsed path components
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ParsedPath {
    pub filename_start: u16,    // Offset where filename begins
    pub filename_len: u16,      // Length of filename
    pub extension_start: u16,   // Offset where extension begins (after last '.')
    pub extension_len: u16,     // Length of extension
    pub depth: u16,             // Directory depth (number of '/')
    pub flags: u16,             // is_hidden, is_directory, etc.
}

/// Flags for ParsedPath
pub const PATH_FLAG_HIDDEN: u16 = 1;      // Starts with '.'
pub const PATH_FLAG_DIRECTORY: u16 = 2;   // Ends with '/'
pub const PATH_FLAG_HAS_EXTENSION: u16 = 4;

// ============================================================================
// GPU String Processor
// ============================================================================

/// GPU-accelerated string processing
///
/// All operations run entirely on GPU - CPU just submits raw bytes.
pub struct GpuStringProcessor {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,

    // Pipelines
    tokenize_pipeline: ComputePipelineState,
    parse_path_pipeline: ComputePipelineState,

    // Buffers
    query_buffer: Buffer,           // Raw query input (MAX_QUERY_LEN bytes)
    words_buffer: Buffer,           // Output: GpuWord array
    result_buffer: Buffer,          // Output: TokenizeResult

    path_buffer: Buffer,            // Raw path input (MAX_PATH_LEN bytes)
    parsed_path_buffer: Buffer,     // Output: ParsedPath

    // Batch path parsing
    #[allow(dead_code)]
    paths_batch_buffer: Buffer,     // Multiple paths for batch parsing
    #[allow(dead_code)]
    parsed_batch_buffer: Buffer,    // Multiple ParsedPath results

    // Async support
    #[allow(dead_code)]
    shared_event: SharedEvent,
    #[allow(dead_code)]
    next_signal: Arc<AtomicU64>,
}

impl GpuStringProcessor {
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "GpuStringProcessor");
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = builder.compile_library(&get_gpu_string_shader())?;
        let tokenize_pipeline = builder.create_compute_pipeline(&library, "tokenize_query_kernel")?;
        let parse_path_pipeline = builder.create_compute_pipeline(&library, "parse_path_kernel")?;

        // Allocate buffers
        let query_buffer = builder.create_buffer(MAX_QUERY_LEN);
        let words_buffer = builder.create_buffer(MAX_QUERY_WORDS * mem::size_of::<GpuWord>());
        let result_buffer = builder.create_buffer(mem::size_of::<TokenizeResult>());

        let path_buffer = builder.create_buffer(MAX_PATH_LEN);
        let parsed_path_buffer = builder.create_buffer(mem::size_of::<ParsedPath>());

        // Batch buffers (for parsing many paths at once)
        let batch_size = 10000;
        let paths_batch_buffer = builder.create_buffer(batch_size * MAX_PATH_LEN);
        let parsed_batch_buffer = builder.create_buffer(batch_size * mem::size_of::<ParsedPath>());

        let shared_event = device.new_shared_event();

        Ok(Self {
            device: device.clone(),
            command_queue,
            tokenize_pipeline,
            parse_path_pipeline,
            query_buffer,
            words_buffer,
            result_buffer,
            path_buffer,
            parsed_path_buffer,
            paths_batch_buffer,
            parsed_batch_buffer,
            shared_event,
            next_signal: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Tokenize a query string entirely on GPU
    ///
    /// Input: "Foo BAR  baz" (raw bytes)
    /// Output: ["foo", "bar", "baz"] (lowercased, whitespace-split)
    ///
    /// CPU work: ONE memcpy. GPU does all parsing.
    pub fn tokenize_query(&self, query: &str) -> Vec<GpuWord> {
        let bytes = query.as_bytes();
        let len = bytes.len().min(MAX_QUERY_LEN);

        if len == 0 {
            return vec![];
        }

        // Copy raw query bytes to GPU buffer (THE ONLY CPU WORK)
        unsafe {
            let ptr = self.query_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
            // Zero the rest
            std::ptr::write_bytes(ptr.add(len), 0, MAX_QUERY_LEN - len);

            // Reset result counter
            let result = self.result_buffer.contents() as *mut TokenizeResult;
            (*result).word_count = 0;
        }

        // Dispatch GPU kernel
        let command_buffer = self.command_queue.new_command_buffer();
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.tokenize_pipeline);
            encoder.set_buffer(0, Some(&self.query_buffer), 0);
            encoder.set_buffer(1, Some(&self.words_buffer), 0);
            encoder.set_buffer(2, Some(&self.result_buffer), 0);
            encoder.set_bytes(3, mem::size_of::<u32>() as u64, &(len as u32) as *const _ as *const _);

            // One thread per character
            let threads = MTLSize::new(len as u64, 1, 1);
            let threadgroup_size = MTLSize::new(len.min(256) as u64, 1, 1);
            encoder.dispatch_threads(threads, threadgroup_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        unsafe {
            let result = &*(self.result_buffer.contents() as *const TokenizeResult);
            let word_count = (result.word_count as usize).min(MAX_QUERY_WORDS);

            let words_ptr = self.words_buffer.contents() as *const GpuWord;
            (0..word_count).map(|i| *words_ptr.add(i)).collect()
        }
    }

    /// Parse a path to extract filename, extension, depth - entirely on GPU
    pub fn parse_path(&self, path: &str) -> ParsedPath {
        let bytes = path.as_bytes();
        let len = bytes.len().min(MAX_PATH_LEN);

        if len == 0 {
            return ParsedPath::default();
        }

        // Copy raw path bytes (THE ONLY CPU WORK)
        unsafe {
            let ptr = self.path_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
            std::ptr::write_bytes(ptr.add(len), 0, MAX_PATH_LEN - len);
        }

        let command_buffer = self.command_queue.new_command_buffer();
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.parse_path_pipeline);
            encoder.set_buffer(0, Some(&self.path_buffer), 0);
            encoder.set_buffer(1, Some(&self.parsed_path_buffer), 0);
            encoder.set_bytes(2, mem::size_of::<u32>() as u64, &(len as u32) as *const _ as *const _);

            // Single threadgroup - path parsing needs coordination
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe { *(self.parsed_path_buffer.contents() as *const ParsedPath) }
    }

    /// Get the words buffer for direct use in search kernel
    pub fn words_buffer(&self) -> &Buffer {
        &self.words_buffer
    }

    /// Get the result buffer (contains word_count)
    pub fn result_buffer(&self) -> &Buffer {
        &self.result_buffer
    }
}

// ============================================================================
// GPU Shader Source
// ============================================================================

fn get_gpu_string_shader() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

// Constants matching Rust side
constant uint MAX_QUERY_LEN = 256;
constant uint MAX_QUERY_WORDS = 8;
constant uint MAX_WORD_LEN = 32;
constant uint MAX_PATH_LEN = 256;

// Flags for ParsedPath
constant ushort PATH_FLAG_HIDDEN = 1;
constant ushort PATH_FLAG_DIRECTORY = 2;
constant ushort PATH_FLAG_HAS_EXTENSION = 4;

// ============================================================================
// Data Structures
// ============================================================================

struct GpuWord {
    uchar chars[32];    // MAX_WORD_LEN
    ushort len;
    ushort start_offset;
};

struct TokenizeResult {
    atomic_uint word_count;
    uint _padding[3];
};

struct ParsedPath {
    ushort filename_start;
    ushort filename_len;
    ushort extension_start;
    ushort extension_len;
    ushort depth;
    ushort flags;
};

// ============================================================================
// Helper Functions
// ============================================================================

inline bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline char to_lowercase(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c + 32;
    }
    return c;
}

// ============================================================================
// Tokenize Query Kernel
// ============================================================================
//
// Each thread processes one character position.
// Threads at word boundaries atomically claim a word slot and copy the word.
//
// Input:  "Foo BAR  baz"
// Output: GpuWord[0] = "foo", GpuWord[1] = "bar", GpuWord[2] = "baz"

kernel void tokenize_query_kernel(
    device const char* query [[buffer(0)]],
    device GpuWord* words [[buffer(1)]],
    device TokenizeResult* result [[buffer(2)]],
    constant uint& query_len [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= query_len) return;

    char c = query[tid];
    char prev = (tid > 0) ? query[tid - 1] : ' ';

    bool c_is_space = is_whitespace(c);
    bool prev_is_space = is_whitespace(prev) || tid == 0;

    // Word starts where non-space follows space
    if (!c_is_space && prev_is_space) {
        // Claim a word slot atomically
        uint word_idx = atomic_fetch_add_explicit(&result->word_count, 1, memory_order_relaxed);

        if (word_idx >= MAX_QUERY_WORDS) {
            // Too many words - decrement and bail
            atomic_fetch_sub_explicit(&result->word_count, 1, memory_order_relaxed);
            return;
        }

        // Find word end and copy with lowercase
        uint word_len = 0;
        for (uint i = tid; i < query_len && word_len < MAX_WORD_LEN - 1; i++) {
            char ch = query[i];
            if (is_whitespace(ch)) break;

            words[word_idx].chars[word_len++] = to_lowercase(ch);
        }

        words[word_idx].len = word_len;
        words[word_idx].start_offset = tid;
    }
}

// ============================================================================
// Parse Path Kernel
// ============================================================================
//
// Extracts: filename, extension, depth, flags
// Single thread scans the path (paths are small, parallelism overhead not worth it)

kernel void parse_path_kernel(
    device const char* path [[buffer(0)]],
    device ParsedPath* result [[buffer(1)]],
    constant uint& path_len [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread

    // Initialize
    result->filename_start = 0;
    result->filename_len = 0;
    result->extension_start = 0;
    result->extension_len = 0;
    result->depth = 0;
    result->flags = 0;

    if (path_len == 0) return;

    // Find last slash and count depth
    int last_slash = -1;
    ushort depth = 0;
    int last_dot = -1;

    for (uint i = 0; i < path_len; i++) {
        char c = path[i];
        if (c == '/') {
            last_slash = i;
            depth++;
        } else if (c == '.') {
            last_dot = i;
        }
    }

    result->depth = depth;

    // Filename starts after last slash
    uint filename_start = (last_slash >= 0) ? last_slash + 1 : 0;
    result->filename_start = filename_start;

    // Check for trailing slash (directory)
    if (path_len > 0 && path[path_len - 1] == '/') {
        result->flags |= PATH_FLAG_DIRECTORY;
        result->filename_len = 0;
    } else {
        result->filename_len = path_len - filename_start;
    }

    // Check for hidden file (starts with .)
    if (filename_start < path_len && path[filename_start] == '.') {
        result->flags |= PATH_FLAG_HIDDEN;
    }

    // Extension (only if dot is after filename start and not at start)
    if (last_dot > (int)filename_start) {
        result->flags |= PATH_FLAG_HAS_EXTENSION;
        result->extension_start = last_dot + 1;
        result->extension_len = path_len - last_dot - 1;
    }
}

// ============================================================================
// Batch Path Parse Kernel
// ============================================================================
//
// Parse many paths in parallel - one thread per path

kernel void parse_paths_batch_kernel(
    device const char* paths [[buffer(0)]],      // Packed paths, MAX_PATH_LEN each
    device ParsedPath* results [[buffer(1)]],
    constant uint& path_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= path_count) return;

    // Each path is at offset tid * MAX_PATH_LEN
    device const char* path = paths + tid * MAX_PATH_LEN;
    device ParsedPath* result = results + tid;

    // Find actual path length (null-terminated or MAX_PATH_LEN)
    uint path_len = 0;
    for (uint i = 0; i < MAX_PATH_LEN; i++) {
        if (path[i] == 0) break;
        path_len++;
    }

    // Initialize
    result->filename_start = 0;
    result->filename_len = 0;
    result->extension_start = 0;
    result->extension_len = 0;
    result->depth = 0;
    result->flags = 0;

    if (path_len == 0) return;

    // Find last slash and count depth
    int last_slash = -1;
    ushort depth = 0;
    int last_dot = -1;

    for (uint i = 0; i < path_len; i++) {
        char c = path[i];
        if (c == '/') {
            last_slash = i;
            depth++;
        } else if (c == '.') {
            last_dot = i;
        }
    }

    result->depth = depth;

    uint filename_start = (last_slash >= 0) ? last_slash + 1 : 0;
    result->filename_start = filename_start;

    if (path_len > 0 && path[path_len - 1] == '/') {
        result->flags |= PATH_FLAG_DIRECTORY;
        result->filename_len = 0;
    } else {
        result->filename_len = path_len - filename_start;
    }

    if (filename_start < path_len && path[filename_start] == '.') {
        result->flags |= PATH_FLAG_HIDDEN;
    }

    if (last_dot > (int)filename_start) {
        result->flags |= PATH_FLAG_HAS_EXTENSION;
        result->extension_start = last_dot + 1;
        result->extension_len = path_len - last_dot - 1;
    }
}

"#.to_string()
}
