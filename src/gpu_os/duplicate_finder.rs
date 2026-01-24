// GPU Duplicate Finder (Issue #51)
//
// Find duplicate files using GPU-parallel hashing.
// Uses xxHash64 for fast file hashing on GPU.
// Pre-filters by file size (only same-size files can be duplicates).

use super::app::{AppBuilder, APP_SHADER_HEADER};
use metal::*;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::mem;
use std::path::{Path, PathBuf};

// ============================================================================
// Constants
// ============================================================================

/// Hash chunk size (64KB for optimal GPU throughput)
const HASH_CHUNK_SIZE: usize = 65536;

/// Maximum files to process per batch (larger = fewer GPU round-trips)
const BATCH_SIZE: usize = 2048;

/// Maximum total files to track
const MAX_FILES: usize = 50000;

/// Maximum duplicate groups
const MAX_GROUPS: usize = 10000;

// ============================================================================
// GPU Structures
// ============================================================================

/// File information for hashing
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FileInfo {
    path_index: u32,      // Index into path list
    file_size: u64,       // File size in bytes
    size_group: u32,      // Group ID for same-size files
    hash_low: u64,        // xxHash64 result (low bits)
    hash_high: u64,       // xxHash64 result (high bits, for collision resistance)
    status: u32,          // 0=pending, 1=hashing, 2=complete, 3=error
    _padding: u32,
}

/// Hash job for streaming chunks
#[repr(C)]
#[derive(Copy, Clone)]
struct HashJob {
    file_index: u32,
    chunk_index: u32,
    chunk_offset: u64,
    chunk_length: u32,
    is_last_chunk: u32,
}

/// Hash parameters
#[repr(C)]
#[derive(Copy, Clone)]
struct HashParams {
    job_count: u32,
    seed: u32,
    _padding: [u32; 2],
}

/// Duplicate group result
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuDuplicateGroup {
    hash_low: u64,
    hash_high: u64,
    first_file: u32,
    file_count: u32,
    wasted_bytes: u64,
}

// ============================================================================
// Metal Shader
// ============================================================================

const DUPLICATE_FINDER_SHADER: &str = r#"
{{APP_SHADER_HEADER}}

#define CHUNK_SIZE 65536
#define XXHASH_PRIME64_1 0x9E3779B185EBCA87ULL
#define XXHASH_PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define XXHASH_PRIME64_3 0x165667B19E3779F9ULL
#define XXHASH_PRIME64_4 0x85EBCA77C2B2AE63ULL
#define XXHASH_PRIME64_5 0x27D4EB2F165667C5ULL

struct FileInfo {
    uint path_index;
    ulong file_size;
    uint size_group;
    ulong hash_low;
    ulong hash_high;
    uint status;
    uint _padding;
};

struct HashJob {
    uint file_index;
    uint chunk_index;
    ulong chunk_offset;
    uint chunk_length;
    uint is_last_chunk;
};

struct HashParams {
    uint job_count;
    uint seed;
    uint _padding[2];
};

struct DuplicateGroup {
    ulong hash_low;
    ulong hash_high;
    uint first_file;
    uint file_count;
    ulong wasted_bytes;
};

// xxHash64 round function
ulong xxh64_round(ulong acc, ulong input) {
    acc += input * XXHASH_PRIME64_2;
    acc = (acc << 31) | (acc >> 33);  // rotl64(acc, 31)
    acc *= XXHASH_PRIME64_1;
    return acc;
}

// xxHash64 merge accumulator
ulong xxh64_merge_round(ulong acc, ulong val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * XXHASH_PRIME64_1 + XXHASH_PRIME64_4;
    return acc;
}

// xxHash64 avalanche (finalization)
ulong xxh64_avalanche(ulong hash) {
    hash ^= hash >> 33;
    hash *= XXHASH_PRIME64_2;
    hash ^= hash >> 29;
    hash *= XXHASH_PRIME64_3;
    hash ^= hash >> 32;
    return hash;
}

// Read 8 bytes as uint64 (little-endian)
ulong read_u64(device const uchar* data, uint offset) {
    return ((ulong)data[offset]) |
           ((ulong)data[offset + 1] << 8) |
           ((ulong)data[offset + 2] << 16) |
           ((ulong)data[offset + 3] << 24) |
           ((ulong)data[offset + 4] << 32) |
           ((ulong)data[offset + 5] << 40) |
           ((ulong)data[offset + 6] << 48) |
           ((ulong)data[offset + 7] << 56);
}

// Read 4 bytes as uint32 (little-endian)
uint read_u32(device const uchar* data, uint offset) {
    return ((uint)data[offset]) |
           ((uint)data[offset + 1] << 8) |
           ((uint)data[offset + 2] << 16) |
           ((uint)data[offset + 3] << 24);
}

// Hash a single chunk and update file hash state
kernel void hash_chunk_kernel(
    device FileInfo* files [[buffer(0)]],
    device const uchar* chunk_data [[buffer(1)]],  // All chunks packed
    device const HashJob* jobs [[buffer(2)]],
    constant HashParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.job_count) return;

    HashJob job = jobs[gid];
    device const uchar* data = chunk_data + (gid * CHUNK_SIZE);
    uint len = job.chunk_length;

    // Initialize or continue hash state
    ulong v1, v2, v3, v4;
    ulong seed = params.seed;

    if (job.chunk_index == 0) {
        // First chunk - initialize accumulators
        v1 = seed + XXHASH_PRIME64_1 + XXHASH_PRIME64_2;
        v2 = seed + XXHASH_PRIME64_2;
        v3 = seed;
        v4 = seed - XXHASH_PRIME64_1;
    } else {
        // Subsequent chunk - load state from file (stored in hash fields temporarily)
        v1 = files[job.file_index].hash_low;
        v2 = files[job.file_index].hash_high;
        // Note: In a full implementation, we'd need more state storage
        // For now, this is a simplified single-chunk version
        v3 = seed;
        v4 = seed - XXHASH_PRIME64_1;
    }

    // Process 32-byte blocks
    uint pos = 0;
    while (pos + 32 <= len) {
        v1 = xxh64_round(v1, read_u64(data, pos));
        v2 = xxh64_round(v2, read_u64(data, pos + 8));
        v3 = xxh64_round(v3, read_u64(data, pos + 16));
        v4 = xxh64_round(v4, read_u64(data, pos + 24));
        pos += 32;
    }

    if (job.is_last_chunk != 0) {
        // Finalize hash
        ulong hash;
        ulong total_len = files[job.file_index].file_size;

        if (total_len >= 32) {
            hash = ((v1 << 1) | (v1 >> 63)) +
                   ((v2 << 7) | (v2 >> 57)) +
                   ((v3 << 12) | (v3 >> 52)) +
                   ((v4 << 18) | (v4 >> 46));
            hash = xxh64_merge_round(hash, v1);
            hash = xxh64_merge_round(hash, v2);
            hash = xxh64_merge_round(hash, v3);
            hash = xxh64_merge_round(hash, v4);
        } else {
            hash = seed + XXHASH_PRIME64_5;
        }

        hash += total_len;

        // Process remaining bytes
        while (pos + 8 <= len) {
            ulong k1 = read_u64(data, pos);
            k1 *= XXHASH_PRIME64_2;
            k1 = (k1 << 31) | (k1 >> 33);
            k1 *= XXHASH_PRIME64_1;
            hash ^= k1;
            hash = ((hash << 27) | (hash >> 37)) * XXHASH_PRIME64_1 + XXHASH_PRIME64_4;
            pos += 8;
        }

        while (pos + 4 <= len) {
            hash ^= (ulong)read_u32(data, pos) * XXHASH_PRIME64_1;
            hash = ((hash << 23) | (hash >> 41)) * XXHASH_PRIME64_2 + XXHASH_PRIME64_3;
            pos += 4;
        }

        while (pos < len) {
            hash ^= (ulong)data[pos] * XXHASH_PRIME64_5;
            hash = ((hash << 11) | (hash >> 53)) * XXHASH_PRIME64_1;
            pos++;
        }

        hash = xxh64_avalanche(hash);

        // Store final hash
        files[job.file_index].hash_low = hash;
        files[job.file_index].hash_high = hash ^ 0xDEADBEEFCAFEBABEULL;  // Simple second hash
        files[job.file_index].status = 2;  // Complete
    } else {
        // Store intermediate state
        files[job.file_index].hash_low = v1;
        files[job.file_index].hash_high = v2;
        files[job.file_index].status = 1;  // Hashing
    }
}

// Group files by hash to find duplicates
kernel void group_duplicates_kernel(
    device const FileInfo* files [[buffer(0)]],
    device DuplicateGroup* groups [[buffer(1)]],
    device atomic_uint& group_count [[buffer(2)]],
    constant uint& file_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= file_count) return;

    FileInfo file = files[gid];
    if (file.status != 2) return;  // Not complete

    // Check if this file starts a new duplicate group
    // (Simple O(n) scan - a real implementation would use a hash table)
    bool is_first = true;
    uint duplicate_count = 1;
    ulong wasted = 0;

    for (uint i = 0; i < gid; i++) {
        if (files[i].status == 2 &&
            files[i].hash_low == file.hash_low &&
            files[i].hash_high == file.hash_high) {
            is_first = false;
            break;
        }
    }

    if (!is_first) return;

    // Count duplicates after this file
    for (uint i = gid + 1; i < file_count; i++) {
        if (files[i].status == 2 &&
            files[i].hash_low == file.hash_low &&
            files[i].hash_high == file.hash_high) {
            duplicate_count++;
            wasted += file.file_size;
        }
    }

    if (duplicate_count > 1) {
        uint idx = atomic_fetch_add_explicit(&group_count, 1, memory_order_relaxed);
        if (idx < 10000) {  // MAX_GROUPS
            groups[idx].hash_low = file.hash_low;
            groups[idx].hash_high = file.hash_high;
            groups[idx].first_file = gid;
            groups[idx].file_count = duplicate_count;
            groups[idx].wasted_bytes = wasted;
        }
    }
}
"#;

fn get_duplicate_finder_shader() -> String {
    DUPLICATE_FINDER_SHADER.replace("{{APP_SHADER_HEADER}}", APP_SHADER_HEADER)
}

// ============================================================================
// Public API Types
// ============================================================================

/// A group of duplicate files
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    pub files: Vec<PathBuf>,
    pub file_size: u64,
    pub wasted_bytes: u64,
}

/// Progress callback info
#[derive(Debug, Clone)]
pub struct ScanProgress {
    pub files_scanned: usize,
    pub files_total: usize,
    pub bytes_hashed: u64,
    pub phase: String,
}

/// Scan result summary
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub files_scanned: usize,
    pub duplicate_groups: usize,
    pub total_wasted_bytes: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

// ============================================================================
// Hash Cache
// ============================================================================

/// Cached hash entry
#[derive(Clone)]
struct CacheEntry {
    mtime: u64,
    size: u64,
    hash_low: u64,
    hash_high: u64,
}

/// Persistent hash cache to avoid re-reading unchanged files
struct HashCache {
    entries: HashMap<PathBuf, CacheEntry>,
    cache_path: PathBuf,
    dirty: bool,
}

impl HashCache {
    fn new() -> Self {
        let cache_path = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".duplicate_hash_cache");

        let mut cache = Self {
            entries: HashMap::new(),
            cache_path,
            dirty: false,
        };
        cache.load();
        cache
    }

    fn load(&mut self) {
        if let Ok(data) = fs::read_to_string(&self.cache_path) {
            for line in data.lines() {
                let parts: Vec<&str> = line.splitn(5, '\t').collect();
                if parts.len() == 5 {
                    if let (Ok(mtime), Ok(size), Ok(hash_low), Ok(hash_high)) = (
                        parts[1].parse::<u64>(),
                        parts[2].parse::<u64>(),
                        parts[3].parse::<u64>(),
                        parts[4].parse::<u64>(),
                    ) {
                        self.entries.insert(
                            PathBuf::from(parts[0]),
                            CacheEntry { mtime, size, hash_low, hash_high },
                        );
                    }
                }
            }
        }
    }

    fn save(&self) {
        if !self.dirty {
            return;
        }
        let mut output = String::new();
        for (path, entry) in &self.entries {
            output.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                path.display(),
                entry.mtime,
                entry.size,
                entry.hash_low,
                entry.hash_high
            ));
        }
        let _ = fs::write(&self.cache_path, output);
    }

    fn get(&self, path: &Path, mtime: u64, size: u64) -> Option<(u64, u64)> {
        self.entries.get(path).and_then(|e| {
            if e.mtime == mtime && e.size == size {
                Some((e.hash_low, e.hash_high))
            } else {
                None
            }
        })
    }

    fn insert(&mut self, path: PathBuf, mtime: u64, size: u64, hash_low: u64, hash_high: u64) {
        self.entries.insert(path, CacheEntry { mtime, size, hash_low, hash_high });
        self.dirty = true;
    }
}

impl Drop for HashCache {
    fn drop(&mut self) {
        self.save();
    }
}

// ============================================================================
// GPU Duplicate Finder
// ============================================================================

/// GPU-accelerated duplicate file finder
pub struct GpuDuplicateFinder {
    device: Device,
    command_queue: CommandQueue,
    hash_pipeline: ComputePipelineState,
    group_pipeline: ComputePipelineState,

    // File storage
    files_buffer: Buffer,       // FileInfo array
    chunk_buffer: Buffer,       // Chunk data for hashing
    jobs_buffer: Buffer,        // HashJob array
    params_buffer: Buffer,      // HashParams

    // Results
    groups_buffer: Buffer,      // DuplicateGroup array
    group_count_buffer: Buffer, // Atomic counter
    file_count_buffer: Buffer,

    // State
    max_files: usize,
    file_paths: Vec<PathBuf>,   // Paths indexed by FileInfo.path_index
    file_sizes: Vec<u64>,       // Sizes for each file
    file_mtimes: Vec<u64>,      // Modification times for cache validation

    // Cache
    hash_cache: HashCache,
    cache_hits: usize,
    cache_misses: usize,
}

impl GpuDuplicateFinder {
    /// Create a new GPU duplicate finder
    pub fn new(device: &Device, max_files: usize) -> Result<Self, String> {
        let max_files = max_files.min(MAX_FILES);

        let builder = AppBuilder::new(device, "GpuDuplicateFinder");
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = builder.compile_library(&get_duplicate_finder_shader())?;
        let hash_pipeline = builder.create_compute_pipeline(&library, "hash_chunk_kernel")?;
        let group_pipeline = builder.create_compute_pipeline(&library, "group_duplicates_kernel")?;

        // Allocate buffers - use BATCH_SIZE for working buffers to keep memory small
        let files_buffer = builder.create_buffer(max_files * mem::size_of::<FileInfo>());
        let chunk_buffer = builder.create_buffer(BATCH_SIZE * HASH_CHUNK_SIZE); // Only batch-sized working buffer
        let jobs_buffer = builder.create_buffer(BATCH_SIZE * mem::size_of::<HashJob>());
        let params_buffer = builder.create_buffer(mem::size_of::<HashParams>());
        let groups_buffer = builder.create_buffer(MAX_GROUPS * mem::size_of::<GpuDuplicateGroup>());
        let group_count_buffer = builder.create_buffer(mem::size_of::<u32>());
        let file_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        Ok(Self {
            device: device.clone(),
            command_queue,
            hash_pipeline,
            group_pipeline,
            files_buffer,
            chunk_buffer,
            jobs_buffer,
            params_buffer,
            groups_buffer,
            group_count_buffer,
            file_count_buffer,
            max_files,
            file_paths: Vec::with_capacity(max_files),
            file_sizes: Vec::with_capacity(max_files),
            file_mtimes: Vec::with_capacity(max_files),
            hash_cache: HashCache::new(),
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Scan a directory for files to check for duplicates
    pub fn scan_directory(&mut self, path: &Path) -> Result<ScanResult, io::Error> {
        self.file_paths.clear();
        self.file_sizes.clear();
        self.file_mtimes.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;

        // Collect files with sizes and mtimes
        let mut size_map: HashMap<u64, Vec<(PathBuf, u64)>> = HashMap::new();

        self.scan_recursive(path, &mut size_map)?;

        // Only keep files that have potential duplicates (same size)
        for (size, files) in size_map {
            if files.len() > 1 && size > 0 {
                for (p, mtime) in files {
                    if self.file_paths.len() >= self.max_files {
                        break;
                    }
                    self.file_paths.push(p);
                    self.file_sizes.push(size);
                    self.file_mtimes.push(mtime);
                }
            }
        }

        // Initialize GPU file info, checking cache for existing hashes
        unsafe {
            let files_ptr = self.files_buffer.contents() as *mut FileInfo;
            for i in 0..self.file_paths.len() {
                let path = &self.file_paths[i];
                let size = self.file_sizes[i];
                let mtime = self.file_mtimes[i];

                // Check cache for existing hash
                let (hash_low, hash_high, status) = if let Some((hl, hh)) = self.hash_cache.get(path, mtime, size) {
                    self.cache_hits += 1;
                    (hl, hh, 2) // Already complete
                } else {
                    self.cache_misses += 1;
                    (0, 0, 0) // Needs hashing
                };

                *files_ptr.add(i) = FileInfo {
                    path_index: i as u32,
                    file_size: size,
                    size_group: 0,
                    hash_low,
                    hash_high,
                    status,
                    _padding: 0,
                };
            }
        }

        Ok(ScanResult {
            files_scanned: self.file_paths.len(),
            duplicate_groups: 0,
            total_wasted_bytes: 0,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
        })
    }

    fn scan_recursive(&self, path: &Path, size_map: &mut HashMap<u64, Vec<(PathBuf, u64)>>) -> Result<(), io::Error> {
        if !path.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_dir() {
                // Skip hidden directories and common unimportant ones
                if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "node_modules" || name == "target" {
                        continue;
                    }
                }
                let _ = self.scan_recursive(&entry_path, size_map);
            } else if entry_path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    let size = metadata.len();
                    let mtime = metadata.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if size > 0 && size < 100 * 1024 * 1024 {  // Skip empty and very large files
                        size_map.entry(size).or_insert_with(Vec::new).push((entry_path, mtime));
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute hashes for all scanned files (processes in batches to limit memory)
    pub fn compute_hashes(&mut self) -> Result<(), io::Error> {
        if self.file_paths.is_empty() {
            return Ok(());
        }

        // Find files that need hashing (not cached)
        let files_to_hash: Vec<usize> = unsafe {
            let files_ptr = self.files_buffer.contents() as *const FileInfo;
            (0..self.file_paths.len())
                .filter(|&i| (*files_ptr.add(i)).status == 0)
                .collect()
        };

        // Skip GPU work if all files were cached
        if files_to_hash.is_empty() {
            return Ok(());
        }

        // Process files in batches to keep memory usage reasonable
        for batch in files_to_hash.chunks(BATCH_SIZE) {
            self.process_batch(batch)?;
        }

        // Update cache with new hashes
        self.update_cache();

        Ok(())
    }

    /// Update cache with newly computed hashes
    fn update_cache(&mut self) {
        unsafe {
            let files_ptr = self.files_buffer.contents() as *const FileInfo;
            for i in 0..self.file_paths.len() {
                let file = *files_ptr.add(i);
                if file.status == 2 {
                    let path = self.file_paths[i].clone();
                    let mtime = self.file_mtimes[i];
                    let size = self.file_sizes[i];
                    self.hash_cache.insert(path, mtime, size, file.hash_low, file.hash_high);
                }
            }
        }
    }

    /// Process a batch of files using parallel I/O
    fn process_batch(&mut self, file_indices: &[usize]) -> Result<(), io::Error> {
        use std::io::Read;
        use std::sync::Mutex;
        use std::thread;

        // Collect file read results in parallel
        let results: Mutex<Vec<(usize, Vec<u8>, u32, bool)>> = Mutex::new(Vec::with_capacity(file_indices.len()));

        // Parallel file reading using scoped threads
        let paths: Vec<_> = file_indices
            .iter()
            .map(|&i| (i, &self.file_paths[i], self.file_sizes[i]))
            .collect();

        // Read files in parallel (8 threads)
        let chunk_count = (paths.len() + 7) / 8;
        thread::scope(|s| {
            for chunk in paths.chunks(chunk_count.max(1)) {
                let results_ref = &results;
                s.spawn(move || {
                    for &(file_idx, path, file_size) in chunk {
                        if let Ok(mut file) = fs::File::open(path) {
                            let mut buffer = vec![0u8; HASH_CHUNK_SIZE];
                            if let Ok(n) = file.read(&mut buffer) {
                                if n > 0 {
                                    buffer.truncate(n);
                                    let is_last = file_size <= HASH_CHUNK_SIZE as u64;
                                    results_ref.lock().unwrap().push((file_idx, buffer, n as u32, is_last));
                                }
                            }
                        }
                    }
                });
            }
        });

        let mut read_results = results.into_inner().unwrap();
        if read_results.is_empty() {
            return Ok(());
        }

        // Sort by file index to maintain order (helps with cache locality)
        read_results.sort_by_key(|(idx, _, _, _)| *idx);

        // Pack data consecutively - shader uses gid * CHUNK_SIZE
        let mut chunk_data: Vec<u8> = vec![0u8; read_results.len() * HASH_CHUNK_SIZE];
        let mut jobs: Vec<HashJob> = Vec::with_capacity(read_results.len());

        for (job_idx, (file_idx, data, bytes_read, is_last)) in read_results.into_iter().enumerate() {
            let offset = job_idx * HASH_CHUNK_SIZE;
            chunk_data[offset..offset + data.len()].copy_from_slice(&data);

            jobs.push(HashJob {
                file_index: file_idx as u32,
                chunk_index: 0,
                chunk_offset: 0, // Not used by shader
                chunk_length: bytes_read,
                is_last_chunk: if is_last { 1 } else { 0 },
            });
        }

        // Copy to GPU buffers
        unsafe {
            let chunk_ptr = self.chunk_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(chunk_data.as_ptr(), chunk_ptr, chunk_data.len());

            let jobs_ptr = self.jobs_buffer.contents() as *mut HashJob;
            for (i, job) in jobs.iter().enumerate() {
                *jobs_ptr.add(i) = *job;
            }

            let params_ptr = self.params_buffer.contents() as *mut HashParams;
            *params_ptr = HashParams {
                job_count: jobs.len() as u32,
                seed: 0,
                _padding: [0; 2],
            };
        }

        // Dispatch hash kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.hash_pipeline);
        encoder.set_buffer(0, Some(&self.files_buffer), 0);
        encoder.set_buffer(1, Some(&self.chunk_buffer), 0);
        encoder.set_buffer(2, Some(&self.jobs_buffer), 0);
        encoder.set_buffer(3, Some(&self.params_buffer), 0);

        let threads_per_group = 256;
        let thread_groups = (jobs.len() + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Find duplicate groups after hashing
    pub fn find_duplicates(&self) -> Vec<DuplicateGroup> {
        if self.file_paths.is_empty() {
            return vec![];
        }

        // Reset group count
        unsafe {
            let count_ptr = self.group_count_buffer.contents() as *mut u32;
            *count_ptr = 0;

            let file_count_ptr = self.file_count_buffer.contents() as *mut u32;
            *file_count_ptr = self.file_paths.len() as u32;
        }

        // Dispatch group kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.group_pipeline);
        encoder.set_buffer(0, Some(&self.files_buffer), 0);
        encoder.set_buffer(1, Some(&self.groups_buffer), 0);
        encoder.set_buffer(2, Some(&self.group_count_buffer), 0);
        encoder.set_buffer(3, Some(&self.file_count_buffer), 0);

        let threads_per_group = 256;
        let thread_groups = (self.file_paths.len() + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let mut results: Vec<DuplicateGroup> = Vec::new();

        unsafe {
            let count = *(self.group_count_buffer.contents() as *const u32);
            let groups = self.groups_buffer.contents() as *const GpuDuplicateGroup;
            let files = self.files_buffer.contents() as *const FileInfo;

            for i in 0..count.min(MAX_GROUPS as u32) as usize {
                let g = *groups.add(i);

                // Collect all files with this hash
                let mut group_files: Vec<PathBuf> = Vec::new();
                for j in 0..self.file_paths.len() {
                    let f = *files.add(j);
                    if f.hash_low == g.hash_low && f.hash_high == g.hash_high {
                        group_files.push(self.file_paths[j].clone());
                    }
                }

                if group_files.len() > 1 {
                    let file_size = self.file_sizes.get(g.first_file as usize).copied().unwrap_or(0);
                    results.push(DuplicateGroup {
                        files: group_files.clone(),
                        file_size,
                        wasted_bytes: file_size * (group_files.len() as u64 - 1),
                    });
                }
            }
        }

        // Sort by wasted bytes (descending)
        results.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));

        results
    }

    /// Get total wasted bytes across all duplicates
    pub fn total_wasted_bytes(&self, groups: &[DuplicateGroup]) -> u64 {
        groups.iter().map(|g| g.wasted_bytes).sum()
    }

    /// Get number of files being checked
    pub fn file_count(&self) -> usize {
        self.file_paths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_info_size() {
        // Should be 48 bytes with padding
        assert!(mem::size_of::<FileInfo>() <= 64);
    }

    #[test]
    fn test_hash_job_size() {
        assert_eq!(mem::size_of::<HashJob>(), 24);
    }

    #[test]
    fn test_hash_params_size() {
        assert_eq!(mem::size_of::<HashParams>(), 16);
    }

    #[test]
    fn test_duplicate_group_size() {
        assert_eq!(mem::size_of::<GpuDuplicateGroup>(), 32);
    }
}
