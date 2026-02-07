// GPU-Native Filesystem
//
// Issue #19: Core Filesystem Data Structures
// Implements filesystem as a GpuApp that performs all operations on GPU
//
// Key reuse from existing framework:
// - Same parent/child/sibling pattern as WidgetCompact
// - Uses AppBuilder for shader compilation
// - Leverages APP_SHADER_HEADER
// - Implements GpuApp trait for integration with GpuRuntime

use super::app::{AppBuilder, GpuApp, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::content_search::{GpuContentSearch, ContentMatch, SearchOptions};
use super::batch_io::GpuBatchLoader;
use metal::*;
use std::cell::Cell;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use dispatch::{Queue, QueueAttribute};

// ============================================================================
// Constants (matching existing framework)
// ============================================================================

/// Block size (4KB - standard for filesystems)
pub const BLOCK_SIZE: usize = 4096;

/// Root inode ID (like ROOT widget)
pub const ROOT_INODE_ID: u32 = 0;

/// Invalid inode (like INVALID widget)
pub const INVALID_INODE: u32 = 0xFFFFFFFF;

/// Maximum path depth
pub const MAX_PATH_DEPTH: usize = 16;

/// Maximum filename length
pub const MAX_FILENAME_LEN: usize = 255;

// ============================================================================
// Data Structures (following WidgetCompact pattern)
// ============================================================================

/// Compact inode structure - 64 bytes (cache-line aligned)
/// Mirrors WidgetCompact's parent/child/sibling pattern
/// Fields ordered for optimal packing: u64s first, then u32s, then u16s
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)] // Some fields accessed via unsafe pointers
pub struct InodeCompact {
    // u64 fields first (24 bytes)
    pub inode_id: u64,          // Unique inode number
    pub size: u64,              // File size in bytes
    pub timestamps: u64,        // Packed: created|modified|accessed

    // u32 fields (32 bytes = 8 fields)
    pub parent_id: u32,         // Parent directory (like widget.parent_id)
    pub first_child: u32,       // First child inode (like widget.first_child)
    pub next_sibling: u32,      // Next sibling (like widget.next_sibling)
    pub prev_sibling: u32,      // Previous sibling
    pub blocks: u32,            // Number of 4KB blocks allocated
    pub block_ptr: u32,         // Offset in block map table
    pub uid_gid: u32,           // Packed: uid(16) | gid(16)
    pub checksum: u32,          // CRC32 of file content

    // u16 fields (8 bytes)
    pub mode: u16,              // Permissions (rwxrwxrwx)
    pub flags: u16,             // Type, compression, encryption flags
    pub refcount: u16,          // Hard link count
    pub _padding: u16,          // Alignment to 64 bytes
}
// Total: 24 + 32 + 8 = 64 bytes exactly

/// Directory entry - 32 bytes (like DirEntryCompact, half cache line)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DirEntryCompact {
    pub inode_id: u32,          // Target inode
    pub name_hash: u32,         // xxHash3 of filename
    pub name_len: u16,          // Filename length
    pub file_type: u8,          // Cached from inode
    pub _padding: u8,
    pub name: [u8; 20],         // Short filename (inline for <20 chars)
}
// Total: 4 + 4 + 2 + 1 + 1 + 20 = 32 bytes

/// Block map entry - 8 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlockMapEntry {
    pub physical_block: u32,    // Physical block number on disk
    pub flags: u16,             // Sparse, compressed, encrypted, CoW
    pub refcount: u16,          // For deduplication/CoW
}

/// Hash table entry for O(1) directory lookups (Issue #129)
/// Key: (parent_inode, name_hash)
/// Value: entry_index into entries array
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DirHashEntry {
    pub parent_inode: u32,  // Part of key
    pub name_hash: u32,     // Part of key
    pub entry_index: u32,   // Value: index into entries array (0xFFFFFFFF = empty)
    pub _padding: u32,      // Alignment to 16 bytes
}

impl DirHashEntry {
    /// Create an empty hash table entry
    pub fn empty() -> Self {
        Self {
            parent_inode: 0,
            name_hash: 0,
            entry_index: 0xFFFFFFFF,  // Sentinel for empty
            _padding: 0,
        }
    }

    /// Check if this entry is empty
    pub fn is_empty(&self) -> bool {
        self.entry_index == 0xFFFFFFFF
    }
}

/// File types
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FileType {
    Regular = 0,
    Directory = 1,
    Symlink = 2,
    BlockDevice = 3,
    CharDevice = 4,
    Fifo = 5,
    Socket = 6,
    Unknown = 7,
}

impl FileType {
    pub fn from_u8(value: u8) -> Self {
        match value & 0x0F {
            0 => Self::Regular,
            1 => Self::Directory,
            2 => Self::Symlink,
            3 => Self::BlockDevice,
            4 => Self::CharDevice,
            5 => Self::Fifo,
            6 => Self::Socket,
            _ => Self::Unknown,
        }
    }
}

// ============================================================================
// Helper implementations (like WidgetCompact helpers)
// ============================================================================

impl Default for InodeCompact {
    fn default() -> Self {
        Self {
            inode_id: 0,
            size: 0,
            timestamps: 0,
            parent_id: 0,
            first_child: INVALID_INODE,
            next_sibling: INVALID_INODE,
            prev_sibling: INVALID_INODE,
            blocks: 0,
            block_ptr: 0,
            uid_gid: 0,
            checksum: 0,
            mode: 0o644,
            flags: 0,
            refcount: 1,
            _padding: 0,
        }
    }
}

impl InodeCompact {
    pub fn new(inode_id: u64, parent_id: u32, file_type: FileType) -> Self {
        let mut inode = Self::default();
        inode.inode_id = inode_id;
        inode.parent_id = parent_id;
        inode.set_file_type(file_type);
        inode
    }

    // Flag manipulation (like WidgetCompact flags)
    pub fn set_file_type(&mut self, file_type: FileType) {
        self.flags = (self.flags & 0xFFF0) | (file_type as u16 & 0x0F);
    }

    pub fn get_file_type(&self) -> FileType {
        FileType::from_u8((self.flags & 0x0F) as u8)
    }

    pub fn set_compressed(&mut self, enabled: bool) {
        if enabled {
            self.flags |= 0x0010;
        } else {
            self.flags &= !0x0010;
        }
    }

    pub fn is_compressed(&self) -> bool {
        (self.flags & 0x0010) != 0
    }

    pub fn is_directory(&self) -> bool {
        self.get_file_type() == FileType::Directory
    }
}

impl Default for DirEntryCompact {
    fn default() -> Self {
        Self {
            inode_id: 0,
            name_hash: 0,
            name_len: 0,
            file_type: 0,
            _padding: 0,
            name: [0; 20],
        }
    }
}

impl DirEntryCompact {
    pub fn new(inode_id: u32, name: &str) -> Self {
        let mut entry = Self::default();
        entry.inode_id = inode_id;
        entry.name_len = name.len().min(20) as u16;
        entry.name_hash = xxhash3(name.as_bytes());
        entry.name[..entry.name_len as usize].copy_from_slice(&name.as_bytes()[..entry.name_len as usize]);
        entry
    }
}

// Simple xxHash3 (placeholder - use real implementation in production)
fn xxhash3(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x9E3779B1;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x85EBCA77);
    }
    hash ^ (hash >> 16)
}

// ============================================================================
// Metal Shader (uses APP_SHADER_HEADER from framework)
// ============================================================================

const FILESYSTEM_SHADER: &str = r#"
{{APP_SHADER_HEADER}}

// ============================================================================
// Filesystem Structures (must match Rust)
// ============================================================================

struct InodeCompact {
    // u64 fields first (24 bytes)
    uint64_t inode_id;
    uint64_t size;
    uint64_t timestamps;
    // u32 fields (32 bytes = 8 fields)
    uint32_t parent_id;
    uint32_t first_child;
    uint32_t next_sibling;
    uint32_t prev_sibling;
    uint32_t blocks;
    uint32_t block_ptr;
    uint32_t uid_gid;
    uint32_t checksum;
    // u16 fields (8 bytes)
    uint16_t mode;
    uint16_t flags;
    uint16_t refcount;
    uint16_t _padding;
};
// Total: 24 + 32 + 8 = 64 bytes

struct DirEntryCompact {
    uint32_t inode_id;
    uint32_t name_hash;
    uint16_t name_len;
    uint8_t file_type;
    uint8_t _padding;
    char name[20];
};
// Total: 32 bytes

struct FsParams {
    uint32_t current_directory;
    uint32_t total_inodes;
    uint32_t total_entries;
    uint32_t selected_file;
};

struct Vertex {
    float4 position [[position]];
    float4 color;
};

// Path lookup structures
struct PathComponent {
    uint32_t hash;
    char name[20];
    uint16_t len;
    uint16_t _padding;
};

struct PathLookupParams {
    uint32_t start_inode;     // Starting inode (usually ROOT or current dir)
    uint32_t component_count; // Number of path components
    uint32_t total_entries;   // Total directory entries in filesystem
    uint32_t _padding;
};

// Simple hash function (matches Rust xxhash3)
uint32_t xxhash3(constant char* data, uint16_t len) {
    uint32_t hash = 0x9E3779B1;
    for (uint16_t i = 0; i < len; i++) {
        hash ^= (uint32_t)data[i];
        hash *= 0x85EBCA77;
    }
    return hash ^ (hash >> 16);
}

// ============================================================================
// Directory Hash Table (Issue #129) - O(1) Lookup
// ============================================================================

// Hash table entry for directory lookups
struct DirHashEntry {
    uint32_t parent_inode;  // Part of key
    uint32_t name_hash;     // Part of key
    uint32_t entry_index;   // Value: index into entries array (0xFFFFFFFF = empty)
    uint32_t _padding;      // Alignment to 16 bytes
};

// Combined key hash using Fibonacci/golden ratio hashing
inline uint64_t dir_hash_key(uint32_t parent_inode, uint32_t name_hash) {
    uint64_t key = ((uint64_t)parent_inode << 32) | (uint64_t)name_hash;
    return key * 0x9E3779B97F4A7C15ULL;  // Golden ratio hash
}

// Compute hash slot from key
inline uint32_t hash_slot(uint64_t key, uint32_t mask) {
    return (uint32_t)(key >> 32) & mask;  // Use high bits for slot
}

// O(1) single directory entry lookup using hash table
kernel void dir_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    constant uint32_t& parent_inode [[buffer(2)]],
    constant uint32_t& name_hash [[buffer(3)]],
    device uint32_t* result_entry_idx [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single-threaded lookup

    uint64_t key = dir_hash_key(parent_inode, name_hash);
    uint32_t slot = hash_slot(key, table_mask);

    // Linear probing with max 32 attempts
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t idx = (slot + probe) & table_mask;
        DirHashEntry entry = hash_table[idx];

        if (entry.entry_index == 0xFFFFFFFF) {
            // Empty slot - not found
            *result_entry_idx = 0xFFFFFFFF;
            return;
        }

        if (entry.parent_inode == parent_inode && entry.name_hash == name_hash) {
            *result_entry_idx = entry.entry_index;
            return;
        }
    }

    // Max probes exceeded - not found
    *result_entry_idx = 0xFFFFFFFF;
}

// O(1) path resolution using hash table
// Resolves full path by walking components and doing O(1) lookup at each step
kernel void path_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    device const DirEntryCompact* entries [[buffer(2)]],
    constant PathComponent* components [[buffer(3)]],
    constant uint32_t& component_count [[buffer(4)]],
    constant uint32_t& start_inode [[buffer(5)]],
    device uint32_t* result_inode [[buffer(6)]],
    device uint32_t* result_status [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single-threaded for sequential path walk

    const uint32_t STATUS_SUCCESS = 0;
    const uint32_t STATUS_NOT_FOUND = 1;

    uint32_t current = start_inode;

    for (uint32_t i = 0; i < component_count; i++) {
        uint64_t key = dir_hash_key(current, components[i].hash);
        uint32_t slot = hash_slot(key, table_mask);

        bool found = false;
        for (uint32_t probe = 0; probe < 32; probe++) {
            uint32_t idx = (slot + probe) & table_mask;
            DirHashEntry entry = hash_table[idx];

            if (entry.entry_index == 0xFFFFFFFF) {
                // Empty slot - path component not found
                break;
            }

            if (entry.parent_inode == current && entry.name_hash == components[i].hash) {
                // Found! Verify name matches (handle hash collisions)
                DirEntryCompact dir_entry = entries[entry.entry_index];

                bool name_matches = true;
                if (dir_entry.name_len == components[i].len) {
                    for (uint16_t j = 0; j < components[i].len; j++) {
                        if (dir_entry.name[j] != components[i].name[j]) {
                            name_matches = false;
                            break;
                        }
                    }
                } else {
                    name_matches = false;
                }

                if (name_matches) {
                    current = dir_entry.inode_id;  // Move to this entry's inode
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            *result_inode = 0xFFFFFFFF;
            *result_status = STATUS_NOT_FOUND;
            return;
        }
    }

    *result_inode = current;
    *result_status = STATUS_SUCCESS;
}

// Batch O(1) directory lookup - one thread per lookup request
kernel void batch_dir_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    device const DirEntryCompact* entries [[buffer(2)]],
    device const uint32_t* parent_inodes [[buffer(3)]],
    device const uint32_t* name_hashes [[buffer(4)]],
    device uint32_t* result_entry_indices [[buffer(5)]],
    constant uint32_t& lookup_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= lookup_count) return;

    uint32_t parent = parent_inodes[gid];
    uint32_t name_hash = name_hashes[gid];

    uint64_t key = dir_hash_key(parent, name_hash);
    uint32_t slot = hash_slot(key, table_mask);

    // Each thread probes independently - no divergence!
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t idx = (slot + probe) & table_mask;
        DirHashEntry entry = hash_table[idx];

        if (entry.entry_index == 0xFFFFFFFF) {
            // Empty slot - not found
            result_entry_indices[gid] = 0xFFFFFFFF;
            return;
        }

        if (entry.parent_inode == parent && entry.name_hash == name_hash) {
            result_entry_indices[gid] = entry.entry_index;
            return;
        }
    }

    result_entry_indices[gid] = 0xFFFFFFFF;  // Not found
}

// ============================================================================
// Path Lookup Kernel (Issue #21)
// Resolves "/foo/bar/file.txt" to inode ID using parallel hash lookup
// ============================================================================

kernel void path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],           // All inodes
    device DirEntryCompact* entries [[buffer(1)]],       // All directory entries
    constant PathLookupParams& params [[buffer(2)]],     // Lookup params
    constant PathComponent* components [[buffer(3)]],    // Path components to resolve
    device atomic_uint& result_inode [[buffer(4)]],      // Output: found inode ID
    device atomic_uint& result_status [[buffer(5)]],     // Output: 0=success, 1=not_found
    uint tid [[thread_index_in_threadgroup]]
)
{
    const uint32_t INVALID_INODE = 0xFFFFFFFF;
    const uint32_t STATUS_SUCCESS = 0;
    const uint32_t STATUS_NOT_FOUND = 1;

    // Initialize result (thread 0 only)
    if (tid == 0) {
        atomic_store_explicit(&result_inode, INVALID_INODE, memory_order_relaxed);
        atomic_store_explicit(&result_status, STATUS_NOT_FOUND, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Walk path components sequentially (can't parallelize tree traversal)
    // But we can parallelize the search within each directory
    uint32_t current_inode = params.start_inode;

    for (uint32_t comp_idx = 0; comp_idx < params.component_count; comp_idx++) {
        PathComponent component = components[comp_idx];

        // Parallel search: find entry with matching parent_id and hash
        threadgroup uint32_t found_inode;
        threadgroup atomic_uint found_flag;

        if (tid == 0) {
            found_inode = INVALID_INODE;
            atomic_store_explicit(&found_flag, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread searches different entries
        for (uint32_t i = tid; i < params.total_entries; i += 1024) {
            uint32_t entry_inode = entries[i].inode_id;

            // Check if this entry is in the current directory
            if (inodes[entry_inode].parent_id == current_inode) {
                // Check if hash matches
                if (entries[i].name_hash == component.hash) {
                    // Verify name (in case of hash collision)
                    bool name_matches = true;
                    if (entries[i].name_len == component.len) {
                        for (uint16_t j = 0; j < component.len; j++) {
                            if (entries[i].name[j] != component.name[j]) {
                                name_matches = false;
                                break;
                            }
                        }
                    } else {
                        name_matches = false;
                    }

                    if (name_matches) {
                        // Found it! (First thread to find wins)
                        uint32_t old_flag = atomic_exchange_explicit(&found_flag, 1, memory_order_relaxed);
                        if (old_flag == 0) {
                            found_inode = entry_inode;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if we found the component
        if (found_inode == INVALID_INODE) {
            // Path component not found - abort
            if (tid == 0) {
                atomic_store_explicit(&result_status, STATUS_NOT_FOUND, memory_order_relaxed);
            }
            return;
        }

        // Move to next level
        current_inode = found_inode;
    }

    // Success - we resolved all components
    if (tid == 0) {
        atomic_store_explicit(&result_inode, current_inode, memory_order_relaxed);
        atomic_store_explicit(&result_status, STATUS_SUCCESS, memory_order_relaxed);
    }
}

// ============================================================================
// Batch Path Lookup Kernel (Issue #26)
// Multiple paths processed in parallel, one threadgroup per path
// ============================================================================

struct PathMetadata {
    uint32_t start_idx;      // Index into components buffer
    uint32_t component_count;// Number of components in this path
    uint32_t start_inode;    // Starting inode (ROOT or current_dir)
    uint32_t _padding;
};

struct BatchResult {
    uint32_t inode_id;  // Result inode (or INVALID_INODE)
    uint32_t status;    // 0=success, 1=not_found
};

// GPU Cache Entry (Issue #29)
struct PathCacheEntry {
    uint64_t path_hash;
    uint32_t inode_id;
    uint32_t access_count;
    uint64_t timestamp;
    uint32_t path_len;
    uint32_t _padding;
    char path[32];  // 32 bytes for 64-byte total
};

constant uint32_t CACHE_SIZE = 1024;
constant uint64_t CACHE_MASK = 1023;

// Compute hash of full path from components
uint64_t compute_full_path_hash(
    device PathComponent* components,
    uint32_t start_idx,
    uint32_t count
) {
    uint64_t hash = 0x9E3779B185EBCA87UL;

    for (uint32_t i = 0; i < count; i++) {
        PathComponent comp = components[start_idx + i];
        hash ^= comp.hash;
        hash *= 0x9E3779B185EBCA87UL;
    }

    return hash;
}

kernel void batch_path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* entries [[buffer(1)]],
    constant uint32_t& total_inodes [[buffer(2)]],
    constant uint32_t& total_entries [[buffer(3)]],
    device PathComponent* all_components [[buffer(4)]],
    device PathMetadata* path_metadata [[buffer(5)]],
    device BatchResult* results [[buffer(6)]],
    constant uint32_t& batch_size [[buffer(7)]],
    device PathCacheEntry* cache [[buffer(8)]],           // GPU cache
    device atomic_uint* cache_hits [[buffer(9)]],         // Hit counter
    device atomic_uint* cache_misses [[buffer(10)]],      // Miss counter
    constant uint32_t& frame_number [[buffer(11)]],       // Frame counter
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    const uint32_t INVALID_INODE = 0xFFFFFFFF;
    const uint32_t STATUS_SUCCESS = 0;
    const uint32_t STATUS_NOT_FOUND = 1;

    // Calculate which path this threadgroup is processing
    uint path_idx = gid / 1024;

    if (path_idx >= batch_size) return;

    // Load this path's metadata
    PathMetadata meta = path_metadata[path_idx];

    // === GPU CACHE CHECK (Issue #29) ===
    threadgroup bool cache_hit_flag;
    threadgroup uint32_t cached_inode;

    if (tid == 0) {
        cache_hit_flag = false;

        // Compute full path hash
        uint64_t full_path_hash = compute_full_path_hash(
            all_components, meta.start_idx, meta.component_count
        );

        // Probe cache (direct-mapped)
        uint32_t cache_slot = (uint32_t)(full_path_hash & CACHE_MASK);
        PathCacheEntry entry = cache[cache_slot];

        // Check if entry matches our path
        if (entry.path_hash == full_path_hash) {
            cache_hit_flag = true;
            cached_inode = entry.inode_id;

            // Update statistics
            atomic_fetch_add_explicit(cache_hits, 1u, memory_order_relaxed);
            cache[cache_slot].access_count++;
            cache[cache_slot].timestamp = frame_number;
        }

        if (!cache_hit_flag) {
            atomic_fetch_add_explicit(cache_misses, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === CACHE HIT - Fast return ===
    if (cache_hit_flag) {
        if (tid == 0) {
            results[path_idx].inode_id = cached_inode;
            results[path_idx].status = STATUS_SUCCESS;
        }
        return;  // Done! ~50ns
    }

    // === CACHE MISS - Full lookup ===
    uint32_t current_inode = meta.start_inode;

    // Walk through each component of the path
    for (uint32_t comp_idx = 0; comp_idx < meta.component_count; comp_idx++) {
        PathComponent component = all_components[meta.start_idx + comp_idx];

        // Shared variables for this threadgroup
        threadgroup uint32_t found_inode;
        threadgroup atomic_uint found_flag;

        if (tid == 0) {
            found_inode = INVALID_INODE;
            atomic_store_explicit(&found_flag, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel search: each thread checks different entries
        for (uint32_t i = tid; i < total_entries; i += 1024) {
            uint32_t entry_inode = entries[i].inode_id;

            // Check if this entry is in the current directory
            if (inodes[entry_inode].parent_id == current_inode) {
                // Check hash match
                if (entries[i].name_hash == component.hash) {
                    // Verify name (handle hash collisions)
                    bool name_matches = true;
                    if (entries[i].name_len == component.len) {
                        for (uint16_t j = 0; j < component.len; j++) {
                            if (entries[i].name[j] != component.name[j]) {
                                name_matches = false;
                                break;
                            }
                        }
                    } else {
                        name_matches = false;
                    }

                    if (name_matches) {
                        // Found it! First thread to find wins
                        uint32_t old = atomic_exchange_explicit(&found_flag, 1,
                            memory_order_relaxed);
                        if (old == 0) {
                            found_inode = entry_inode;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if component was found
        if (found_inode == INVALID_INODE) {
            // Path component not found - mark as error and exit
            if (tid == 0) {
                results[path_idx].inode_id = INVALID_INODE;
                results[path_idx].status = STATUS_NOT_FOUND;
            }
            return;
        }

        // Move to next level
        current_inode = found_inode;
    }

    // All components resolved successfully
    if (tid == 0) {
        results[path_idx].inode_id = current_inode;
        results[path_idx].status = STATUS_SUCCESS;

        // === UPDATE GPU CACHE (Issue #29) ===
        if (current_inode != INVALID_INODE) {
            // Compute full path hash
            uint64_t full_path_hash = compute_full_path_hash(
                all_components, meta.start_idx, meta.component_count
            );

            uint32_t cache_slot = (uint32_t)(full_path_hash & CACHE_MASK);

            // Store in cache
            cache[cache_slot].path_hash = full_path_hash;
            cache[cache_slot].inode_id = current_inode;
            cache[cache_slot].access_count = 1;
            cache[cache_slot].timestamp = frame_number;
            cache[cache_slot].path_len = 0; // TODO: store actual path
        }
    }
}

// ============================================================================
// Compute Kernel (lists current directory, renders as UI)
// ============================================================================

kernel void filesystem_compute_kernel(
    device FrameState& frame_state [[buffer(0)]],      // OS-provided
    device InputQueue& input_queue [[buffer(1)]],      // OS-provided
    constant FsParams& params [[buffer(2)]],           // App params
    device InodeCompact* inodes [[buffer(3)]],         // Filesystem inodes
    device DirEntryCompact* entries [[buffer(4)]],     // Filesystem entries
    device Vertex* vertices [[buffer(5)]],             // Output vertices
    device atomic_uint& vertex_count [[buffer(6)]],    // Output vertex count
    uint tid [[thread_index_in_threadgroup]]
)
{
    // Phase 1: Filter directory entries for current directory
    threadgroup uint32_t local_matches[1024];
    threadgroup atomic_uint match_count;

    if (tid == 0) {
        atomic_store_explicit(&match_count, 0, memory_order_relaxed);
        atomic_store_explicit(&vertex_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread checks different entries
    for (uint32_t i = tid; i < params.total_entries; i += 1024) {
        uint32_t entry_inode = entries[i].inode_id;
        if (inodes[entry_inode].parent_id == params.current_directory) {
            uint32_t idx = atomic_fetch_add_explicit(&match_count, 1, memory_order_relaxed);
            local_matches[idx] = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t count = atomic_load_explicit(&match_count, memory_order_relaxed);

    // Phase 2: Generate vertices for file list UI
    if (tid < count) {
        uint32_t entry_idx = local_matches[tid];
        DirEntryCompact entry = entries[entry_idx];

        // Position: Each file is a rectangle at y = tid * 0.05
        float y = -0.9 + float(tid) * 0.05;
        float height = 0.04;
        float width = 0.8;

        // Color: Selected file is highlighted
        float3 color = (entry_idx == params.selected_file)
            ? float3(0.3, 0.5, 0.8)  // Blue (selected)
            : float3(0.2, 0.2, 0.2); // Gray (normal)

        // Generate 6 vertices (2 triangles) for this file entry
        uint32_t base = atomic_fetch_add_explicit(&vertex_count, 6, memory_order_relaxed);

        Vertex v0; v0.position = float4(-width, y, 0, 1); v0.color = float4(color, 1);
        Vertex v1; v1.position = float4(width, y, 0, 1); v1.color = float4(color, 1);
        Vertex v2; v2.position = float4(-width, y + height, 0, 1); v2.color = float4(color, 1);
        Vertex v3; v3.position = float4(width, y, 0, 1); v3.color = float4(color, 1);
        Vertex v4; v4.position = float4(width, y + height, 0, 1); v4.color = float4(color, 1);
        Vertex v5; v5.position = float4(-width, y + height, 0, 1); v5.color = float4(color, 1);

        vertices[base + 0] = v0;
        vertices[base + 1] = v1;
        vertices[base + 2] = v2;
        vertices[base + 3] = v3;
        vertices[base + 4] = v4;
        vertices[base + 5] = v5;
    }
}

// ============================================================================
// Vertex Shader (pass-through)
// ============================================================================

vertex Vertex filesystem_vertex_shader(
    uint vertex_id [[vertex_id]],
    const device Vertex* vertices [[buffer(0)]]
) {
    return vertices[vertex_id];
}

// ============================================================================
// Fragment Shader (simple color)
// ============================================================================

fragment float4 filesystem_fragment_shader(Vertex in [[stage_in]]) {
    return in.color;
}

// ============================================================================
// GPU Fuzzy Search Kernel (for filesystem browser)
// ============================================================================

constant uint32_t MAX_PATH_LEN = 256;
constant uint32_t MAX_QUERY_WORDS = 8;
constant uint32_t MAX_WORD_LEN = 32;

struct SearchParams {
    uint32_t path_count;
    uint32_t word_count;
    uint32_t max_results;
    uint32_t _padding;
};

struct SearchWord {
    char chars[32];   // Word characters (lowercase)
    uint16_t len;
    uint16_t _padding;
};

// Tokenize result (for GPU tokenization)
struct TokenizeResult {
    atomic_uint word_count;
    uint32_t _padding[3];
};

// ============================================================================
// GPU Query Tokenizer (Issue #79 - Zero CPU String Processing)
// ============================================================================
//
// Tokenizes a raw query string entirely on GPU.
// Input: "Foo BAR  baz" (raw bytes from CPU - THE ONLY CPU WORK)
// Output: SearchWord[0] = "foo", SearchWord[1] = "bar", SearchWord[2] = "baz"
//
// Each thread processes one character position. Threads at word boundaries
// atomically claim a word slot and copy the word with lowercase conversion.

inline bool is_whitespace_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline char to_lowercase_char(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c + 32;
    }
    return c;
}

kernel void tokenize_query_kernel(
    device const char* query [[buffer(0)]],          // Raw query bytes
    device SearchWord* words [[buffer(1)]],          // Output: tokenized words
    device TokenizeResult* result [[buffer(2)]],     // Output: word count
    constant uint32_t& query_len [[buffer(3)]],      // Query length
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= query_len) return;

    char c = query[tid];
    char prev = (tid > 0) ? query[tid - 1] : ' ';

    bool c_is_space = is_whitespace_char(c);
    bool prev_is_space = is_whitespace_char(prev) || tid == 0;

    // Word starts where non-space follows space
    if (!c_is_space && prev_is_space) {
        // Claim a word slot atomically
        uint word_idx = atomic_fetch_add_explicit(&result->word_count, 1, memory_order_relaxed);

        if (word_idx >= MAX_QUERY_WORDS) {
            // Too many words - decrement and bail
            atomic_fetch_sub_explicit(&result->word_count, 1, memory_order_relaxed);
            return;
        }

        // Find word end and copy with lowercase conversion
        uint word_len = 0;
        for (uint i = tid; i < query_len && word_len < MAX_WORD_LEN - 1; i++) {
            char ch = query[i];
            if (is_whitespace_char(ch)) break;

            words[word_idx].chars[word_len++] = to_lowercase_char(ch);
        }

        // Null-terminate
        words[word_idx].chars[word_len] = 0;
        words[word_idx].len = word_len;
        words[word_idx]._padding = 0;
    }
}

// Compact search result - only matches written here
struct SearchResult {
    uint32_t path_index;
    int32_t score;
};

// TextChar for direct GPU text rendering
struct TextChar {
    float x;
    float y;
    uint32_t char_code;
    uint32_t color;
};

// Fuzzy match: all characters of needle appear in haystack in order
bool fuzzy_match_gpu(device const char* haystack, uint16_t haystack_len,
                     device const char* needle, uint16_t needle_len) {
    uint16_t h_idx = 0;

    for (uint16_t n_idx = 0; n_idx < needle_len; n_idx++) {
        char needle_char = needle[n_idx];
        bool found = false;

        while (h_idx < haystack_len) {
            char h_char = haystack[h_idx];
            h_idx++;

            // Convert to lowercase for comparison
            if (h_char >= 'A' && h_char <= 'Z') {
                h_char = h_char + 32;
            }

            if (h_char == needle_char) {
                found = true;
                break;
            }
        }

        if (!found) return false;
    }

    return true;
}

// Check if haystack contains needle as substring (case-insensitive)
bool contains_substring_gpu(device const char* haystack, uint16_t haystack_len,
                           device const char* needle, uint16_t needle_len) {
    if (needle_len > haystack_len) return false;

    for (uint16_t i = 0; i <= haystack_len - needle_len; i++) {
        bool match = true;
        for (uint16_t j = 0; j < needle_len; j++) {
            char h_char = haystack[i + j];
            char n_char = needle[j];

            // Convert to lowercase
            if (h_char >= 'A' && h_char <= 'Z') h_char += 32;

            if (h_char != n_char) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Extract filename from path (find last '/')
uint16_t find_filename_start(device const char* path, uint16_t path_len) {
    for (int16_t i = path_len - 1; i >= 0; i--) {
        if (path[i] == '/') {
            return i + 1;
        }
    }
    return 0;
}

kernel void fuzzy_search_kernel(
    device const char* paths [[buffer(0)]],           // All paths (packed, MAX_PATH_LEN each)
    device const uint16_t* path_lengths [[buffer(1)]], // Length of each path
    constant SearchParams& params [[buffer(2)]],       // Search parameters
    device const SearchWord* words [[buffer(3)]],      // Query words (lowercase) - device for GPU tokenization
    device SearchResult* results [[buffer(4)]],        // Output: compact results (matches only)
    device atomic_uint& result_count [[buffer(5)]],    // Atomic counter for results
    device const TokenizeResult* tokenize_result [[buffer(6)]], // Optional: GPU tokenization result
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.path_count) return;

    // Determine actual word count:
    // - If tokenize_result is provided (GPU-native search), use its word_count
    // - If tokenize_result->word_count == 0, it means CPU tokenization (use params.word_count)
    // - Or it means GPU tokenization found no words (whitespace-only query)
    uint32_t word_count = params.word_count;
    if (tokenize_result != nullptr) {
        // Read word count from GPU tokenization result
        uint32_t gpu_count = atomic_load_explicit(
            (device atomic_uint*)&tokenize_result->word_count,
            memory_order_relaxed
        );
        // GPU-native search sets params.word_count = MAX_QUERY_WORDS as sentinel
        // If gpu_count > 0, use it; if gpu_count == 0 and params.word_count == MAX_QUERY_WORDS,
        // this is a GPU-native search with 0 words (whitespace-only query)
        if (gpu_count > 0 && gpu_count <= MAX_QUERY_WORDS) {
            word_count = gpu_count;
        } else if (params.word_count == MAX_QUERY_WORDS && gpu_count == 0) {
            // GPU-native search with 0 words found - no match possible
            word_count = 0;
        }
        // else: CPU tokenization with valid word_count in params
    }

    // Skip if no words to search
    if (word_count == 0) return;

    // Get this thread's path
    device const char* path = paths + (gid * MAX_PATH_LEN);
    uint16_t path_len = path_lengths[gid];

    // Find filename start
    uint16_t filename_start = find_filename_start(path, path_len);
    device const char* filename = path + filename_start;
    uint16_t filename_len = path_len - filename_start;

    // Check if ALL words match (fuzzy)
    int32_t score = 0;

    for (uint32_t w = 0; w < word_count; w++) {
        device const char* word = words[w].chars;
        uint16_t word_len = words[w].len;

        // Check for exact substring in filename (best match)
        if (contains_substring_gpu(filename, filename_len, word, word_len)) {
            score += 100;
            // Bonus for matching at start of filename
            bool starts_with = true;
            for (uint16_t i = 0; i < word_len && i < filename_len; i++) {
                char f_char = filename[i];
                if (f_char >= 'A' && f_char <= 'Z') f_char += 32;
                if (f_char != word[i]) {
                    starts_with = false;
                    break;
                }
            }
            if (starts_with) score += 50;
        }
        // Check fuzzy match in filename only
        else if (fuzzy_match_gpu(filename, filename_len, word, word_len)) {
            score += 20;
        }
        // Word doesn't match filename - reject this path
        // NOTE: We intentionally skip full-path matching to avoid matching on directory names
        else {
            return;  // No match in filename, skip this file
        }
    }

    // Penalty for deeper paths (count slashes)
    int32_t depth = 0;
    for (uint16_t i = 0; i < path_len; i++) {
        if (path[i] == '/') depth++;
    }
    score -= min(depth, 10);

    // Atomically grab a slot in results buffer
    uint idx = atomic_fetch_add_explicit(&result_count, 1, memory_order_relaxed);
    if (idx < params.max_results) {
        results[idx].path_index = gid;
        results[idx].score = score;
    }
}

// Sort results by score (simple bubble sort for small arrays, runs on single thread)
kernel void sort_results_kernel(
    device SearchResult* results [[buffer(0)]],
    device const atomic_uint& result_count [[buffer(1)]],
    constant uint& max_to_sort [[buffer(2)]]
) {
    uint count = min(atomic_load_explicit(&result_count, memory_order_relaxed), max_to_sort);

    // Simple insertion sort (good enough for <1000 items on GPU)
    for (uint i = 1; i < count; i++) {
        SearchResult key = results[i];
        int j = i - 1;
        while (j >= 0 && results[j].score < key.score) {
            results[j + 1] = results[j];
            j--;
        }
        results[j + 1] = key;
    }
}

// Generate TextChar array directly from search results
kernel void generate_results_text_kernel(
    device const char* paths [[buffer(0)]],              // All paths
    device const uint16_t* path_lengths [[buffer(1)]],   // Path lengths
    device const SearchResult* results [[buffer(2)]],    // Search results
    device const atomic_uint& result_count [[buffer(3)]],// Number of results
    device TextChar* text_chars [[buffer(4)]],           // Output text characters
    device atomic_uint& char_count [[buffer(5)]],        // Output char count
    constant float& start_y [[buffer(6)]],               // Y position to start rendering
    constant float& line_height [[buffer(7)]],           // Line height
    constant uint& max_display [[buffer(8)]],            // Max results to display
    uint gid [[thread_position_in_grid]]
) {
    uint num_results = min(atomic_load_explicit(&result_count, memory_order_relaxed), max_display);
    if (gid >= num_results) return;

    SearchResult result = results[gid];
    device const char* path = paths + (result.path_index * MAX_PATH_LEN);
    uint16_t path_len = path_lengths[result.path_index];

    // Calculate Y position for this line
    float y = start_y + (float(gid) * line_height);
    float x = 20.0;

    // Write each character of the path
    uint base_idx = atomic_fetch_add_explicit(&char_count, path_len, memory_order_relaxed);

    uint32_t color = (gid == 0) ? 0x00FF00FF : 0xFFFFFFFF;  // Green for first, white for rest

    for (uint16_t i = 0; i < path_len && (base_idx + i) < 20000; i++) {
        text_chars[base_idx + i].x = x + (float(i) * 8.0);
        text_chars[base_idx + i].y = y;
        text_chars[base_idx + i].char_code = (uint32_t)path[i];
        text_chars[base_idx + i].color = color;
    }
}
"#;

// Replace placeholder with actual header
fn get_filesystem_shader() -> String {
    FILESYSTEM_SHADER.replace("{{APP_SHADER_HEADER}}", APP_SHADER_HEADER)
}

// ============================================================================
// Filesystem Parameters
// ============================================================================

#[repr(C)]
struct FsParams {
    current_directory: u32,
    total_inodes: u32,
    total_entries: u32,
    selected_file: u32,
}

// ============================================================================
// Path Lookup Structures (Issue #21)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PathComponent {
    hash: u32,
    name: [u8; 20],
    len: u16,
    _padding: u16,
}
// Total: 4 + 20 + 2 + 2 = 28 bytes

#[repr(C)]
struct PathLookupParams {
    start_inode: u32,
    component_count: u32,
    total_entries: u32,
    _padding: u32,
}

// ============================================================================
// Batch Lookup Structures (Issue #26)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PathMetadata {
    start_idx: u32,       // Index into components buffer
    component_count: u32, // Number of components in this path
    start_inode: u32,     // Starting inode (ROOT or current_dir)
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BatchResult {
    inode_id: u32,   // Result inode (or INVALID_INODE)
    status: u32,     // 0=success, 1=not_found
}

#[repr(C)]
struct BatchParams {
    total_inodes: u32,
    total_entries: u32,
    batch_size: u32,
    frame_number: u32,  // For cache timestamps
}

// ============================================================================
// GPU Cache Structures (Issue #29)
// ============================================================================

/// GPU-side cache entry - 64 bytes (cache-line aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PathCacheEntry {
    pub path_hash: u64,      // xxHash3 of full path
    pub inode_id: u32,       // Cached inode result
    pub access_count: u32,   // Statistics
    pub timestamp: u64,      // Last access (frame number)
    pub path_len: u32,       // Path length
    pub _padding: u32,
    pub path: [u8; 32],      // Inline path for verification (32 bytes for 64-byte total)
}
// Total: 64 bytes (8+4+4+8+4+4+32)

impl PathCacheEntry {
    pub fn is_valid(&self) -> bool {
        self.path_hash != 0
    }
}

#[derive(Debug, Clone)]
pub struct PathCacheStats {
    pub hits: u32,
    pub misses: u32,
    pub hit_rate: f64,
    pub total_entries: usize,
}

const CACHE_SIZE: usize = 1024;
const CACHE_MASK: u64 = 1023; // CACHE_SIZE - 1

// ============================================================================
// GPU Directory Hash Table (Issue #129)
// ============================================================================
//
// O(1) directory entry lookup using a GPU-resident hash table.
// Key: (parent_inode, name_hash) combined using Fibonacci hashing
// Value: entry_index into the directory entries array
//
// Uses open addressing with linear probing for collision resolution.
// 50% load factor ensures O(1) average case lookup.

/// Combined key hash using Fibonacci/golden ratio hashing
fn dir_hash_key(parent_inode: u32, name_hash: u32) -> u64 {
    let key = ((parent_inode as u64) << 32) | (name_hash as u64);
    key.wrapping_mul(0x9E3779B97F4A7C15)  // Golden ratio hash
}

/// Compute hash slot from key
fn hash_slot(key: u64, mask: u32) -> u32 {
    ((key >> 32) as u32) & mask  // Use high bits for slot
}

/// GPU-resident hash table for O(1) directory lookups
pub struct GpuDirHashTable {
    /// GPU buffer containing [DirHashEntry; capacity]
    pub table_buffer: Buffer,
    /// Table capacity (power of 2)
    pub capacity: u32,
    /// Mask for fast modulo (capacity - 1)
    pub mask: u32,
    /// Number of entries in the table
    pub entry_count: u32,
}

impl GpuDirHashTable {
    /// Build a hash table from directory entries and inodes.
    ///
    /// Creates a GPU-resident hash table for O(1) directory entry lookup.
    /// Uses 2x capacity for ~50% load factor (optimal for linear probing).
    ///
    /// # Arguments
    /// * `device` - Metal device for buffer allocation
    /// * `entries` - Slice of directory entries
    /// * `inodes` - Slice of inodes (to get parent_id for each entry)
    ///
    /// # Example
    /// ```ignore
    /// let hash_table = GpuDirHashTable::build(&device, &dir_entries, &inodes);
    /// ```
    pub fn build(device: &Device, entries: &[DirEntryCompact], inodes: &[InodeCompact]) -> Self {
        // Use 2x capacity for ~50% load factor
        let capacity = ((entries.len() * 2).max(16)).next_power_of_two() as u32;
        let mask = capacity - 1;

        // Initialize empty table
        let mut table = vec![DirHashEntry::empty(); capacity as usize];

        // Insert all entries
        for (i, entry) in entries.iter().enumerate() {
            // Get parent from the inode this entry points to
            let inode_idx = entry.inode_id as usize;
            if inode_idx >= inodes.len() {
                continue;  // Skip invalid entries
            }
            let parent = inodes[inode_idx].parent_id;

            // Compute hash and find slot
            let key = dir_hash_key(parent, entry.name_hash);
            let mut slot = hash_slot(key, mask);

            // Linear probing to find empty slot
            let mut probes = 0;
            while !table[slot as usize].is_empty() && probes < capacity {
                slot = (slot + 1) & mask;
                probes += 1;
            }

            // Insert entry
            if probes < capacity {
                table[slot as usize] = DirHashEntry {
                    parent_inode: parent,
                    name_hash: entry.name_hash,
                    entry_index: i as u32,
                    _padding: 0,
                };
            }
        }

        // Create GPU buffer
        let table_buffer = device.new_buffer_with_data(
            table.as_ptr() as *const _,
            (capacity as u64) * (mem::size_of::<DirHashEntry>() as u64),
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            table_buffer,
            capacity,
            mask,
            entry_count: entries.len() as u32,
        }
    }

    /// Lookup an entry in the hash table (CPU-side, for testing)
    pub fn lookup(&self, parent_inode: u32, name_hash: u32) -> Option<u32> {
        let key = dir_hash_key(parent_inode, name_hash);
        let mut slot = hash_slot(key, self.mask);

        // Read table from GPU buffer
        let table_ptr = self.table_buffer.contents() as *const DirHashEntry;

        // Linear probing with max 32 attempts
        for _ in 0..32 {
            let entry = unsafe { *table_ptr.add(slot as usize) };

            if entry.is_empty() {
                return None;  // Empty slot - not found
            }

            if entry.parent_inode == parent_inode && entry.name_hash == name_hash {
                return Some(entry.entry_index);
            }

            slot = (slot + 1) & self.mask;
        }

        None  // Max probes exceeded
    }

    /// Get load factor (for diagnostics)
    pub fn load_factor(&self) -> f32 {
        self.entry_count as f32 / self.capacity as f32
    }

    /// Get table size in bytes
    pub fn size_bytes(&self) -> usize {
        (self.capacity as usize) * mem::size_of::<DirHashEntry>()
    }
}

// ============================================================================
// GpuFilesystem - implements GpuApp
// ============================================================================

pub struct GpuFilesystem {
    device: Device,
    command_queue: CommandQueue,  // Reusable queue to avoid context leaks

    // Pipelines (reusing AppBuilder pattern)
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,
    path_lookup_pipeline: ComputePipelineState,  // Issue #21
    batch_lookup_pipeline: ComputePipelineState, // Issue #26

    // Filesystem buffers (slots 3-6)
    inode_buffer: Buffer,           // Slot 3
    dir_entry_buffer: Buffer,       // Slot 4
    vertices_buffer: Buffer,        // Slot 5
    vertex_count_buffer: Buffer,    // Slot 6

    // Path lookup buffers (Issue #21)
    path_components_buffer: Buffer,
    path_lookup_params_buffer: Buffer,
    path_result_inode_buffer: Buffer,
    path_result_status_buffer: Buffer,

    // Batch lookup buffers (Issue #26)
    batch_components_buffer: Buffer,   // All PathComponents (flattened)
    batch_metadata_buffer: Buffer,     // PathMetadata array
    batch_results_buffer: Buffer,      // BatchResult array
    batch_params_buffer: Buffer,       // BatchParams

    // GPU cache buffers (Issue #29)
    cache_buffer: Buffer,              // PathCacheEntry array (1024 entries)
    cache_hits_buffer: Buffer,         // Atomic hit counter
    cache_misses_buffer: Buffer,       // Atomic miss counter

    // Hash table for O(1) lookups (Issue #129)
    #[allow(dead_code)]  // Reserved for single-entry lookup
    hash_lookup_pipeline: ComputePipelineState,     // dir_lookup_hash kernel
    path_lookup_hash_pipeline: ComputePipelineState, // path_lookup_hash kernel
    #[allow(dead_code)]  // Reserved for batch lookup API
    batch_hash_lookup_pipeline: ComputePipelineState, // batch_dir_lookup_hash kernel
    dir_hash_table: Option<GpuDirHashTable>,        // Built lazily when index is created
    hash_table_mask_buffer: Buffer,                 // Table mask for GPU
    #[allow(dead_code)]  // Reserved for single-entry lookup
    hash_lookup_result_buffer: Buffer,              // Result buffer for hash lookups

    // App params
    params_buffer: Buffer,

    // State
    current_directory: u32,
    selected_file: u32,
    #[allow(dead_code)]  // Reserved for capacity checks
    max_inodes: usize,
    #[allow(dead_code)]  // Reserved for capacity checks
    max_entries: usize,
    current_inode_count: usize,
    current_entry_count: usize,
    max_batch_size: usize,  // Issue #26
    frame_number: Cell<u32>, // Issue #29 - for cache timestamps (interior mutability)
}

impl GpuFilesystem {
    pub fn new(device: &Device, max_inodes: usize) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "GpuFilesystem");

        // Create reusable command queue (avoid context leaks from creating many queues)
        let command_queue = device.new_command_queue();

        // Compile shaders using AppBuilder
        let library = builder.compile_library(&get_filesystem_shader())?;

        let compute_pipeline = builder.create_compute_pipeline(
            &library,
            "filesystem_compute_kernel"
        )?;

        let render_pipeline = builder.create_render_pipeline(
            &library,
            "filesystem_vertex_shader",
            "filesystem_fragment_shader"
        )?;

        let path_lookup_pipeline = builder.create_compute_pipeline(
            &library,
            "path_lookup_kernel"
        )?;

        let batch_lookup_pipeline = builder.create_compute_pipeline(
            &library,
            "batch_path_lookup_kernel"
        )?;

        // Hash table lookup pipelines (Issue #129)
        let hash_lookup_pipeline = builder.create_compute_pipeline(
            &library,
            "dir_lookup_hash"
        )?;

        let path_lookup_hash_pipeline = builder.create_compute_pipeline(
            &library,
            "path_lookup_hash"
        )?;

        let batch_hash_lookup_pipeline = builder.create_compute_pipeline(
            &library,
            "batch_dir_lookup_hash"
        )?;

        // Allocate buffers (following GpuMemory pattern)
        let max_entries = max_inodes * 10; // Assume avg 10 entries per directory

        let inode_buffer = builder.create_buffer(max_inodes * mem::size_of::<InodeCompact>());
        let dir_entry_buffer = builder.create_buffer(max_entries * mem::size_of::<DirEntryCompact>());
        let vertices_buffer = builder.create_buffer(max_entries * 6 * mem::size_of::<f32>() * 8);
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<u32>());
        let params_buffer = builder.create_buffer(mem::size_of::<FsParams>());

        // Path lookup buffers (Issue #21)
        let path_components_buffer = builder.create_buffer(MAX_PATH_DEPTH * mem::size_of::<PathComponent>());
        let path_lookup_params_buffer = builder.create_buffer(mem::size_of::<PathLookupParams>());
        let path_result_inode_buffer = builder.create_buffer(mem::size_of::<u32>());
        let path_result_status_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Batch lookup buffers (Issue #26)
        let max_batch_size = 256; // Support up to 256 concurrent paths
        let batch_components_buffer = builder.create_buffer(
            max_batch_size * MAX_PATH_DEPTH * mem::size_of::<PathComponent>()
        );
        let batch_metadata_buffer = builder.create_buffer(
            max_batch_size * mem::size_of::<PathMetadata>()
        );
        let batch_results_buffer = builder.create_buffer(
            max_batch_size * mem::size_of::<BatchResult>()
        );
        let batch_params_buffer = builder.create_buffer(mem::size_of::<BatchParams>());

        // GPU cache buffers (Issue #29)
        let cache_buffer = builder.create_buffer(CACHE_SIZE * mem::size_of::<PathCacheEntry>());
        let cache_hits_buffer = builder.create_buffer(mem::size_of::<u32>());
        let cache_misses_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Hash table buffers (Issue #129)
        let hash_table_mask_buffer = builder.create_buffer(mem::size_of::<u32>());
        let hash_lookup_result_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Initialize root directory inode
        let root = InodeCompact::new(ROOT_INODE_ID as u64, 0, FileType::Directory);
        unsafe {
            let ptr = inode_buffer.contents() as *mut InodeCompact;
            *ptr = root;
        }

        // Initialize GPU cache (Issue #29)
        unsafe {
            let ptr = cache_buffer.contents() as *mut PathCacheEntry;
            for i in 0..CACHE_SIZE {
                *ptr.add(i) = PathCacheEntry {
                    path_hash: 0,
                    inode_id: 0,
                    access_count: 0,
                    timestamp: 0,
                    path_len: 0,
                    _padding: 0,
                    path: [0; 32],
                };
            }

            // Initialize statistics
            *(cache_hits_buffer.contents() as *mut u32) = 0;
            *(cache_misses_buffer.contents() as *mut u32) = 0;
        }

        Ok(Self {
            device: device.clone(),
            command_queue,
            compute_pipeline,
            render_pipeline,
            path_lookup_pipeline,
            batch_lookup_pipeline,
            inode_buffer,
            dir_entry_buffer,
            vertices_buffer,
            vertex_count_buffer,
            path_components_buffer,
            path_lookup_params_buffer,
            path_result_inode_buffer,
            path_result_status_buffer,
            batch_components_buffer,
            batch_metadata_buffer,
            batch_results_buffer,
            batch_params_buffer,
            params_buffer,
            current_directory: ROOT_INODE_ID,
            selected_file: 0,
            max_inodes,
            max_entries,
            current_inode_count: 1, // Root inode
            current_entry_count: 0,
            max_batch_size,
            cache_buffer,
            cache_hits_buffer,
            cache_misses_buffer,
            frame_number: Cell::new(0),
            // Hash table (Issue #129)
            hash_lookup_pipeline,
            path_lookup_hash_pipeline,
            batch_hash_lookup_pipeline,
            dir_hash_table: None,  // Built lazily when files are added
            hash_table_mask_buffer,
            hash_lookup_result_buffer,
        })
    }

    /// Add a file to the filesystem (for testing)
    pub fn add_file(&mut self, parent_id: u32, name: &str, file_type: FileType) -> Result<u32, String> {
        if self.current_inode_count >= self.max_inodes {
            return Err("Too many inodes".to_string());
        }

        let inode_id = self.current_inode_count as u32;
        let inode = InodeCompact::new(inode_id as u64, parent_id, file_type);

        // Write inode
        unsafe {
            let ptr = self.inode_buffer.contents() as *mut InodeCompact;
            *ptr.add(inode_id as usize) = inode;
        }

        // Create directory entry
        let entry = DirEntryCompact::new(inode_id, name);
        unsafe {
            let ptr = self.dir_entry_buffer.contents() as *mut DirEntryCompact;
            *ptr.add(self.current_entry_count) = entry;
        }

        self.current_inode_count += 1;
        self.current_entry_count += 1;

        // Invalidate hash table (needs rebuild)
        self.dir_hash_table = None;

        Ok(inode_id)
    }

    /// Build the hash table for O(1) directory lookups (Issue #129)
    ///
    /// Call this after adding all files to enable O(1) path lookup.
    /// The hash table is built on CPU and uploaded to GPU for fast lookups.
    ///
    /// # Example
    /// ```ignore
    /// fs.add_file(ROOT_INODE_ID, "src", FileType::Directory)?;
    /// fs.add_file(1, "main.rs", FileType::Regular)?;
    /// fs.build_hash_table();  // Enable O(1) lookups
    /// ```
    pub fn build_hash_table(&mut self) {
        if self.current_entry_count == 0 {
            self.dir_hash_table = None;
            return;
        }

        // Read entries and inodes from GPU buffers
        let entries: Vec<DirEntryCompact> = unsafe {
            let ptr = self.dir_entry_buffer.contents() as *const DirEntryCompact;
            (0..self.current_entry_count)
                .map(|i| *ptr.add(i))
                .collect()
        };

        let inodes: Vec<InodeCompact> = unsafe {
            let ptr = self.inode_buffer.contents() as *const InodeCompact;
            (0..self.current_inode_count)
                .map(|i| *ptr.add(i))
                .collect()
        };

        // Build hash table
        let hash_table = GpuDirHashTable::build(&self.device, &entries, &inodes);

        // Update mask buffer for GPU
        unsafe {
            *(self.hash_table_mask_buffer.contents() as *mut u32) = hash_table.mask;
        }

        self.dir_hash_table = Some(hash_table);
    }

    /// Check if hash table is built
    pub fn has_hash_table(&self) -> bool {
        self.dir_hash_table.is_some()
    }

    /// Get hash table statistics
    pub fn hash_table_stats(&self) -> Option<(f32, usize)> {
        self.dir_hash_table.as_ref().map(|ht| (ht.load_factor(), ht.size_bytes()))
    }

    /// Lookup a path using O(1) hash table (Issue #129)
    ///
    /// This is significantly faster than lookup_path() for large filesystems.
    /// Requires build_hash_table() to be called first.
    ///
    /// # Example
    /// ```ignore
    /// fs.build_hash_table();
    /// let inode = fs.lookup_path_hash("/src/main.rs")?;
    /// ```
    pub fn lookup_path_hash(&self, path: &str) -> Result<u32, String> {
        // Ensure hash table is built
        let hash_table = self.dir_hash_table.as_ref()
            .ok_or_else(|| "Hash table not built - call build_hash_table() first".to_string())?;

        // Handle empty path
        if path.is_empty() {
            return Err("Empty path".to_string());
        }

        // Handle root path
        if path == "/" {
            return Ok(ROOT_INODE_ID);
        }

        // Determine starting inode (absolute vs relative)
        let (start_inode, path_to_parse) = if path.starts_with('/') {
            (ROOT_INODE_ID, &path[1..])
        } else {
            (self.current_directory, path)
        };

        // Split path into components
        let components: Vec<&str> = path_to_parse
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if components.is_empty() {
            return Ok(start_inode);
        }

        if components.len() > MAX_PATH_DEPTH {
            return Err(format!("Path too deep (max {})", MAX_PATH_DEPTH));
        }

        // Create PathComponent structs with hashes
        let mut path_components = Vec::with_capacity(components.len());
        for component in &components {
            if component.len() > 20 {
                return Err(format!("Component '{}' too long (max 20 chars)", component));
            }

            let mut pc = PathComponent {
                hash: xxhash3(component.as_bytes()),
                name: [0; 20],
                len: component.len() as u16,
                _padding: 0,
            };
            pc.name[..component.len()].copy_from_slice(component.as_bytes());
            path_components.push(pc);
        }

        // Write components to GPU buffer
        unsafe {
            let ptr = self.path_components_buffer.contents() as *mut PathComponent;
            for (i, component) in path_components.iter().enumerate() {
                *ptr.add(i) = *component;
            }
        }

        // Initialize result buffers
        unsafe {
            *(self.path_result_inode_buffer.contents() as *mut u32) = INVALID_INODE;
            *(self.path_result_status_buffer.contents() as *mut u32) = 1;  // Not found
        }

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.path_lookup_hash_pipeline);
        encoder.set_buffer(0, Some(&hash_table.table_buffer), 0);
        encoder.set_buffer(1, Some(&self.hash_table_mask_buffer), 0);
        encoder.set_buffer(2, Some(&self.dir_entry_buffer), 0);
        encoder.set_buffer(3, Some(&self.path_components_buffer), 0);

        // Write component count and start inode as constants
        let component_count = path_components.len() as u32;
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &component_count as *const _ as *const _);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &start_inode as *const _ as *const _);

        encoder.set_buffer(6, Some(&self.path_result_inode_buffer), 0);
        encoder.set_buffer(7, Some(&self.path_result_status_buffer), 0);

        // Dispatch single thread (path walk is sequential)
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        unsafe {
            let status = *(self.path_result_status_buffer.contents() as *const u32);
            if status == 0 {
                let inode_id = *(self.path_result_inode_buffer.contents() as *const u32);
                Ok(inode_id)
            } else {
                Err(format!("Path not found: {}", path))
            }
        }
    }

    /// Single directory entry lookup using hash table (Issue #129)
    ///
    /// Returns the entry index for a given (parent_inode, name_hash) pair.
    /// This is an O(1) operation.
    pub fn lookup_entry_hash(&self, parent_inode: u32, name_hash: u32) -> Option<u32> {
        self.dir_hash_table.as_ref()?.lookup(parent_inode, name_hash)
    }

    /// Lookup a path using GPU parallel hash search (Issue #21)
    ///
    /// Examples:
    ///   lookup_path("/foo/bar")        - absolute path from root
    ///   lookup_path("foo/bar")         - relative from current_directory
    ///   lookup_path("/")               - returns root inode
    ///
    /// Returns: Ok(inode_id) on success, Err(message) if not found
    pub fn lookup_path(&self, path: &str) -> Result<u32, String> {
        // Handle empty path
        if path.is_empty() {
            return Err("Empty path".to_string());
        }

        // Handle root path
        if path == "/" {
            return Ok(ROOT_INODE_ID);
        }

        // Determine starting inode (absolute vs relative)
        let (start_inode, path_to_parse) = if path.starts_with('/') {
            (ROOT_INODE_ID, &path[1..])  // Absolute path
        } else {
            (self.current_directory, path)  // Relative path
        };

        // Split path into components
        let components: Vec<&str> = path_to_parse
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if components.is_empty() {
            return Ok(start_inode);
        }

        if components.len() > MAX_PATH_DEPTH {
            return Err(format!("Path too deep (max {})", MAX_PATH_DEPTH));
        }

        // Create PathComponent structs with hashes
        let mut path_components = Vec::with_capacity(components.len());
        for component in &components {
            if component.len() > 20 {
                return Err(format!("Component '{}' too long (max 20 chars)", component));
            }

            let mut pc = PathComponent {
                hash: xxhash3(component.as_bytes()),
                name: [0; 20],
                len: component.len() as u16,
                _padding: 0,
            };
            pc.name[..component.len()].copy_from_slice(component.as_bytes());
            path_components.push(pc);
        }

        // Write components to GPU buffer
        unsafe {
            let ptr = self.path_components_buffer.contents() as *mut PathComponent;
            for (i, component) in path_components.iter().enumerate() {
                *ptr.add(i) = *component;
            }
        }

        // Write lookup params
        unsafe {
            let params = self.path_lookup_params_buffer.contents() as *mut PathLookupParams;
            *params = PathLookupParams {
                start_inode,
                component_count: path_components.len() as u32,
                total_entries: self.current_entry_count as u32,
                _padding: 0,
            };
        }

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.path_lookup_pipeline);
        encoder.set_buffer(0, Some(&self.inode_buffer), 0);
        encoder.set_buffer(1, Some(&self.dir_entry_buffer), 0);
        encoder.set_buffer(2, Some(&self.path_lookup_params_buffer), 0);
        encoder.set_buffer(3, Some(&self.path_components_buffer), 0);
        encoder.set_buffer(4, Some(&self.path_result_inode_buffer), 0);
        encoder.set_buffer(5, Some(&self.path_result_status_buffer), 0);

        // Dispatch 1024 threads (1 threadgroup)
        let threads = MTLSize::new(1024, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        unsafe {
            let status = *(self.path_result_status_buffer.contents() as *const u32);
            if status == 0 {
                let inode_id = *(self.path_result_inode_buffer.contents() as *const u32);
                Ok(inode_id)
            } else {
                Err(format!("Path not found: {}", path))
            }
        }
    }

    /// Lookup multiple paths in a single GPU dispatch (Issue #26)
    ///
    /// # Example
    /// ```
    /// let paths = vec!["/src/main.rs", "/src/lib.rs", "/tests/test.rs"];
    /// let results = fs.lookup_batch(&paths)?;
    ///
    /// for (path, result) in paths.iter().zip(results.iter()) {
    ///     match result {
    ///         Ok(inode_id) => println!("{} → {}", path, inode_id),
    ///         Err(e) => println!("{} → Error: {}", path, e),
    ///     }
    /// }
    /// ```
    pub fn lookup_batch(&self, paths: &[&str]) -> Result<Vec<Result<u32, String>>, String> {
        // 1. Validate input
        if paths.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = paths.len();
        if batch_size > self.max_batch_size {
            return Err(format!(
                "Batch size {} exceeds max {}",
                batch_size, self.max_batch_size
            ));
        }

        // 2. Parse all paths into components
        let mut all_components = Vec::new();
        let mut metadata = Vec::new();

        for path in paths.iter() {
            // Handle empty path
            if path.is_empty() {
                return Err("Empty path in batch".to_string());
            }

            // Handle root path
            if *path == "/" {
                metadata.push(PathMetadata {
                    start_idx: all_components.len() as u32,
                    component_count: 0,
                    start_inode: ROOT_INODE_ID,
                    _padding: 0,
                });
                continue;
            }

            // Determine starting inode (absolute vs relative)
            let (start_inode, path_to_parse) = if path.starts_with('/') {
                (ROOT_INODE_ID, &path[1..])
            } else {
                (self.current_directory, *path)
            };

            // Split path into components
            let components: Vec<&str> = path_to_parse
                .split('/')
                .filter(|s| !s.is_empty())
                .collect();

            if components.is_empty() {
                metadata.push(PathMetadata {
                    start_idx: all_components.len() as u32,
                    component_count: 0,
                    start_inode,
                    _padding: 0,
                });
                continue;
            }

            if components.len() > MAX_PATH_DEPTH {
                return Err(format!("Path too deep (max {}): {}", MAX_PATH_DEPTH, path));
            }

            // Create PathComponent structs
            let start_idx = all_components.len() as u32;

            for component in &components {
                if component.len() > 20 {
                    return Err(format!("Component '{}' too long (max 20 chars)", component));
                }

                let mut pc = PathComponent {
                    hash: xxhash3(component.as_bytes()),
                    name: [0; 20],
                    len: component.len() as u16,
                    _padding: 0,
                };
                pc.name[..component.len()].copy_from_slice(component.as_bytes());
                all_components.push(pc);
            }

            metadata.push(PathMetadata {
                start_idx,
                component_count: components.len() as u32,
                start_inode,
                _padding: 0,
            });
        }

        // 3. Write to GPU buffers
        unsafe {
            // Write components
            let comp_ptr = self.batch_components_buffer.contents() as *mut PathComponent;
            for (i, comp) in all_components.iter().enumerate() {
                *comp_ptr.add(i) = *comp;
            }

            // Write metadata
            let meta_ptr = self.batch_metadata_buffer.contents() as *mut PathMetadata;
            for (i, meta) in metadata.iter().enumerate() {
                *meta_ptr.add(i) = *meta;
            }

            // Write batch parameters
            let current_frame = self.frame_number.get();
            let params = self.batch_params_buffer.contents() as *mut BatchParams;
            *params = BatchParams {
                total_inodes: self.current_inode_count as u32,
                total_entries: self.current_entry_count as u32,
                batch_size: batch_size as u32,
                frame_number: current_frame,
            };
        }

        // Increment frame number for next batch
        self.frame_number.set(self.frame_number.get().wrapping_add(1));

        // 4. Dispatch GPU kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.batch_lookup_pipeline);
        encoder.set_buffer(0, Some(&self.inode_buffer), 0);
        encoder.set_buffer(1, Some(&self.dir_entry_buffer), 0);
        encoder.set_buffer(2, Some(&self.batch_params_buffer), 0);  // total_inodes
        encoder.set_buffer(3, Some(&self.batch_params_buffer), 4);  // total_entries
        encoder.set_buffer(4, Some(&self.batch_components_buffer), 0);
        encoder.set_buffer(5, Some(&self.batch_metadata_buffer), 0);
        encoder.set_buffer(6, Some(&self.batch_results_buffer), 0);
        encoder.set_buffer(7, Some(&self.batch_params_buffer), 8);  // batch_size
        encoder.set_buffer(8, Some(&self.cache_buffer), 0);      // GPU cache (Issue #29)
        encoder.set_buffer(9, Some(&self.cache_hits_buffer), 0);  // Cache hit counter
        encoder.set_buffer(10, Some(&self.cache_misses_buffer), 0); // Cache miss counter
        encoder.set_buffer(11, Some(&self.batch_params_buffer), 12); // frame_number

        // One threadgroup per path, 1024 threads per group
        let threadgroups = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_group = MTLSize::new(1024, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // 5. Read results
        let mut results = Vec::with_capacity(batch_size);
        unsafe {
            let result_ptr = self.batch_results_buffer.contents() as *const BatchResult;

            for i in 0..batch_size {
                let result = *result_ptr.add(i);

                if result.status == 0 {
                    results.push(Ok(result.inode_id));
                } else {
                    results.push(Err(format!("Path not found: {}", paths[i])));
                }
            }
        }

        Ok(results)
    }

    /// Get GPU cache statistics (Issue #29)
    pub fn cache_stats(&self) -> PathCacheStats {
        unsafe {
            let hits = *(self.cache_hits_buffer.contents() as *const u32);
            let misses = *(self.cache_misses_buffer.contents() as *const u32);
            let total = hits + misses;
            let hit_rate = if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            };

            // Count non-zero entries in cache
            let cache_ptr = self.cache_buffer.contents() as *const PathCacheEntry;
            let mut total_entries = 0;
            for i in 0..CACHE_SIZE {
                let entry = *cache_ptr.add(i);
                if entry.path_hash != 0 {
                    total_entries += 1;
                }
            }

            PathCacheStats {
                hits,
                misses,
                hit_rate,
                total_entries,
            }
        }
    }

    /// Clear GPU cache and reset statistics (Issue #29)
    pub fn clear_cache(&mut self) {
        unsafe {
            // Clear cache entries
            let ptr = self.cache_buffer.contents() as *mut PathCacheEntry;
            for i in 0..CACHE_SIZE {
                *ptr.add(i) = PathCacheEntry {
                    path_hash: 0,
                    inode_id: 0,
                    access_count: 0,
                    timestamp: 0,
                    path_len: 0,
                    _padding: 0,
                    path: [0; 32],
                };
            }

            // Reset statistics
            *(self.cache_hits_buffer.contents() as *mut u32) = 0;
            *(self.cache_misses_buffer.contents() as *mut u32) = 0;
        }

        // Reset frame number
        self.frame_number.set(0);
    }
}

// ============================================================================
// GpuApp Implementation
// ============================================================================

impl GpuApp for GpuFilesystem {
    fn name(&self) -> &str {
        "GPU-Native Filesystem"
    }

    fn compute_pipeline(&self) -> &ComputePipelineState {
        &self.compute_pipeline
    }

    fn render_pipeline(&self) -> &RenderPipelineState {
        &self.render_pipeline
    }

    fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    fn vertex_count(&self) -> usize {
        // Read from GPU buffer
        unsafe {
            *(self.vertex_count_buffer.contents() as *const u32) as usize
        }
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.inode_buffer,         // Slot 3
            &self.dir_entry_buffer,     // Slot 4
            &self.vertices_buffer,      // Slot 5
            &self.vertex_count_buffer,  // Slot 6
        ]
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn update_params(&mut self, _frame_state: &FrameState, _delta_time: f32) {
        // Update filesystem parameters
        unsafe {
            let params = self.params_buffer.contents() as *mut FsParams;
            (*params) = FsParams {
                current_directory: self.current_directory,
                total_inodes: self.current_inode_count as u32,
                total_entries: self.current_entry_count as u32,
                selected_file: self.selected_file,
            };
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        // Handle keyboard navigation
        if event.event_type == InputEventType::KeyDown as u16 {
            match event.keycode {
                0x7D => { // Down arrow
                    self.selected_file = (self.selected_file + 1) % self.current_entry_count as u32;
                }
                0x7E => { // Up arrow
                    if self.selected_file > 0 {
                        self.selected_file -= 1;
                    }
                }
                _ => {}
            }
        }
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.1, 0.1, 0.12, 1.0)
    }
}

// ============================================================================
// GPU Path Search (for filesystem browser fuzzy search)
// ============================================================================

const GPU_MAX_PATH_LEN: usize = 256;
const GPU_MAX_QUERY_WORDS: usize = 8;
const GPU_MAX_WORD_LEN: usize = 32;
const GPU_MAX_RESULTS: usize = 1000;  // Max matches to collect
const GPU_MAX_QUERY_LEN: usize = 256; // Max query length (Issue #79)

#[repr(C)]
#[derive(Copy, Clone)]
struct SearchParams {
    path_count: u32,
    word_count: u32,
    max_results: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SearchWord {
    chars: [u8; 32],
    len: u16,
    _padding: u16,
}

/// Compact search result from GPU
#[repr(C)]
#[derive(Copy, Clone)]
struct SearchResult {
    path_index: u32,
    score: i32,
}

/// GPU tokenization result (Issue #79)
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct TokenizeResult {
    word_count: u32,
    _padding: [u32; 3],
}

// ============================================================================
// SearchHandle - Async Search Result (Issue #76)
// ============================================================================

/// Handle to a pending async search operation.
///
/// Allows non-blocking check of search completion and result retrieval.
///
/// # Example
/// ```ignore
/// let handle = search.search_async("query");
///
/// // Do other work while GPU processes...
///
/// // Check if ready (non-blocking)
/// if let Some(results) = handle.try_get_results() {
///     for (path_idx, score) in results {
///         println!("Found: {} (score: {})", path_idx, score);
///     }
/// }
///
/// // Or wait for results (blocking)
/// let results = handle.wait_and_get_results();
/// ```
pub struct SearchHandle {
    shared_event: SharedEvent,
    signal_value: u64,
    results_buffer: Buffer,
    result_count_buffer: Buffer,
    max_results: usize,
}

impl SearchHandle {
    /// Check if search is complete (non-blocking)
    pub fn is_complete(&self) -> bool {
        self.shared_event.signaled_value() >= self.signal_value
    }

    /// Get results if ready, None if still processing (non-blocking)
    pub fn try_get_results(&self) -> Option<Vec<(usize, i32)>> {
        if self.is_complete() {
            Some(self.read_results())
        } else {
            None
        }
    }

    /// Block until complete and return results
    pub fn wait_and_get_results(self) -> Vec<(usize, i32)> {
        // Spin-wait with yield (could use more sophisticated waiting)
        while !self.is_complete() {
            std::hint::spin_loop();
        }
        self.read_results()
    }

    /// Get the signal value for this search (for debugging/tracking)
    pub fn signal_value(&self) -> u64 {
        self.signal_value
    }

    fn read_results(&self) -> Vec<(usize, i32)> {
        let mut results = Vec::new();
        unsafe {
            let count = (*(self.result_count_buffer.contents() as *const u32) as usize)
                .min(self.max_results);
            let ptr = self.results_buffer.contents() as *const SearchResult;
            for i in 0..count {
                let r = *ptr.add(i);
                results.push((r.path_index as usize, r.score));
            }
        }
        results
    }
}

/// GPU-accelerated fuzzy path search
///
/// Searches millions of paths in parallel using Metal compute shaders.
/// Each GPU thread checks one path against all query words.
/// Results stay on GPU and can be rendered directly without CPU readback.
pub struct GpuPathSearch {
    device: Device,
    command_queue: CommandQueue,
    search_pipeline: ComputePipelineState,
    sort_pipeline: ComputePipelineState,
    text_gen_pipeline: ComputePipelineState,
    tokenize_pipeline: ComputePipelineState, // Issue #79: GPU tokenization

    // Path storage
    paths_buffer: Buffer,        // All paths (packed, 256 bytes each)
    path_lengths_buffer: Buffer, // Length of each path (u16)

    // Query
    params_buffer: Buffer,       // SearchParams
    words_buffer: Buffer,        // Query words

    // GPU-native query processing (Issue #79)
    raw_query_buffer: Buffer,       // Raw query bytes (only CPU memcpy)
    tokenize_result_buffer: Buffer, // TokenizeResult (word count)

    // Results (stay on GPU)
    results_buffer: Buffer,      // Compact SearchResult array
    result_count_buffer: Buffer, // Atomic counter

    // Text output (for direct GPU rendering)
    text_chars_buffer: Buffer,   // TextChar array for renderer
    char_count_buffer: Buffer,   // Atomic counter for chars

    // Constants for text generation
    text_params_buffer: Buffer,  // start_y, line_height
    max_sort_buffer: Buffer,     // max_to_sort (u32)

    // State
    max_paths: usize,
    current_path_count: usize,
    paths: Vec<String>,          // Original paths for result lookup (only for legacy API)

    // Async search support (Issue #76)
    shared_event: SharedEvent,
    shared_event_listener: SharedEventListener,
    next_signal_value: Arc<AtomicU64>,
    _callback_queue: Queue,
}

impl GpuPathSearch {
    pub fn new(device: &Device, max_paths: usize) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "GpuPathSearch");
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = builder.compile_library(&get_filesystem_shader())?;
        let search_pipeline = builder.create_compute_pipeline(&library, "fuzzy_search_kernel")?;
        let sort_pipeline = builder.create_compute_pipeline(&library, "sort_results_kernel")?;
        let text_gen_pipeline = builder.create_compute_pipeline(&library, "generate_results_text_kernel")?;
        let tokenize_pipeline = builder.create_compute_pipeline(&library, "tokenize_query_kernel")?;

        // Path storage
        let paths_buffer = builder.create_buffer(max_paths * GPU_MAX_PATH_LEN);
        let path_lengths_buffer = builder.create_buffer(max_paths * mem::size_of::<u16>());

        // Query buffers
        let params_buffer = builder.create_buffer(mem::size_of::<SearchParams>());
        let words_buffer = builder.create_buffer(GPU_MAX_QUERY_WORDS * mem::size_of::<SearchWord>());

        // GPU-native query processing (Issue #79)
        let raw_query_buffer = builder.create_buffer(GPU_MAX_QUERY_LEN);
        let tokenize_result_buffer = builder.create_buffer(mem::size_of::<TokenizeResult>());

        // Results buffer (compact, max 1000 results)
        let results_buffer = builder.create_buffer(GPU_MAX_RESULTS * mem::size_of::<SearchResult>());
        let result_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Text output for direct GPU rendering
        let text_chars_buffer = builder.create_buffer(20000 * 16); // 20K chars * 16 bytes each
        let char_count_buffer = builder.create_buffer(mem::size_of::<u32>());
        let text_params_buffer = builder.create_buffer(8); // start_y, line_height (f32 each)
        let max_sort_buffer = builder.create_buffer(mem::size_of::<u32>()); // max_to_sort

        // Async search support (Issue #76)
        let shared_event = device.new_shared_event();
        let callback_queue = Queue::create(
            "com.gpu-native-os.search-callback",
            QueueAttribute::Serial,
        );
        let shared_event_listener = SharedEventListener::from_queue(&callback_queue);

        Ok(Self {
            device: device.clone(),
            command_queue,
            search_pipeline,
            sort_pipeline,
            text_gen_pipeline,
            tokenize_pipeline,
            paths_buffer,
            path_lengths_buffer,
            params_buffer,
            words_buffer,
            raw_query_buffer,
            tokenize_result_buffer,
            results_buffer,
            result_count_buffer,
            text_chars_buffer,
            char_count_buffer,
            text_params_buffer,
            max_sort_buffer,
            max_paths,
            current_path_count: 0,
            paths: Vec::with_capacity(max_paths),
            shared_event,
            shared_event_listener,
            next_signal_value: Arc::new(AtomicU64::new(1)),
            _callback_queue: callback_queue,
        })
    }

    /// Add paths to the search index (call once after scanning filesystem)
    pub fn add_paths(&mut self, paths: &[String]) -> Result<(), String> {
        if paths.len() > self.max_paths {
            return Err(format!("Too many paths: {} > {}", paths.len(), self.max_paths));
        }

        self.paths = paths.to_vec();
        self.current_path_count = paths.len();

        // Write paths to GPU buffer
        unsafe {
            let path_ptr = self.paths_buffer.contents() as *mut u8;
            let len_ptr = self.path_lengths_buffer.contents() as *mut u16;

            for (i, path) in paths.iter().enumerate() {
                let bytes = path.as_bytes();
                let len = bytes.len().min(GPU_MAX_PATH_LEN - 1);

                // Copy path (truncated if necessary)
                let dest = path_ptr.add(i * GPU_MAX_PATH_LEN);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                // Zero-terminate
                *dest.add(len) = 0;

                // Store length
                *len_ptr.add(i) = len as u16;
            }
        }

        Ok(())
    }

    /// Perform GPU fuzzy search and generate text for rendering
    /// ALL computation stays on GPU - including tokenization! Returns result count.
    ///
    /// CPU work: ONE memcpy of raw query bytes
    /// GPU work: tokenization, search, sort, text generation
    pub fn search_and_render(&self, query: &str, start_y: f32, line_height: f32, max_display: usize) -> usize {
        if query.is_empty() || self.current_path_count == 0 {
            unsafe { *(self.char_count_buffer.contents() as *mut u32) = 0; }
            return 0;
        }

        let bytes = query.as_bytes();
        let query_len = bytes.len().min(GPU_MAX_QUERY_LEN);
        let max_results = GPU_MAX_RESULTS.min(max_display);

        // =================================================================
        // THE ONLY CPU WORK: Copy raw query bytes to GPU
        // =================================================================
        unsafe {
            // Copy raw query (NO tokenization, NO lowercasing - GPU will do it)
            let query_ptr = self.raw_query_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), query_ptr, query_len);
            std::ptr::write_bytes(query_ptr.add(query_len), 0, GPU_MAX_QUERY_LEN - query_len);

            // Reset counters
            *(self.result_count_buffer.contents() as *mut u32) = 0;
            *(self.char_count_buffer.contents() as *mut u32) = 0;

            // Reset tokenize result
            let tokenize_result = self.tokenize_result_buffer.contents() as *mut TokenizeResult;
            (*tokenize_result).word_count = 0;

            // Write params (word_count = MAX as sentinel for GPU tokenization)
            let params = self.params_buffer.contents() as *mut SearchParams;
            *params = SearchParams {
                path_count: self.current_path_count as u32,
                word_count: GPU_MAX_QUERY_WORDS as u32, // GPU will use actual count
                max_results: max_results as u32,
                _padding: 0,
            };

            // Text params: [start_y, line_height]
            let tp = self.text_params_buffer.contents() as *mut f32;
            *tp = start_y;
            *tp.add(1) = line_height;

            // max_sort as u32
            let ms = self.max_sort_buffer.contents() as *mut u32;
            *ms = max_results.min(100) as u32;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // =================================================================
        // Pass 0: GPU Tokenization (Issue #79)
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.tokenize_pipeline);
            enc.set_buffer(0, Some(&self.raw_query_buffer), 0);
            enc.set_buffer(1, Some(&self.words_buffer), 0);
            enc.set_buffer(2, Some(&self.tokenize_result_buffer), 0);
            enc.set_bytes(3, mem::size_of::<u32>() as u64, &(query_len as u32) as *const _ as *const _);

            let threads = query_len as u64;
            let threadgroup_size = threads.min(256);
            enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threadgroup_size, 1, 1));
            enc.end_encoding();
        }

        // =================================================================
        // Pass 1: GPU Search (uses GPU-tokenized word count)
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.search_pipeline);
            enc.set_buffer(0, Some(&self.paths_buffer), 0);
            enc.set_buffer(1, Some(&self.path_lengths_buffer), 0);
            enc.set_buffer(2, Some(&self.params_buffer), 0);
            enc.set_buffer(3, Some(&self.words_buffer), 0);
            enc.set_buffer(4, Some(&self.results_buffer), 0);
            enc.set_buffer(5, Some(&self.result_count_buffer), 0);
            enc.set_buffer(6, Some(&self.tokenize_result_buffer), 0); // GPU tokenization word count
            let tpg = 256;
            let tg = (self.current_path_count + tpg - 1) / tpg;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(tpg as u64, 1, 1));
            enc.end_encoding();
        }

        // =================================================================
        // Pass 2: GPU Sort
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.sort_pipeline);
            enc.set_buffer(0, Some(&self.results_buffer), 0);
            enc.set_buffer(1, Some(&self.result_count_buffer), 0);
            enc.set_buffer(2, Some(&self.max_sort_buffer), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
        }

        // =================================================================
        // Pass 3: GPU Text Generation
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.text_gen_pipeline);
            enc.set_buffer(0, Some(&self.paths_buffer), 0);
            enc.set_buffer(1, Some(&self.path_lengths_buffer), 0);
            enc.set_buffer(2, Some(&self.results_buffer), 0);
            enc.set_buffer(3, Some(&self.result_count_buffer), 0);
            enc.set_buffer(4, Some(&self.text_chars_buffer), 0);
            enc.set_buffer(5, Some(&self.char_count_buffer), 0);
            enc.set_buffer(6, Some(&self.text_params_buffer), 0);  // start_y (f32)
            enc.set_buffer(7, Some(&self.text_params_buffer), 4);  // line_height (f32)
            enc.set_buffer(8, Some(&self.max_sort_buffer), 0);     // max_display (u32)
            let dispatch_count = max_results.min(100) as u64;
            enc.dispatch_thread_groups(MTLSize::new((dispatch_count + 255) / 256, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe { *(self.char_count_buffer.contents() as *const u32) as usize }
    }

    /// Get text chars buffer for direct GPU rendering
    pub fn text_chars_buffer(&self) -> &Buffer {
        &self.text_chars_buffer
    }

    /// Get char count
    pub fn char_count(&self) -> usize {
        unsafe { *(self.char_count_buffer.contents() as *const u32) as usize }
    }

    /// Legacy search API (for compatibility)
    pub fn search(&self, query: &str, max_results: usize) -> Vec<(usize, i32)> {
        if query.is_empty() || self.current_path_count == 0 {
            return vec![];
        }
        let _ = self.search_and_render(query, 0.0, 20.0, max_results);

        // Read compact results (just max_results items, not all paths)
        let mut results = Vec::new();
        unsafe {
            let count = (*(self.result_count_buffer.contents() as *const u32) as usize).min(max_results);
            let ptr = self.results_buffer.contents() as *const SearchResult;
            for i in 0..count {
                let r = *ptr.add(i);
                results.push((r.path_index as usize, r.score));
            }
        }
        results
    }

    /// Get path by index
    pub fn get_path(&self, index: usize) -> Option<&str> {
        self.paths.get(index).map(|s| s.as_str())
    }

    /// Get total path count
    pub fn path_count(&self) -> usize {
        self.current_path_count
    }

    // ========================================================================
    // Async Search API (Issue #76)
    // ========================================================================

    /// Perform async GPU fuzzy search - returns immediately with a handle.
    ///
    /// The GPU processes the search in the background. Use the returned
    /// `SearchHandle` to check completion and retrieve results without blocking.
    ///
    /// # Example
    /// ```ignore
    /// let handle = search.search_async("query", 100);
    ///
    /// // Do other work while GPU processes...
    ///
    /// // Non-blocking check
    /// if let Some(results) = handle.try_get_results() {
    ///     // Process results
    /// }
    ///
    /// // Or block until done
    /// let results = handle.wait_and_get_results();
    /// ```
    pub fn search_async(&self, query: &str, max_results: usize) -> SearchHandle {
        let signal_value = self.next_signal_value.fetch_add(1, Ordering::SeqCst);

        // Handle empty/invalid queries
        if query.is_empty() || self.current_path_count == 0 {
            // Return a handle that's immediately complete with no results
            unsafe {
                *(self.result_count_buffer.contents() as *mut u32) = 0;
            }
            // Set signal to complete
            self.shared_event.set_signaled_value(signal_value);

            return SearchHandle {
                shared_event: self.shared_event.clone(),
                signal_value,
                results_buffer: self.results_buffer.clone(),
                result_count_buffer: self.result_count_buffer.clone(),
                max_results,
            };
        }

        let words: Vec<&str> = query.split_whitespace().collect();
        if words.is_empty() || words.len() > GPU_MAX_QUERY_WORDS {
            unsafe {
                *(self.result_count_buffer.contents() as *mut u32) = 0;
            }
            self.shared_event.set_signaled_value(signal_value);

            return SearchHandle {
                shared_event: self.shared_event.clone(),
                signal_value,
                results_buffer: self.results_buffer.clone(),
                result_count_buffer: self.result_count_buffer.clone(),
                max_results,
            };
        }

        let max_results = GPU_MAX_RESULTS.min(max_results);

        // Reset counters
        unsafe {
            *(self.result_count_buffer.contents() as *mut u32) = 0;
        }

        // Write params
        unsafe {
            let params = self.params_buffer.contents() as *mut SearchParams;
            *params = SearchParams {
                path_count: self.current_path_count as u32,
                word_count: words.len() as u32,
                max_results: max_results as u32,
                _padding: 0,
            };

            let words_ptr = self.words_buffer.contents() as *mut SearchWord;
            for (i, word) in words.iter().enumerate() {
                let lower = word.to_lowercase();
                let bytes = lower.as_bytes();
                let len = bytes.len().min(GPU_MAX_WORD_LEN - 1);
                let mut sw = SearchWord { chars: [0; 32], len: len as u16, _padding: 0 };
                sw.chars[..len].copy_from_slice(&bytes[..len]);
                *words_ptr.add(i) = sw;
            }

            // max_sort as u32
            let ms = self.max_sort_buffer.contents() as *mut u32;
            *ms = max_results.min(100) as u32;
        }

        // For CPU-tokenized search, reset tokenize_result.word_count to 0
        // so the kernel uses params.word_count instead
        unsafe {
            let tokenize_result = self.tokenize_result_buffer.contents() as *mut TokenizeResult;
            (*tokenize_result).word_count = 0;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Search
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.search_pipeline);
            enc.set_buffer(0, Some(&self.paths_buffer), 0);
            enc.set_buffer(1, Some(&self.path_lengths_buffer), 0);
            enc.set_buffer(2, Some(&self.params_buffer), 0);
            enc.set_buffer(3, Some(&self.words_buffer), 0);
            enc.set_buffer(4, Some(&self.results_buffer), 0);
            enc.set_buffer(5, Some(&self.result_count_buffer), 0);
            enc.set_buffer(6, Some(&self.tokenize_result_buffer), 0); // word_count=0 for CPU tokenization
            let tpg = 256;
            let tg = (self.current_path_count + tpg - 1) / tpg;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(tpg as u64, 1, 1));
            enc.end_encoding();
        }

        // Pass 2: Sort
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.sort_pipeline);
            enc.set_buffer(0, Some(&self.results_buffer), 0);
            enc.set_buffer(1, Some(&self.result_count_buffer), 0);
            enc.set_buffer(2, Some(&self.max_sort_buffer), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
        }

        // Signal SharedEvent when complete (Issue #76 - async notification)
        command_buffer.encode_signal_event(&self.shared_event, signal_value);

        // Commit without waiting - returns immediately!
        command_buffer.commit();

        SearchHandle {
            shared_event: self.shared_event.clone(),
            signal_value,
            results_buffer: self.results_buffer.clone(),
            result_count_buffer: self.result_count_buffer.clone(),
            max_results,
        }
    }

    /// Get the current signal value (for debugging/tracking)
    pub fn current_signal_value(&self) -> u64 {
        self.shared_event.signaled_value()
    }

    /// Check if a specific search is complete
    pub fn is_search_complete(&self, signal_value: u64) -> bool {
        self.shared_event.signaled_value() >= signal_value
    }

    // ========================================================================
    // GPU-Native Search API (Issue #79)
    // ========================================================================
    //
    // THE GPU IS THE COMPUTER. CPU does ONE memcpy, GPU does EVERYTHING ELSE:
    // - Query tokenization (split whitespace, lowercase)
    // - Fuzzy search
    // - Result sorting
    // - Text generation
    //
    // CPU involvement: ZERO string processing.

    /// Perform a fully GPU-native search.
    ///
    /// CPU work: ONE memcpy of raw query bytes. That's it.
    /// GPU work: Tokenization, search, sort, text generation.
    ///
    /// Returns a SearchHandle for async result retrieval.
    ///
    /// # Example
    /// ```ignore
    /// // CPU just passes raw bytes - NO tokenization, NO lowercasing
    /// let handle = search.search_gpu_native("Foo BAR baz", 100);
    ///
    /// // GPU does everything:
    /// // 1. Tokenize: "Foo BAR baz" → ["foo", "bar", "baz"]
    /// // 2. Search: fuzzy match against 3M paths
    /// // 3. Sort: order by score
    /// // 4. Return: ready for rendering
    ///
    /// let results = handle.wait_and_get_results();
    /// ```
    pub fn search_gpu_native(&self, query: &str, max_results: usize) -> SearchHandle {
        let signal_value = self.next_signal_value.fetch_add(1, Ordering::SeqCst);

        // Handle empty/invalid queries
        if query.is_empty() || self.current_path_count == 0 {
            unsafe {
                *(self.result_count_buffer.contents() as *mut u32) = 0;
            }
            self.shared_event.set_signaled_value(signal_value);
            return SearchHandle {
                shared_event: self.shared_event.clone(),
                signal_value,
                results_buffer: self.results_buffer.clone(),
                result_count_buffer: self.result_count_buffer.clone(),
                max_results,
            };
        }

        let bytes = query.as_bytes();
        let query_len = bytes.len().min(GPU_MAX_QUERY_LEN);
        let max_results = max_results.min(GPU_MAX_RESULTS);

        // =================================================================
        // THE ONLY CPU WORK: Copy raw query bytes to GPU
        // =================================================================
        unsafe {
            // Copy raw query (NO tokenization, NO lowercasing - GPU will do it)
            let query_ptr = self.raw_query_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), query_ptr, query_len);
            // Zero the rest
            std::ptr::write_bytes(query_ptr.add(query_len), 0, GPU_MAX_QUERY_LEN - query_len);

            // Reset tokenize result counter
            let tokenize_result = self.tokenize_result_buffer.contents() as *mut TokenizeResult;
            (*tokenize_result).word_count = 0;

            // Reset search result counter
            *(self.result_count_buffer.contents() as *mut u32) = 0;

            // Set max_sort
            let ms = self.max_sort_buffer.contents() as *mut u32;
            *ms = max_results.min(100) as u32;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // =================================================================
        // Pass 0: GPU Tokenization (Issue #79)
        // =================================================================
        // Input: "Foo BAR  baz" (raw bytes)
        // Output: SearchWord[0] = "foo", SearchWord[1] = "bar", SearchWord[2] = "baz"
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.tokenize_pipeline);
            enc.set_buffer(0, Some(&self.raw_query_buffer), 0);
            enc.set_buffer(1, Some(&self.words_buffer), 0);
            enc.set_buffer(2, Some(&self.tokenize_result_buffer), 0);
            enc.set_bytes(3, mem::size_of::<u32>() as u64, &(query_len as u32) as *const _ as *const _);

            // One thread per character
            let threads = query_len as u64;
            let threadgroup_size = threads.min(256);
            enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threadgroup_size, 1, 1));
            enc.end_encoding();
        }

        // Write params AFTER tokenization completes
        // Note: We need to know word_count for SearchParams, but tokenization runs on GPU.
        // Solution: SearchParams.word_count is set to MAX_QUERY_WORDS (8), and the search
        // kernel checks TokenizeResult.word_count for the actual count.
        //
        // Actually, let's use a simpler approach: we encode params with word_count=8,
        // and the search kernel uses min(params.word_count, actual_tokenized_count).
        //
        // Even simpler: pass tokenize_result_buffer to search kernel!
        unsafe {
            let params = self.params_buffer.contents() as *mut SearchParams;
            *params = SearchParams {
                path_count: self.current_path_count as u32,
                word_count: GPU_MAX_QUERY_WORDS as u32, // GPU will use actual count from TokenizeResult
                max_results: max_results as u32,
                _padding: 0,
            };
        }

        // =================================================================
        // Pass 1: GPU Search (uses GPU-tokenized word count)
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.search_pipeline);
            enc.set_buffer(0, Some(&self.paths_buffer), 0);
            enc.set_buffer(1, Some(&self.path_lengths_buffer), 0);
            enc.set_buffer(2, Some(&self.params_buffer), 0);
            enc.set_buffer(3, Some(&self.words_buffer), 0);
            enc.set_buffer(4, Some(&self.results_buffer), 0);
            enc.set_buffer(5, Some(&self.result_count_buffer), 0);
            enc.set_buffer(6, Some(&self.tokenize_result_buffer), 0); // GPU tokenization word count

            let tpg = 256;
            let tg = (self.current_path_count + tpg - 1) / tpg;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(tpg as u64, 1, 1));
            enc.end_encoding();
        }

        // =================================================================
        // Pass 2: GPU Sort
        // =================================================================
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.sort_pipeline);
            enc.set_buffer(0, Some(&self.results_buffer), 0);
            enc.set_buffer(1, Some(&self.result_count_buffer), 0);
            enc.set_buffer(2, Some(&self.max_sort_buffer), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
        }

        // Signal completion
        command_buffer.encode_signal_event(&self.shared_event, signal_value);
        command_buffer.commit();

        SearchHandle {
            shared_event: self.shared_event.clone(),
            signal_value,
            results_buffer: self.results_buffer.clone(),
            result_count_buffer: self.result_count_buffer.clone(),
            max_results,
        }
    }

    /// Synchronous GPU-native search (blocking).
    ///
    /// CPU work: ONE memcpy. GPU does all string processing.
    pub fn search_gpu_native_blocking(&self, query: &str, max_results: usize) -> Vec<(usize, i32)> {
        let handle = self.search_gpu_native(query, max_results);
        handle.wait_and_get_results()
    }

    // ========================================================================
    // Content Search Integration (GPU ripgrep - 3.5x faster than ripgrep)
    // ========================================================================

    /// Search inside file contents using GPU parallel search.
    ///
    /// This integrates the GPU ripgrep functionality into the filesystem.
    /// Uses MTLIOCommandQueue for GPU-direct I/O when available.
    ///
    /// # Arguments
    /// * `pattern` - The text pattern to search for
    /// * `path_filter` - Optional path filter (e.g., "*.rs" for Rust files only)
    /// * `options` - Search options (case sensitivity, max results)
    ///
    /// # Returns
    /// Vector of ContentMatch containing file path, line number, and context
    ///
    /// # Example
    /// ```ignore
    /// let matches = filesystem.search_content("TODO", Some("*.rs"), &SearchOptions::default());
    /// for m in matches {
    ///     println!("{}:{}: {}", m.file_path, m.line_number, m.context);
    /// }
    /// ```
    pub fn search_content(
        &self,
        pattern: &str,
        path_filter: Option<&str>,
        options: &SearchOptions,
    ) -> Vec<ContentMatch> {
        if pattern.is_empty() || self.current_path_count == 0 {
            return vec![];
        }

        // Step 1: Get matching file paths from the index
        let file_paths: Vec<std::path::PathBuf> = if let Some(filter) = path_filter {
            // Use path search to filter files
            let path_results = self.search(filter, self.current_path_count);
            path_results.iter()
                .filter_map(|(idx, _)| self.get_path(*idx))
                .filter(|p| {
                    // Only include regular files, not directories
                    std::path::Path::new(p).is_file()
                })
                .map(|p| std::path::PathBuf::from(p))
                .collect()
        } else {
            // Search all indexed files
            (0..self.current_path_count)
                .filter_map(|i| self.get_path(i))
                .filter(|p| std::path::Path::new(p).is_file())
                .map(|p| std::path::PathBuf::from(p))
                .collect()
        };

        if file_paths.is_empty() {
            return vec![];
        }

        // Step 2: Load files using GPU-direct I/O (MTLIOCommandQueue)
        let batch_loader = GpuBatchLoader::new(&self.device);

        // Step 3: Create content search engine
        let mut searcher = match GpuContentSearch::new(&self.device, file_paths.len()) {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        // Step 4: Load files and search
        if let Some(loader) = batch_loader {
            // GPU-direct I/O path (fastest - bypasses CPU entirely)
            if let Some(batch_result) = loader.load_batch(&file_paths) {
                if searcher.load_from_batch(&batch_result).is_ok() {
                    return searcher.search(pattern, options);
                }
            }
        }

        // Fallback: Use standard file loading
        let path_refs: Vec<&std::path::Path> = file_paths.iter()
            .map(|p| p.as_path())
            .collect();

        if searcher.load_files(&path_refs).is_ok() {
            return searcher.search(pattern, options);
        }

        vec![]
    }

    /// Search content with default options (case-insensitive, 1000 max results)
    pub fn search_content_simple(&self, pattern: &str, extension_filter: Option<&str>) -> Vec<ContentMatch> {
        self.search_content(pattern, extension_filter, &SearchOptions::default())
    }

    /// Combined path + content search
    ///
    /// First filters files by path pattern, then searches content.
    /// This is the most efficient way to search for content in specific file types.
    ///
    /// # Example
    /// ```ignore
    /// // Search for "async" in all Rust files
    /// let matches = filesystem.search_combined("*.rs", "async", 100);
    /// ```
    pub fn search_combined(
        &self,
        path_pattern: &str,
        content_pattern: &str,
        max_results: usize,
    ) -> Vec<ContentMatch> {
        let options = SearchOptions {
            case_sensitive: false,
            max_results,
        };
        self.search_content(content_pattern, Some(path_pattern), &options)
    }
}

// ============================================================================
// GpuStreamingSearch - Memory-efficient streaming filesystem search
// ============================================================================
//
// Instead of loading all paths into GPU memory (which can exceed 775MB for 3M paths),
// this streams paths in chunks of 50K at a time, processing each chunk on GPU
// and merging results incrementally.
//
// Memory usage: ~24MB fixed (vs 775MB+ for full index)
// Supports: Unlimited filesystem size

/// Streaming search constants
pub const STREAM_CHUNK_SIZE: usize = 50_000;      // Paths per chunk
pub const STREAM_MAX_PATH_LEN: usize = 256;       // Max path length
pub const STREAM_CHUNK_BUFFER_SIZE: usize = STREAM_CHUNK_SIZE * STREAM_MAX_PATH_LEN; // ~12MB
pub const STREAM_MAX_RESULTS: usize = 1000;       // Max results to keep

/// Search result from streaming search
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StreamSearchResult {
    pub path_index: u32,    // Global index (chunk_offset + local_index)
    pub score: i32,         // Match score
}

/// Parameters for streaming search kernel
#[repr(C)]
#[derive(Copy, Clone)]
struct StreamSearchParams {
    chunk_size: u32,        // Number of paths in this chunk
    chunk_offset: u32,      // Global offset for path indices
    word_count: u32,        // Number of query words
    min_score: i32,         // Minimum score threshold
}

/// GPU-accelerated streaming filesystem search
///
/// Uses chunked processing to search unlimited filesystems with fixed memory.
pub struct GpuStreamingSearch {
    device: Device,
    command_queue: CommandQueue,
    search_pipeline: ComputePipelineState,
    sort_pipeline: ComputePipelineState,

    // Double-buffered chunk storage (for potential async loading)
    chunk_buffer_a: Buffer,      // ~12MB - paths for current chunk
    chunk_buffer_b: Buffer,      // ~12MB - paths for next chunk (async load)
    chunk_lengths_a: Buffer,     // Path lengths for chunk A
    chunk_lengths_b: Buffer,     // Path lengths for chunk B
    active_buffer: bool,         // Toggle between A/B

    // Query buffers
    params_buffer: Buffer,
    words_buffer: Buffer,

    // Chunk results (temporary per-chunk)
    chunk_results_buffer: Buffer,
    chunk_result_count: Buffer,

    // Accumulated global results (best across all chunks)
    global_results_buffer: Buffer,
    global_result_count: Buffer,

    // Statistics
    total_paths_searched: usize,
    chunks_processed: usize,
}

impl GpuStreamingSearch {
    /// Create a new streaming search instance
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "GpuStreamingSearch");
        let command_queue = device.new_command_queue();

        // Compile shaders (reuse existing fuzzy_search_kernel)
        let library = builder.compile_library(&get_filesystem_shader())?;
        let search_pipeline = builder.create_compute_pipeline(&library, "fuzzy_search_kernel")?;
        let sort_pipeline = builder.create_compute_pipeline(&library, "sort_results_kernel")?;

        // Double-buffered chunk storage (~24MB total)
        let chunk_buffer_a = builder.create_buffer(STREAM_CHUNK_BUFFER_SIZE);
        let chunk_buffer_b = builder.create_buffer(STREAM_CHUNK_BUFFER_SIZE);
        let chunk_lengths_a = builder.create_buffer(STREAM_CHUNK_SIZE * mem::size_of::<u16>());
        let chunk_lengths_b = builder.create_buffer(STREAM_CHUNK_SIZE * mem::size_of::<u16>());

        // Query buffers
        let params_buffer = builder.create_buffer(mem::size_of::<SearchParams>());
        let words_buffer = builder.create_buffer(GPU_MAX_QUERY_WORDS * mem::size_of::<SearchWord>());

        // Results buffers
        let chunk_results_buffer = builder.create_buffer(STREAM_MAX_RESULTS * mem::size_of::<StreamSearchResult>());
        let chunk_result_count = builder.create_buffer(mem::size_of::<u32>());
        let global_results_buffer = builder.create_buffer(STREAM_MAX_RESULTS * mem::size_of::<StreamSearchResult>());
        let global_result_count = builder.create_buffer(mem::size_of::<u32>());

        Ok(Self {
            device: device.clone(),
            command_queue,
            search_pipeline,
            sort_pipeline,
            chunk_buffer_a,
            chunk_buffer_b,
            chunk_lengths_a,
            chunk_lengths_b,
            active_buffer: false,
            params_buffer,
            words_buffer,
            chunk_results_buffer,
            chunk_result_count,
            global_results_buffer,
            global_result_count,
            total_paths_searched: 0,
            chunks_processed: 0,
        })
    }

    /// Upload a chunk of paths to GPU memory
    fn upload_chunk(&self, paths: &[String], use_buffer_a: bool) {
        let (path_buffer, len_buffer) = if use_buffer_a {
            (&self.chunk_buffer_a, &self.chunk_lengths_a)
        } else {
            (&self.chunk_buffer_b, &self.chunk_lengths_b)
        };

        unsafe {
            let path_ptr = path_buffer.contents() as *mut u8;
            let len_ptr = len_buffer.contents() as *mut u16;

            for (i, path) in paths.iter().enumerate() {
                let bytes = path.as_bytes();
                let len = bytes.len().min(STREAM_MAX_PATH_LEN - 1);

                // Copy path
                let dest = path_ptr.add(i * STREAM_MAX_PATH_LEN);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                *dest.add(len) = 0; // Null-terminate

                // Store length
                *len_ptr.add(i) = len as u16;
            }
        }
    }

    /// Search a single chunk and accumulate results
    fn search_chunk(&self, chunk_size: usize, _chunk_offset: usize, query_words: &[(String, u16)], use_buffer_a: bool) {
        let (path_buffer, len_buffer) = if use_buffer_a {
            (&self.chunk_buffer_a, &self.chunk_lengths_a)
        } else {
            (&self.chunk_buffer_b, &self.chunk_lengths_b)
        };

        // Reset chunk result count
        unsafe {
            *(self.chunk_result_count.contents() as *mut u32) = 0;
        }

        // Set up search params (reuse SearchParams structure)
        unsafe {
            let params = self.params_buffer.contents() as *mut SearchParams;
            (*params).path_count = chunk_size as u32;
            (*params).word_count = query_words.len() as u32;
            (*params).max_results = STREAM_MAX_RESULTS as u32;  // CRITICAL: was missing!
            (*params)._padding = 0;
        }

        // Upload query words
        // Issue #268 fix: Add bounds checking for word length and count
        unsafe {
            let word_ptr = self.words_buffer.contents() as *mut SearchWord;
            for (i, (word, len)) in query_words.iter().take(GPU_MAX_QUERY_WORDS).enumerate() {
                let w = word_ptr.add(i);
                let bytes = word.as_bytes();
                // Clamp length to SearchWord.chars size (32 bytes)
                let safe_len = (*len as usize).min(32).min(bytes.len());
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), (*w).chars.as_mut_ptr() as *mut u8, safe_len);
                // Zero the rest of chars
                if safe_len < 32 {
                    std::ptr::write_bytes((*w).chars.as_mut_ptr().add(safe_len) as *mut u8, 0, 32 - safe_len);
                }
                (*w).len = safe_len as u16;
            }
        }

        // Execute GPU search
        let command_buffer = self.command_queue.new_command_buffer();
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.search_pipeline);
            encoder.set_buffer(0, Some(path_buffer), 0);
            encoder.set_buffer(1, Some(len_buffer), 0);
            encoder.set_buffer(2, Some(&self.params_buffer), 0);
            encoder.set_buffer(3, Some(&self.words_buffer), 0);
            encoder.set_buffer(4, Some(&self.chunk_results_buffer), 0);
            encoder.set_buffer(5, Some(&self.chunk_result_count), 0);

            let threadgroups = ((chunk_size as u64 + 1023) / 1024).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(1024, 1, 1),
            );
            encoder.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Merge chunk results into global results, keeping top N by score
    fn merge_results(&self, chunk_offset: usize) {
        unsafe {
            let chunk_count = *(self.chunk_result_count.contents() as *const u32) as usize;
            let global_count_ptr = self.global_result_count.contents() as *mut u32;
            let mut global_count = *global_count_ptr as usize;

            let chunk_ptr = self.chunk_results_buffer.contents() as *const StreamSearchResult;
            let global_ptr = self.global_results_buffer.contents() as *mut StreamSearchResult;

            // Add chunk results to global (adjusting indices by chunk_offset)
            for i in 0..chunk_count.min(STREAM_MAX_RESULTS) {
                let mut result = *chunk_ptr.add(i);
                result.path_index += chunk_offset as u32;  // Adjust to global index

                if global_count < STREAM_MAX_RESULTS {
                    *global_ptr.add(global_count) = result;
                    global_count += 1;
                } else {
                    // Find minimum score and replace if this is better
                    let mut min_idx = 0;
                    let mut min_score = (*global_ptr.add(0)).score;
                    for j in 1..STREAM_MAX_RESULTS {
                        let s = (*global_ptr.add(j)).score;
                        if s < min_score {
                            min_score = s;
                            min_idx = j;
                        }
                    }
                    if result.score > min_score {
                        *global_ptr.add(min_idx) = result;
                    }
                }
            }

            *global_count_ptr = global_count as u32;
        }
    }

    /// Sort global results by score (descending)
    fn sort_results(&self) {
        let count = unsafe { *(self.global_result_count.contents() as *const u32) as usize };
        if count <= 1 {
            return;
        }

        // Simple CPU insertion sort for now (results are small)
        unsafe {
            let ptr = self.global_results_buffer.contents() as *mut StreamSearchResult;
            for i in 1..count {
                let key = *ptr.add(i);
                let mut j = i;
                while j > 0 && (*ptr.add(j - 1)).score < key.score {
                    *ptr.add(j) = *ptr.add(j - 1);
                    j -= 1;
                }
                *ptr.add(j) = key;
            }
        }
    }

    /// Reset search state
    pub fn reset(&mut self) {
        unsafe {
            *(self.global_result_count.contents() as *mut u32) = 0;
        }
        self.total_paths_searched = 0;
        self.chunks_processed = 0;
    }

    /// Process a chunk of paths (call repeatedly with chunks from PathIterator)
    pub fn process_chunk(&mut self, paths: &[String], chunk_offset: usize, query_words: &[(String, u16)]) {
        if paths.is_empty() || query_words.is_empty() {
            return;
        }

        // Upload paths
        self.upload_chunk(paths, self.active_buffer);

        // Search
        self.search_chunk(paths.len(), chunk_offset, query_words, self.active_buffer);

        // Merge results
        self.merge_results(chunk_offset);

        // Update stats
        self.total_paths_searched += paths.len();
        self.chunks_processed += 1;

        // Toggle buffer for next chunk
        self.active_buffer = !self.active_buffer;
    }

    /// Get final sorted results
    pub fn get_results(&self) -> Vec<StreamSearchResult> {
        let count = unsafe { *(self.global_result_count.contents() as *const u32) as usize };
        let mut results = Vec::with_capacity(count);

        unsafe {
            let ptr = self.global_results_buffer.contents() as *const StreamSearchResult;
            for i in 0..count {
                results.push(*ptr.add(i));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.cmp(&a.score));
        results
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.total_paths_searched, self.chunks_processed)
    }
}

/// Disk-backed path iterator for streaming search
///
/// Reads paths from the filesystem index file in chunks, avoiding loading
/// all paths into memory at once.
pub struct PathIterator {
    index_path: std::path::PathBuf,
    reader: Option<std::io::BufReader<std::fs::File>>,
    current_chunk: Vec<String>,
    chunk_start: usize,
    total_paths: usize,
    paths_read: usize,
}

impl PathIterator {
    /// Create iterator from index file
    pub fn new(index_path: &std::path::Path) -> Result<Self, String> {
        use std::io::BufRead;

        let file = std::fs::File::open(index_path)
            .map_err(|e| format!("Failed to open index: {}", e))?;
        let mut reader = std::io::BufReader::new(file);

        // Read header to get total count
        let mut header = String::new();
        reader.read_line(&mut header).ok();
        let total_paths = header.trim()
            .split_whitespace()
            .last()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(Self {
            index_path: index_path.to_path_buf(),
            reader: Some(reader),
            current_chunk: Vec::with_capacity(STREAM_CHUNK_SIZE),
            chunk_start: 0,
            total_paths,
            paths_read: 0,
        })
    }

    /// Get next chunk of paths with its starting offset
    /// Returns (chunk_paths, chunk_start_offset)
    pub fn next_chunk_with_offset(&mut self) -> Option<(Vec<String>, usize)> {
        use std::io::BufRead;

        let reader = self.reader.as_mut()?;
        self.current_chunk.clear();
        let chunk_start = self.paths_read;

        let mut line = String::new();
        while self.current_chunk.len() < STREAM_CHUNK_SIZE {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // Parse line: "id path" format
                    if let Some(path) = line.trim().split_once(' ').map(|(_, p)| p.to_string()) {
                        self.current_chunk.push(path);
                        self.paths_read += 1;
                    }
                }
                Err(_) => break,
            }
        }

        if self.current_chunk.is_empty() {
            None
        } else {
            Some((self.current_chunk.clone(), chunk_start))
        }
    }

    /// Get next chunk of paths (legacy API, use next_chunk_with_offset for streaming search)
    pub fn next_chunk(&mut self) -> Option<&[String]> {
        use std::io::BufRead;

        let reader = self.reader.as_mut()?;
        self.current_chunk.clear();
        self.chunk_start = self.paths_read;

        let mut line = String::new();
        while self.current_chunk.len() < STREAM_CHUNK_SIZE {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // Parse line: "id path" format
                    if let Some(path) = line.trim().split_once(' ').map(|(_, p)| p.to_string()) {
                        self.current_chunk.push(path);
                        self.paths_read += 1;
                    }
                }
                Err(_) => break,
            }
        }

        if self.current_chunk.is_empty() {
            None
        } else {
            Some(&self.current_chunk)
        }
    }

    /// Reset to beginning
    pub fn reset(&mut self) -> Result<(), String> {
        use std::io::BufRead;

        let file = std::fs::File::open(&self.index_path)
            .map_err(|e| format!("Failed to reopen index: {}", e))?;
        let mut reader = std::io::BufReader::new(file);

        // Skip header
        let mut header = String::new();
        reader.read_line(&mut header).ok();

        self.reader = Some(reader);
        self.current_chunk.clear();
        self.chunk_start = 0;
        self.paths_read = 0;
        Ok(())
    }

    /// Get total path count (from header)
    pub fn total_paths(&self) -> usize {
        self.total_paths
    }

    /// Get current position
    pub fn paths_read(&self) -> usize {
        self.paths_read
    }

    /// Get chunk start offset (for result index adjustment)
    pub fn chunk_start(&self) -> usize {
        self.chunk_start
    }
}

/// Parse query into lowercase words
pub fn parse_query_words(query: &str) -> Vec<(String, u16)> {
    query
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .take(GPU_MAX_QUERY_WORDS)
        .map(|w| {
            let lower = w.to_lowercase();
            let len = lower.len().min(GPU_MAX_WORD_LEN) as u16;
            (lower, len)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_sizes() {
        assert_eq!(mem::size_of::<InodeCompact>(), 64);
        assert_eq!(mem::size_of::<DirEntryCompact>(), 32);
        assert_eq!(mem::size_of::<BlockMapEntry>(), 8);
    }

    #[test]
    fn test_inode_flags() {
        let mut inode = InodeCompact::new(1, 0, FileType::Regular);
        assert_eq!(inode.get_file_type(), FileType::Regular);

        inode.set_compressed(true);
        assert!(inode.is_compressed());
        assert_eq!(inode.get_file_type(), FileType::Regular); // Unchanged
    }

    #[test]
    #[ignore]  // Requires Metal device
    fn test_path_lookup_root() {
        let device = Device::system_default().unwrap();
        let fs = GpuFilesystem::new(&device, 128).unwrap();

        // Root lookup
        assert_eq!(fs.lookup_path("/").unwrap(), ROOT_INODE_ID);
    }

    #[test]
    #[ignore]  // Requires Metal device
    fn test_path_lookup_simple() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 128).unwrap();

        // Create: /src
        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();

        // Lookup absolute
        assert_eq!(fs.lookup_path("/src").unwrap(), src_id);
    }

    #[test]
    #[ignore]  // Requires Metal device
    fn test_path_lookup_nested() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 128).unwrap();

        // Create: /src/main.rs
        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();
        let main_id = fs.add_file(src_id, "main.rs", FileType::Regular).unwrap();

        // Lookup nested path
        assert_eq!(fs.lookup_path("/src/main.rs").unwrap(), main_id);
    }

    #[test]
    #[ignore]  // Requires Metal device
    fn test_path_lookup_deep() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 128).unwrap();

        // Create: /foo/bar/baz/file.txt
        let foo_id = fs.add_file(ROOT_INODE_ID, "foo", FileType::Directory).unwrap();
        let bar_id = fs.add_file(foo_id, "bar", FileType::Directory).unwrap();
        let baz_id = fs.add_file(bar_id, "baz", FileType::Directory).unwrap();
        let file_id = fs.add_file(baz_id, "file.txt", FileType::Regular).unwrap();

        // Lookup deep path
        assert_eq!(fs.lookup_path("/foo/bar/baz/file.txt").unwrap(), file_id);
    }

    #[test]
    #[ignore]  // Requires Metal device
    fn test_path_lookup_not_found() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 128).unwrap();

        fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();

        // Non-existent path
        assert!(fs.lookup_path("/nonexistent").is_err());
        assert!(fs.lookup_path("/src/missing").is_err());
    }

    // ========================================================================
    // Batch Lookup Tests (Issue #26)
    // ========================================================================

    #[test]
    #[ignore]
    fn test_batch_lookup_empty() {
        let device = Device::system_default().unwrap();
        let fs = GpuFilesystem::new(&device, 1024).unwrap();

        let results = fs.lookup_batch(&[]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_single() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();

        let results = fs.lookup_batch(&["/src"]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().unwrap(), &src_id);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_multiple() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();
        let docs_id = fs.add_file(ROOT_INODE_ID, "docs", FileType::Directory).unwrap();
        let tests_id = fs.add_file(ROOT_INODE_ID, "tests", FileType::Directory).unwrap();

        let paths = vec!["/src", "/docs", "/tests"];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap(), &src_id);
        assert_eq!(results[1].as_ref().unwrap(), &docs_id);
        assert_eq!(results[2].as_ref().unwrap(), &tests_id);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_large() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 2048).unwrap();

        // Create 100 files
        let mut expected_ids = Vec::new();
        let mut paths = Vec::new();

        for i in 0..100 {
            let name = format!("file{:03}", i);
            let id = fs.add_file(ROOT_INODE_ID, &name, FileType::Regular).unwrap();
            expected_ids.push(id);
            paths.push(format!("/file{:03}", i));
        }

        // Batch lookup all 100
        let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let results = fs.lookup_batch(&path_refs).unwrap();

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.as_ref().unwrap(), &expected_ids[i],
                "Path {} should resolve to inode {}", paths[i], expected_ids[i]);
        }
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_mixed_success_fail() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();

        let paths = vec!["/src", "/missing", "/src", "/also_missing"];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].as_ref().unwrap(), &src_id);
        assert!(results[1].is_err());
        assert_eq!(results[2].as_ref().unwrap(), &src_id);
        assert!(results[3].is_err());
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_nested_paths() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).unwrap();
        let gpu_os_id = fs.add_file(src_id, "gpu_os", FileType::Directory).unwrap();
        let main_id = fs.add_file(src_id, "main.rs", FileType::Regular).unwrap();
        let fs_id = fs.add_file(gpu_os_id, "filesystem.rs", FileType::Regular).unwrap();

        let paths = vec![
            "/src/main.rs",
            "/src/gpu_os",
            "/src/gpu_os/filesystem.rs",
        ];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap(), &main_id);
        assert_eq!(results[1].as_ref().unwrap(), &gpu_os_id);
        assert_eq!(results[2].as_ref().unwrap(), &fs_id);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_root_paths() {
        let device = Device::system_default().unwrap();
        let fs = GpuFilesystem::new(&device, 1024).unwrap();

        let paths = vec!["/", "/", "/"];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 3);
        for result in &results {
            assert_eq!(result.as_ref().unwrap(), &ROOT_INODE_ID);
        }
    }
}
