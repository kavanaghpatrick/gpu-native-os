# GPU-Native Filesystem - Product Requirements Document

**Version:** 1.0
**Date:** 2026-01-24
**Status:** Draft
**Platform:** Apple Silicon M4 (macOS Sequoia 15.4+)

---

## Executive Summary

A revolutionary filesystem implementation that runs primary operations on GPU compute units instead of CPU cores, leveraging Apple Silicon's unified memory architecture for zero-copy I/O and unprecedented metadata operation throughput.

**Key Innovation:** Treating filesystem metadata operations (path lookup, directory listing, search) as massively parallel compute problems, achieving 10-1000x speedup over traditional CPU-based filesystems.

**Target Performance:**
- Path lookup: <50μs (vs 500μs+ traditional)
- Directory listing (100K files): <2ms (vs 500ms traditional)
- Global search (1M files): <50ms (vs 30+ seconds traditional)
- Deduplication scan: Real-time during writes (vs batch jobs)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Specifications](#2-component-specifications)
3. [Implementation Phases](#3-implementation-phases)
4. [Pseudocode Specifications](#4-pseudocode-specifications)
5. [Test Specifications](#5-test-specifications)
6. [Performance Requirements](#6-performance-requirements)
7. [Success Metrics](#7-success-metrics)
8. [Risk Analysis](#8-risk-analysis)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    macOS Applications                        │
│              (Finder, Terminal, IDEs, etc.)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ POSIX syscalls (open, read, write)
┌──────────────────────────▼──────────────────────────────────┐
│                    macOS Kernel (VFS)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ FSKit protocol
┌──────────────────────────▼──────────────────────────────────┐
│               GpuFS Extension (Userspace)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ FSUnaryFileSystem Implementation                       │ │
│  │  • Path resolution                                     │ │
│  │  • Directory operations                                │ │
│  │  • File I/O coordination                               │ │
│  └────────────┬──────────────────────┬────────────────────┘ │
│               │                      │                      │
│  ┌────────────▼──────────┐  ┌────────▼─────────────┐       │
│  │  Metal Compute Engine │  │  I/O Coordinator     │       │
│  │  ┌─────────────────┐  │  │  • Async read/write  │       │
│  │  │ Metadata Kernel │  │  │  • Buffer management │       │
│  │  │ • Path lookup   │  │  │  • DMA coordination  │       │
│  │  │ • Dir listing   │  │  │                      │       │
│  │  │ • Search        │  │  └──────────┬───────────┘       │
│  │  └─────────────────┘  │             │                   │
│  │  ┌─────────────────┐  │             │                   │
│  │  │ Data Kernel     │  │             │                   │
│  │  │ • Compression   │  │             │                   │
│  │  │ • Encryption    │  │             │                   │
│  │  │ • Checksums     │  │             │                   │
│  │  │ • Deduplication │  │             │                   │
│  │  └─────────────────┘  │             │                   │
│  └───────────┬───────────┘             │                   │
│              │                         │                   │
│  ┌───────────▼─────────────────────────▼────────────────┐  │
│  │         Unified Memory (MTLStorageModeShared)        │  │
│  │  • Metadata structures (inodes, directories)         │  │
│  │  • File data buffers                                 │  │
│  │  • Hash tables, B-trees                              │  │
│  │  • Zero-copy between CPU and GPU                     │  │
│  └──────────────────────────┬───────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────┐
│              Block Device (NVMe SSD)                        │
│              • FSBlockDeviceResource                        │
│              • CPU-mediated I/O (macOS limitation)          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Architecture

Based on M4 research findings, we leverage:

- **Unified Memory:** 120-546 GB/s bandwidth (M4 base to Max)
- **Zero-copy I/O:** MTLStorageModeShared for CPU-GPU data sharing
- **Large buffers:** Up to 65-75% of physical RAM (48GB on 64GB Mac)
- **Cache coherency:** Hardware-managed, no explicit synchronization

**Memory Layout:**

```
Unified Memory Region (example: 24GB on 32GB Mac)
├── Inode Table:        1M inodes × 64B = 64 MB
├── Directory Entries:  10M entries × 32B = 320 MB
├── Hash Table:         1M buckets × 16B = 16 MB
├── B-Tree Nodes:       100K nodes × 4KB = 400 MB
├── Block Bitmap:       1TB / 4KB × 1bit = 32 MB
├── File Data Cache:    20 GB (hot blocks)
└── Working Buffers:    ~4 GB (compression, temp data)
    Total:              ~24.8 GB
```

### 1.3 Thread Architecture

Leveraging M4's 1024 threads per threadgroup:

**Metadata Operations:** All 1024 threads cooperate
- Phase 1: Load metadata from unified memory
- Phase 2: Parallel predicate evaluation
- Phase 3: Parallel sort/aggregate
- Phase 4: Write results
- Barriers between phases for synchronization

**Data Operations:** Each thread handles different blocks
- Thread 0-1023: Process blocks 0-1023 simultaneously
- Compression: 1024 blocks compressed in parallel
- Checksums: 1024 CRC32 computed in parallel
- Encryption: 1024 blocks encrypted in parallel

### 1.4 Constraint: CPU-Mediated I/O

**Critical Finding from Research:** Apple Silicon does NOT support GPU Direct Storage.

**Implication:** All disk I/O must be mediated by CPU, but unified memory eliminates copy overhead.

**I/O Flow:**
```
1. CPU thread: Async read(block_id) → MTLBuffer (shared)
2. CPU signals GPU via MTLEvent
3. GPU kernel: Processes data directly in MTLBuffer (zero-copy)
4. GPU signals CPU via MTLEvent
5. CPU thread: Async write(MTLBuffer) → disk
```

**Performance Optimization:**
- Multiple CPU I/O threads for parallelism
- Double buffering: GPU processes buffer N while CPU loads N+1
- Prefetching: Predict next blocks based on access patterns
- Batching: Coalesce multiple small I/O into large sequential operations

---

## 2. Component Specifications

### 2.1 Core Data Structures

#### 2.1.1 Inode (64 bytes)

**Purpose:** Store file/directory metadata

```rust
#[repr(C)]
pub struct InodeCompact {
    // Identity (16 bytes)
    pub inode_id: u64,          // Unique inode number
    pub parent_id: u32,         // Parent directory inode
    pub generation: u32,        // Reuse counter for deleted inodes

    // Tree structure (12 bytes) - SAME as WidgetCompact!
    pub first_child: u32,       // First child inode (directories)
    pub next_sibling: u32,      // Next sibling in directory
    pub prev_sibling: u32,      // Previous sibling (for fast delete)

    // File metadata (24 bytes)
    pub size: u64,              // File size in bytes
    pub blocks: u32,            // Number of 4KB blocks allocated
    pub block_ptr: u32,         // Offset in block map table
    pub timestamps: u64,        // Packed: created|modified|accessed
    pub uid_gid: u32,           // Packed: uid(16) | gid(16)
    pub mode: u16,              // Permissions (rwxrwxrwx)
    pub flags: u16,             // Type, compression, encryption flags

    // Data integrity (8 bytes)
    pub checksum: u32,          // CRC32 of file content
    pub refcount: u16,          // Hard link count
    pub _padding: u16,          // Alignment

    // Total: 64 bytes (cache-line aligned)
}
```

**Flags Encoding (16 bits):**
```
Bits 0-3:   File type (regular, directory, symlink, etc.)
Bit  4:     Compressed
Bit  5:     Encrypted
Bit  6:     Deduplicated
Bit  7:     Dirty (needs flush)
Bits 8-11:  Compression algo (0=none, 1=LZ4, 2=Zstd)
Bits 12-15: Reserved
```

#### 2.1.2 Directory Entry (32 bytes)

**Purpose:** Map filename to inode in a directory

```rust
#[repr(C)]
pub struct DirEntryCompact {
    pub inode_id: u32,          // Target inode
    pub name_hash: u32,         // xxHash3 of filename
    pub name_len: u16,          // Filename length
    pub file_type: u8,          // Cached from inode (regular, dir, etc.)
    pub _padding: u8,           // Alignment
    pub name: [u8; 24],         // Short filename (inline)
    // For names >24 chars, store in separate name pool

    // Total: 32 bytes
}
```

**Design rationale:**
- 75% of filenames are <24 chars (research finding)
- Inline short names = no pointer chase
- Hash for fast comparison (compare u32 before strcmp)
- File type cached for `ls -F` without inode fetch

#### 2.1.3 Block Map Entry (8 bytes)

**Purpose:** Map logical block to physical block

```rust
#[repr(C)]
pub struct BlockMapEntry {
    pub physical_block: u32,    // Physical block number on disk
    pub flags: u16,             // Sparse, compressed, encrypted, CoW
    pub refcount: u16,          // For deduplication/CoW

    // Total: 8 bytes
}
```

**Flags Encoding:**
```
Bit 0:     Sparse (all zeros, no physical block)
Bit 1:     Compressed
Bit 2:     Encrypted
Bit 3:     Deduplicated (shared block)
Bit 4:     Copy-on-Write
Bits 5-15: Reserved
```

#### 2.1.4 Hash Table (Hive Hash Table - State-of-Art)

**Purpose:** Fast filename lookup in directories

Based on research: Hive achieves 4B lookups/sec, 95% load factors.

```rust
pub struct GpuHashTable {
    pub buckets: MTLBuffer,     // Array of HashBucket
    pub capacity: u32,          // Power of 2
    pub count: AtomicU32,       // Current entries
    pub tombstones: AtomicU32,  // Deleted entries
}

#[repr(C)]
pub struct HashBucket {
    pub hash: u32,              // xxHash3 of key
    pub value: u32,             // Inode ID
    pub status: u32,            // Empty | Occupied | Tombstone
    pub _padding: u32,          // Alignment
}
```

**Operations:**
- Insert: Linear probing with atomic CAS
- Lookup: Linear probe until hash match or empty
- Delete: Mark as tombstone, increment counter
- Resize: When load > 95% or tombstones > 20%

### 2.2 GPU Kernels

#### 2.2.1 Path Lookup Kernel

**Purpose:** Resolve "/path/to/file" to inode ID

**Algorithm:** Parallel path component resolution

```metal
kernel void path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* dir_entries [[buffer(1)]],
    device HashBucket* hash_table [[buffer(2)]],
    constant char* path [[buffer(3)]],
    device u64* result_inode [[buffer(4)]],
    constant u32& inode_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
)
{
    // Phase 1: Parse path into components
    threadgroup char components[16][256];  // Max 16 path components
    threadgroup u32 component_count;

    if (tid == 0) {
        component_count = parse_path(path, components);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Walk directory tree
    u32 current_inode = ROOT_INODE_ID;  // Start at root

    for (u32 i = 0; i < component_count; i++) {
        // All threads cooperate to search directory
        u32 hash = xxhash3(components[i]);

        // Parallel hash table lookup
        u32 found_inode = hash_table_lookup_parallel(
            hash_table,
            hash,
            components[i],
            tid
        );

        if (found_inode == INVALID_INODE) {
            // Path component not found
            if (tid == 0) {
                *result_inode = INVALID_INODE;
            }
            return;
        }

        current_inode = found_inode;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes result
    if (tid == 0) {
        *result_inode = current_inode;
    }
}
```

#### 2.2.2 Directory Listing Kernel

**Purpose:** List all files in a directory

**Algorithm:** Parallel filter + radix sort (research: 2-4x faster than bitonic)

```metal
kernel void list_directory_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* all_entries [[buffer(1)]],
    device DirEntryCompact* output [[buffer(2)]],
    device u32* output_count [[buffer(3)]],
    constant u32& dir_inode_id [[buffer(4)]],
    constant u32& total_entries [[buffer(5)]],
    constant u32& sort_mode [[buffer(6)]],  // 0=name, 1=size, 2=mtime
    uint tid [[thread_index_in_threadgroup]]
)
{
    // Phase 1: Parallel filter (find children of this directory)
    threadgroup u32 local_matches[1024];
    threadgroup atomic_uint match_count;

    if (tid == 0) {
        atomic_store_explicit(&match_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread checks multiple entries
    for (u32 i = tid; i < total_entries; i += 1024) {
        // Check if this entry belongs to target directory
        u32 entry_inode = all_entries[i].inode_id;
        if (inodes[entry_inode].parent_id == dir_inode_id) {
            u32 idx = atomic_fetch_add_explicit(&match_count, 1, memory_order_relaxed);
            local_matches[idx] = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    u32 count = atomic_load_explicit(&match_count, memory_order_relaxed);

    // Phase 2: Copy matches to output
    if (tid < count) {
        output[tid] = all_entries[local_matches[tid]];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: Radix sort by name/size/time (research-proven optimal)
    if (count > 0) {
        radix_sort_dir_entries(output, count, sort_mode, tid);
    }

    // Thread 0 writes count
    if (tid == 0) {
        *output_count = count;
    }
}
```

#### 2.2.3 Global Search Kernel

**Purpose:** Find all files matching pattern

**Algorithm:** HybridSA regex matching (research: 60x faster than CPU)

```metal
kernel void global_search_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* dir_entries [[buffer(1)]],
    device u64* matches [[buffer(2)]],
    device u32* match_count [[buffer(3)]],
    constant char* pattern [[buffer(4)]],
    constant u32& pattern_len [[buffer(5)]],
    constant u32& total_entries [[buffer(6)]],
    constant u32& search_mode [[buffer(7)]],  // 0=name, 1=content
    uint tid [[thread_index_in_threadgroup]]
)
{
    threadgroup atomic_uint local_count;

    if (tid == 0) {
        atomic_store_explicit(&local_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread searches different files
    for (u32 i = tid; i < total_entries; i += 1024) {
        bool match = false;

        if (search_mode == 0) {
            // Filename search using HybridSA algorithm
            match = hybrid_sa_match(
                dir_entries[i].name,
                dir_entries[i].name_len,
                pattern,
                pattern_len
            );
        } else {
            // Content search (load file blocks, search text)
            // TODO: Implement for Phase 2
        }

        if (match) {
            u32 idx = atomic_fetch_add_explicit(&local_count, 1, memory_order_relaxed);
            matches[idx] = dir_entries[i].inode_id;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        *match_count = atomic_load_explicit(&local_count, memory_order_relaxed);
    }
}
```

#### 2.2.4 Compression Kernel

**Purpose:** Compress file blocks in parallel

**Algorithm:** LZ4 (research: 118 GB/s decompression, 36 GB/s compression)

```metal
kernel void compress_blocks_kernel(
    device u8* input_blocks [[buffer(0)]],
    device u8* output_blocks [[buffer(1)]],
    device u32* output_sizes [[buffer(2)]],
    constant u32& block_count [[buffer(3)]],
    constant u32& block_size [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    // Each thread compresses one block
    if (tid >= block_count) return;

    u8* input = &input_blocks[tid * block_size];
    u8* output = &output_blocks[tid * block_size * 2];  // Max 2x for incompressible

    u32 compressed_size = lz4_compress_block(
        input,
        output,
        block_size
    );

    output_sizes[tid] = compressed_size;
}
```

#### 2.2.5 Deduplication Kernel

**Purpose:** Find duplicate blocks via content hashing

**Algorithm:** Parallel xxHash3 + Hive hash table

```metal
kernel void dedup_scan_kernel(
    device u8* blocks [[buffer(0)]],
    device u64* block_hashes [[buffer(1)]],
    device HashBucket* hash_table [[buffer(2)]],
    device u32* dedup_map [[buffer(3)]],  // maps block_id → canonical_block_id
    constant u32& block_count [[buffer(4)]],
    constant u32& block_size [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    // Each thread hashes one block
    if (tid >= block_count) return;

    u8* block = &blocks[tid * block_size];
    u64 hash = xxhash3_64(block, block_size);
    block_hashes[tid] = hash;

    threadgroup_barrier(mem_flags::mem_device);

    // Try to insert into hash table
    u32 existing_block = hash_table_insert_or_get(
        hash_table,
        hash,
        tid
    );

    if (existing_block != tid) {
        // Duplicate found!
        // Verify content match (hash collision check)
        u8* existing = &blocks[existing_block * block_size];
        if (memcmp(block, existing, block_size) == 0) {
            dedup_map[tid] = existing_block;
        } else {
            dedup_map[tid] = tid;  // Hash collision, keep separate
        }
    } else {
        dedup_map[tid] = tid;  // First occurrence
    }
}
```

### 2.3 CPU Components

#### 2.3.1 I/O Coordinator

**Purpose:** Mediate all disk I/O (macOS constraint)

```rust
pub struct IoCoordinator {
    device: Device,
    queue: CommandQueue,
    io_queue: DispatchQueue,  // GCD queue for async I/O

    // Double buffering
    read_buffers: Vec<MTLBuffer>,
    write_buffers: Vec<MTLBuffer>,
    current_buffer: AtomicUsize,

    // Statistics
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
}

impl IoCoordinator {
    pub async fn read_blocks(&self, block_ids: &[u32]) -> Result<MTLBuffer> {
        // Get next available buffer
        let buffer_idx = self.current_buffer.fetch_add(1, Ordering::Relaxed) % 2;
        let buffer = &self.read_buffers[buffer_idx];

        // Async I/O via GCD
        let blocks_data = self.io_queue.async_work(|| {
            // Coalesce sequential blocks for performance
            let ranges = Self::coalesce_ranges(block_ids);
            let mut data = Vec::new();

            for range in ranges {
                // Read from block device
                let bytes = fs::read(&format!("/dev/disk{}", range.start))?;
                data.extend_from_slice(&bytes);
            }
            data
        }).await?;

        // Copy to MTLBuffer (zero-copy after this point)
        unsafe {
            let ptr = buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                blocks_data.as_ptr(),
                ptr,
                blocks_data.len()
            );
        }

        self.bytes_read.fetch_add(blocks_data.len() as u64, Ordering::Relaxed);
        Ok(buffer.clone())
    }

    pub async fn write_blocks(&self, buffer: &MTLBuffer, block_ids: &[u32]) -> Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(
                buffer.contents() as *const u8,
                buffer.length() as usize
            )
        };

        self.io_queue.async_work(move || {
            // Write blocks to disk
            for (i, &block_id) in block_ids.iter().enumerate() {
                let offset = i * BLOCK_SIZE;
                let block_data = &data[offset..offset + BLOCK_SIZE];

                fs::write(
                    &format!("/dev/disk{}", block_id),
                    block_data
                )?;
            }
            Ok(())
        }).await?;

        self.bytes_written.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}
```

#### 2.3.2 FSKit Integration

**Purpose:** Integrate with macOS via FSKit

```rust
pub struct GpuFilesystemExtension: FSUnaryFileSystem {
    metal_engine: MetalComputeEngine,
    io_coordinator: IoCoordinator,
    memory: GpuMemory,
}

impl FSUnaryFileSystemOperations for GpuFilesystemExtension {
    fn lookup(
        &self,
        in_dir: FSVolumeItemIdentifier,
        name: &str
    ) -> Result<FSVolumeItemAttributes> {
        // Convert to path
        let path = self.build_path(in_dir, name)?;

        // Dispatch GPU kernel
        let inode_id = self.metal_engine.path_lookup(&path)?;

        if inode_id == INVALID_INODE {
            return Err(FSError::NotFound);
        }

        // Load inode from unified memory
        let inode = self.memory.load_inode(inode_id)?;

        // Convert to FSKit attributes
        Ok(FSVolumeItemAttributes {
            item_id: inode_id as u64,
            size: inode.size,
            mode: inode.mode,
            uid: (inode.uid_gid >> 16) as u32,
            gid: (inode.uid_gid & 0xFFFF) as u32,
            // ... more attributes
        })
    }

    fn read_directory(
        &self,
        dir: FSVolumeItemIdentifier,
        offset: u64,
        limit: u32
    ) -> Result<Vec<FSDirEntry>> {
        // Dispatch GPU kernel
        let entries = self.metal_engine.list_directory(dir as u32, offset, limit)?;

        // Convert to FSKit format
        Ok(entries.iter().map(|e| FSDirEntry {
            name: String::from_utf8_lossy(&e.name).to_string(),
            item_id: e.inode_id as u64,
            file_type: e.file_type,
        }).collect())
    }

    fn read(
        &self,
        file: FSVolumeItemIdentifier,
        offset: u64,
        length: u64
    ) -> Result<Vec<u8>> {
        // Load inode
        let inode = self.memory.load_inode(file as u32)?;

        // Calculate required blocks
        let start_block = (offset / BLOCK_SIZE as u64) as u32;
        let end_block = ((offset + length) / BLOCK_SIZE as u64) as u32;
        let block_ids: Vec<u32> = (start_block..=end_block).collect();

        // CPU: Read blocks from disk into unified memory
        let buffer = self.io_coordinator.read_blocks(&block_ids).await?;

        // GPU: Decompress if needed
        if inode.is_compressed() {
            let decompressed = self.metal_engine.decompress_blocks(&buffer, &block_ids)?;
            buffer = decompressed;
        }

        // GPU: Decrypt if needed
        if inode.is_encrypted() {
            let decrypted = self.metal_engine.decrypt_blocks(&buffer, &block_ids)?;
            buffer = decrypted;
        }

        // Extract requested range
        let start_offset = (offset % BLOCK_SIZE as u64) as usize;
        let data = unsafe {
            let ptr = buffer.contents() as *const u8;
            std::slice::from_raw_parts(ptr, buffer.length() as usize)
        };

        Ok(data[start_offset..start_offset + length as usize].to_vec())
    }

    fn write(
        &self,
        file: FSVolumeItemIdentifier,
        offset: u64,
        data: &[u8]
    ) -> Result<u64> {
        // Similar to read, but in reverse:
        // 1. Load existing blocks (CPU I/O)
        // 2. Modify in unified memory
        // 3. Compress (GPU)
        // 4. Encrypt (GPU)
        // 5. Write back (CPU I/O)
        // 6. Update inode metadata

        // TODO: Implement
        Ok(data.len() as u64)
    }
}
```

---

## 3. Implementation Phases

### Phase 1: Minimum Viable Filesystem (MVP) - 3 weeks

**Goal:** Prove GPU-native filesystem works on M4

**Deliverables:**
1. Single-level directory (no subdirectories)
2. Fixed 4KB blocks
3. Simple bitmap allocation
4. Basic operations: create, read, write, delete
5. FSKit integration
6. Benchmark vs tmpfs

**Success Criteria:**
- Mounts in Finder
- Can create 10K files
- Read/write performance within 2x of tmpfs
- Path lookup <100μs

### Phase 2: Performance Optimizations - 4 weeks

**Goal:** Achieve 10x+ speedup on metadata operations

**Deliverables:**
1. Multi-level directories
2. Hash tables for fast lookup (Hive algorithm)
3. Radix sort for directory listings
4. LZ4 compression
5. Block cache in unified memory
6. Prefetching

**Success Criteria:**
- Directory listing (100K files) <5ms
- Global search (1M files) <100ms
- Compression: >30 GB/s
- Sustained I/O: >80% of raw NVMe bandwidth

### Phase 3: Advanced Features - 6 weeks

**Goal:** Modern filesystem capabilities

**Deliverables:**
1. Copy-on-write
2. Snapshots
3. Content-addressable storage (deduplication)
4. AES-256 encryption
5. CRC32 checksums
6. B-tree for large directories

**Success Criteria:**
- Snapshot creation: <10ms
- Real-time deduplication: <5% overhead
- Encryption: >100 GB/s
- Checksumming: >50 GB/s

---

## 4. Pseudocode Specifications

### 4.1 High-Level Operations

#### 4.1.1 Open File

```python
def open(path: str, flags: int) -> FileDescriptor:
    """
    Open a file and return file descriptor.

    Args:
        path: Absolute path like "/foo/bar.txt"
        flags: O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, etc.

    Returns:
        File descriptor (integer handle)
    """

    # GPU: Resolve path to inode
    inode_id = gpu_path_lookup(path)

    if inode_id == INVALID:
        if flags & O_CREAT:
            # Create new file
            parent_path = dirname(path)
            filename = basename(path)
            parent_inode = gpu_path_lookup(parent_path)

            if parent_inode == INVALID:
                raise FileNotFoundError(parent_path)

            # Allocate new inode
            inode_id = allocate_inode()

            # Add directory entry
            gpu_add_dir_entry(parent_inode, filename, inode_id)
        else:
            raise FileNotFoundError(path)

    # Check permissions
    inode = load_inode(inode_id)
    if not check_permissions(inode, flags):
        raise PermissionError(path)

    # Allocate file descriptor
    fd = allocate_fd()
    fd_table[fd] = FileDescriptorEntry(
        inode_id=inode_id,
        offset=0,
        flags=flags
    )

    return fd
```

#### 4.1.2 Read File

```python
def read(fd: int, count: int) -> bytes:
    """
    Read bytes from file descriptor.

    Args:
        fd: File descriptor from open()
        count: Number of bytes to read

    Returns:
        Byte array of read data
    """

    # Get FD entry
    entry = fd_table[fd]
    inode = load_inode(entry.inode_id)

    # Calculate block range
    offset = entry.offset
    start_block = offset // BLOCK_SIZE
    end_block = (offset + count - 1) // BLOCK_SIZE

    # CPU: Read blocks from disk to unified memory
    buffer = io_coordinator.read_blocks(
        inode.block_map[start_block:end_block+1]
    )

    # GPU: Decompress if needed
    if inode.flags & FLAG_COMPRESSED:
        buffer = gpu_decompress(buffer)

    # GPU: Decrypt if needed
    if inode.flags & FLAG_ENCRYPTED:
        buffer = gpu_decrypt(buffer, inode.inode_id)

    # Extract requested range
    start_offset = offset % BLOCK_SIZE
    data = buffer[start_offset:start_offset + count]

    # Update offset
    entry.offset += len(data)

    return data
```

#### 4.1.3 Write File

```python
def write(fd: int, data: bytes) -> int:
    """
    Write bytes to file descriptor.

    Args:
        fd: File descriptor from open()
        data: Bytes to write

    Returns:
        Number of bytes written
    """

    entry = fd_table[fd]
    inode = load_inode(entry.inode_id)
    offset = entry.offset

    # Calculate affected blocks
    start_block = offset // BLOCK_SIZE
    end_block = (offset + len(data) - 1) // BLOCK_SIZE
    blocks_needed = end_block - start_block + 1

    # Ensure blocks allocated
    while len(inode.block_map) <= end_block:
        new_block = allocate_block()
        inode.block_map.append(new_block)

    # CPU: Read existing blocks if partial write
    if offset % BLOCK_SIZE != 0 or len(data) % BLOCK_SIZE != 0:
        buffer = io_coordinator.read_blocks(
            inode.block_map[start_block:end_block+1]
        )
    else:
        buffer = allocate_buffer(blocks_needed * BLOCK_SIZE)

    # Modify buffer with new data
    start_offset = offset % BLOCK_SIZE
    memcpy(buffer[start_offset:], data)

    # GPU: Compress if enabled
    if inode.flags & FLAG_COMPRESSED:
        buffer = gpu_compress(buffer)

    # GPU: Encrypt if enabled
    if inode.flags & FLAG_ENCRYPTED:
        buffer = gpu_encrypt(buffer, inode.inode_id)

    # GPU: Compute checksum
    if inode.flags & FLAG_CHECKSUMMED:
        inode.checksum = gpu_crc32(buffer)

    # CPU: Write blocks to disk
    io_coordinator.write_blocks(
        buffer,
        inode.block_map[start_block:end_block+1]
    )

    # Update metadata
    inode.size = max(inode.size, offset + len(data))
    inode.mtime = now()
    store_inode(inode)

    # Update offset
    entry.offset += len(data)

    return len(data)
```

#### 4.1.4 List Directory

```python
def readdir(path: str) -> List[DirEntry]:
    """
    List directory contents.

    Args:
        path: Directory path like "/foo/bar"

    Returns:
        List of directory entries
    """

    # GPU: Resolve path to inode
    dir_inode_id = gpu_path_lookup(path)

    if dir_inode_id == INVALID:
        raise FileNotFoundError(path)

    inode = load_inode(dir_inode_id)

    if not inode.is_directory():
        raise NotADirectoryError(path)

    # GPU: List directory (parallel filter + sort)
    entries = gpu_list_directory(
        dir_inode_id,
        sort_by=SORT_BY_NAME
    )

    return entries
```

#### 4.1.5 Search Files

```python
def find(root: str, pattern: str) -> List[str]:
    """
    Search for files matching pattern.

    Args:
        root: Root directory to search from
        pattern: Glob pattern like "*.rs" or regex

    Returns:
        List of matching file paths
    """

    # GPU: Global search using HybridSA
    matches = gpu_global_search(
        root_inode_id=gpu_path_lookup(root),
        pattern=pattern,
        mode=SEARCH_NAME  # or SEARCH_CONTENT
    )

    # Convert inode IDs to paths
    paths = []
    for inode_id in matches:
        path = reconstruct_path(inode_id)
        paths.append(path)

    return paths
```

### 4.2 GPU Kernel Pseudocode

#### 4.2.1 Path Lookup Kernel (Detailed)

```metal
kernel void path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* dir_entries [[buffer(1)]],
    device HashBucket* hash_table [[buffer(2)]],
    constant char* path [[buffer(3)]],
    device PathLookupResult* result [[buffer(4)]],
    constant u32& dir_entry_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
)
{
    /*
     * Algorithm:
     * 1. Parse path into components (thread 0)
     * 2. For each component:
     *    a. Hash filename
     *    b. Parallel hash table lookup (all threads)
     *    c. Verify full filename match
     *    d. Load next directory
     * 3. Return final inode ID
     */

    // Shared memory for path components
    threadgroup char components[16][256];  // Max 16 components, 255 chars each
    threadgroup u32 component_lens[16];
    threadgroup u32 component_count;
    threadgroup u32 current_dir_inode;
    threadgroup atomic_uint found_inode;

    // === Phase 1: Parse path ===
    if (tid == 0) {
        component_count = 0;
        const char* p = path;

        // Skip leading slash
        if (*p == '/') p++;

        // Split on '/'
        while (*p && component_count < 16) {
            char* comp = components[component_count];
            u32 len = 0;

            while (*p && *p != '/' && len < 255) {
                comp[len++] = *p++;
            }
            comp[len] = '\0';
            component_lens[component_count] = len;
            component_count++;

            if (*p == '/') p++;
        }

        current_dir_inode = ROOT_INODE_ID;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 2: Walk directory tree ===
    for (u32 comp_idx = 0; comp_idx < component_count; comp_idx++) {
        if (tid == 0) {
            atomic_store_explicit(&found_inode, INVALID_INODE, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Get current component
        const char* component = components[comp_idx];
        u32 comp_len = component_lens[comp_idx];
        u32 comp_hash = xxhash3_32(component, comp_len);

        // === Parallel hash table probe ===
        u32 dir_inode = current_dir_inode;

        // Each thread probes different hash buckets
        // Assume directory hash table starts at offset for this dir
        u32 hash_table_offset = inodes[dir_inode].hash_table_offset;
        u32 hash_table_size = inodes[dir_inode].hash_table_size;

        // Linear probing with stride
        u32 start_bucket = comp_hash % hash_table_size;

        for (u32 probe = tid; probe < hash_table_size; probe += 1024) {
            u32 bucket_idx = (start_bucket + probe) % hash_table_size;
            HashBucket bucket = hash_table[hash_table_offset + bucket_idx];

            if (bucket.status == BUCKET_EMPTY) {
                // Not found
                break;
            }

            if (bucket.status == BUCKET_OCCUPIED && bucket.hash == comp_hash) {
                // Potential match - verify full string
                DirEntryCompact entry = dir_entries[bucket.value];

                if (entry.name_len == comp_len) {
                    bool match = true;
                    for (u32 i = 0; i < comp_len; i++) {
                        if (entry.name[i] != component[i]) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        // Found it!
                        atomic_store_explicit(&found_inode, entry.inode_id, memory_order_relaxed);
                        break;
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if found
        u32 next_inode = atomic_load_explicit(&found_inode, memory_order_relaxed);
        if (next_inode == INVALID_INODE) {
            // Path component not found
            if (tid == 0) {
                result->inode_id = INVALID_INODE;
                result->error = ERROR_NOT_FOUND;
            }
            return;
        }

        // Advance to next directory
        if (tid == 0) {
            current_dir_inode = next_inode;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Phase 3: Return result ===
    if (tid == 0) {
        result->inode_id = current_dir_inode;
        result->error = ERROR_SUCCESS;
    }
}
```

#### 4.2.2 Radix Sort for Directory Listing

```metal
kernel void radix_sort_dir_entries_kernel(
    device DirEntryCompact* entries [[buffer(0)]],
    device DirEntryCompact* temp [[buffer(1)]],
    constant u32& count [[buffer(2)]],
    constant u32& pass [[buffer(3)]],  // 0-3 for 4-byte radix
    uint tid [[thread_index_in_threadgroup]]
)
{
    /*
     * Radix sort on name_hash field (4 bytes = 4 passes)
     * Research shows radix sort is 2-4x faster than bitonic for large counts
     *
     * Each pass sorts on 8 bits (256 buckets)
     */

    const u32 RADIX = 256;
    threadgroup u32 local_counts[256];
    threadgroup u32 global_offsets[256];

    // === Phase 1: Count occurrences ===
    if (tid < RADIX) {
        local_counts[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Extract byte for this pass
    u32 shift = pass * 8;
    u32 mask = 0xFF;

    for (u32 i = tid; i < count; i += 1024) {
        u32 hash = entries[i].name_hash;
        u32 byte = (hash >> shift) & mask;
        atomic_fetch_add_explicit(&local_counts[byte], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 2: Prefix sum (scan) ===
    // Use parallel reduction for prefix sum
    if (tid < RADIX) {
        // Simple sequential scan within threadgroup (can be parallelized further)
        if (tid == 0) {
            global_offsets[0] = 0;
            for (u32 i = 1; i < RADIX; i++) {
                global_offsets[i] = global_offsets[i-1] + local_counts[i-1];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 3: Scatter ===
    threadgroup atomic_uint bucket_positions[256];

    if (tid < RADIX) {
        atomic_store_explicit(&bucket_positions[tid], global_offsets[tid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (u32 i = tid; i < count; i += 1024) {
        DirEntryCompact entry = entries[i];
        u32 hash = entry.name_hash;
        u32 byte = (hash >> shift) & mask;

        u32 pos = atomic_fetch_add_explicit(&bucket_positions[byte], 1, memory_order_relaxed);
        temp[pos] = entry;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // === Phase 4: Copy back ===
    for (u32 i = tid; i < count; i += 1024) {
        entries[i] = temp[i];
    }
}
```

---

## 5. Test Specifications

### 5.1 Unit Tests

#### Test 1: Path Parsing

```rust
#[test]
fn test_path_parsing() {
    let cases = vec![
        ("/", vec![]),
        ("/foo", vec!["foo"]),
        ("/foo/bar", vec!["foo", "bar"]),
        ("/foo/bar/baz.txt", vec!["foo", "bar", "baz.txt"]),
        ("/foo//bar", vec!["foo", "bar"]),  // Double slash
        ("/foo/./bar", vec!["foo", "bar"]),  // Dot component
    ];

    for (path, expected) in cases {
        let components = parse_path(path);
        assert_eq!(components, expected, "Failed for path: {}", path);
    }
}
```

#### Test 2: Inode Encoding/Decoding

```rust
#[test]
fn test_inode_compact_layout() {
    // Verify size
    assert_eq!(std::mem::size_of::<InodeCompact>(), 64);

    // Test flag encoding
    let mut inode = InodeCompact::new();
    inode.set_file_type(FileType::Regular);
    inode.set_compressed(true);
    inode.set_encrypted(false);

    assert_eq!(inode.get_file_type(), FileType::Regular);
    assert!(inode.is_compressed());
    assert!(!inode.is_encrypted());

    // Test timestamp packing
    let now = SystemTime::now();
    inode.set_timestamps(now, now, now);

    let (created, modified, accessed) = inode.get_timestamps();
    assert!((created - now).abs() < Duration::from_secs(1));
}
```

#### Test 3: Hash Table Operations

```rust
#[test]
fn test_hash_table_insert_lookup() {
    let device = Device::system_default().unwrap();
    let hash_table = GpuHashTable::new(&device, 1024);

    // Insert entries
    for i in 0..100 {
        let key = format!("file_{}.txt", i);
        let hash = xxhash3(&key);
        hash_table.insert(hash, i).unwrap();
    }

    // Lookup
    for i in 0..100 {
        let key = format!("file_{}.txt", i);
        let hash = xxhash3(&key);
        let value = hash_table.lookup(hash).unwrap();
        assert_eq!(value, Some(i));
    }

    // Negative test
    let hash = xxhash3("nonexistent.txt");
    assert_eq!(hash_table.lookup(hash).unwrap(), None);
}
```

#### Test 4: Directory Listing Sort

```rust
#[test]
fn test_directory_listing_sort() {
    let device = Device::system_default().unwrap();
    let metal_engine = MetalComputeEngine::new(&device).unwrap();

    // Create 1000 directory entries with random names
    let mut entries: Vec<DirEntryCompact> = (0..1000)
        .map(|i| {
            let name = format!("file_{:04}.txt", rand::random::<u16>());
            DirEntryCompact::new(i, &name)
        })
        .collect();

    // Sort on GPU
    let sorted = metal_engine.sort_dir_entries(&entries).unwrap();

    // Verify sorted
    for i in 1..sorted.len() {
        assert!(
            sorted[i-1].name_hash <= sorted[i].name_hash,
            "Not sorted at index {}", i
        );
    }
}
```

### 5.2 Integration Tests

#### Test 5: Create and Read File

```rust
#[tokio::test]
async fn test_create_read_file() {
    let fs = GpuFilesystem::new().await.unwrap();

    // Create file
    let fd = fs.open("/test.txt", O_CREAT | O_WRONLY).await.unwrap();
    let data = b"Hello, GPU filesystem!";
    let written = fs.write(fd, data).await.unwrap();
    assert_eq!(written, data.len());
    fs.close(fd).await.unwrap();

    // Read file
    let fd = fs.open("/test.txt", O_RDONLY).await.unwrap();
    let read_data = fs.read(fd, data.len()).await.unwrap();
    assert_eq!(read_data, data);
    fs.close(fd).await.unwrap();
}
```

#### Test 6: Directory Operations

```rust
#[tokio::test]
async fn test_directory_operations() {
    let fs = GpuFilesystem::new().await.unwrap();

    // Create directory
    fs.mkdir("/testdir").await.unwrap();

    // Create files in directory
    for i in 0..100 {
        let path = format!("/testdir/file_{}.txt", i);
        let fd = fs.open(&path, O_CREAT | O_WRONLY).await.unwrap();
        fs.write(fd, b"test").await.unwrap();
        fs.close(fd).await.unwrap();
    }

    // List directory
    let entries = fs.readdir("/testdir").await.unwrap();
    assert_eq!(entries.len(), 100);

    // Verify sorted
    for i in 1..entries.len() {
        assert!(entries[i-1].name <= entries[i].name);
    }
}
```

#### Test 7: Compression

```rust
#[tokio::test]
async fn test_compression() {
    let fs = GpuFilesystem::new().await.unwrap();
    fs.set_compression(true, CompressionAlgo::LZ4);

    // Create file with compressible data
    let fd = fs.open("/compressed.txt", O_CREAT | O_WRONLY).await.unwrap();
    let data = b"a".repeat(4096 * 10);  // 10 blocks of 'a'
    fs.write(fd, &data).await.unwrap();
    fs.close(fd).await.unwrap();

    // Verify compressed on disk
    let inode = fs.stat("/compressed.txt").await.unwrap();
    assert!(inode.is_compressed());
    assert!(inode.blocks < 10);  // Should be <10 blocks due to compression

    // Verify read returns original data
    let fd = fs.open("/compressed.txt", O_RDONLY).await.unwrap();
    let read_data = fs.read(fd, data.len()).await.unwrap();
    assert_eq!(read_data, data);
    fs.close(fd).await.unwrap();
}
```

### 5.3 Performance Tests

#### Test 8: Path Lookup Benchmark

```rust
#[bench]
fn bench_path_lookup(b: &mut Bencher) {
    let fs = GpuFilesystem::new().unwrap();

    // Create deep directory tree
    fs.mkdir_all("/a/b/c/d/e/f/g/h/i/j").unwrap();
    fs.create("/a/b/c/d/e/f/g/h/i/j/file.txt").unwrap();

    b.iter(|| {
        let inode = fs.path_lookup("/a/b/c/d/e/f/g/h/i/j/file.txt").unwrap();
        assert_ne!(inode, INVALID_INODE);
    });

    // Target: <50μs per lookup
}
```

#### Test 9: Directory Listing Benchmark

```rust
#[bench]
fn bench_directory_listing(b: &mut Bencher) {
    let fs = GpuFilesystem::new().unwrap();

    // Create directory with 100K files
    fs.mkdir("/large_dir").unwrap();
    for i in 0..100_000 {
        fs.create(&format!("/large_dir/file_{}.txt", i)).unwrap();
    }

    b.iter(|| {
        let entries = fs.readdir("/large_dir").unwrap();
        assert_eq!(entries.len(), 100_000);
    });

    // Target: <5ms for 100K files
}
```

#### Test 10: Global Search Benchmark

```rust
#[bench]
fn bench_global_search(b: &mut Bencher) {
    let fs = GpuFilesystem::new().unwrap();

    // Create 1M files
    for i in 0..1_000_000 {
        let ext = if i % 10 == 0 { "rs" } else { "txt" };
        fs.create(&format!("/file_{}.{}", i, ext)).unwrap();
    }

    b.iter(|| {
        let matches = fs.find("/", "*.rs").unwrap();
        assert_eq!(matches.len(), 100_000);
    });

    // Target: <100ms for 1M files
}
```

### 5.4 Correctness Tests

#### Test 11: POSIX Compliance

```rust
#[test]
fn test_posix_compliance() {
    let fs = GpuFilesystem::new().unwrap();

    // Test open flags
    let fd = fs.open("/new.txt", O_CREAT | O_EXCL | O_WRONLY).unwrap();
    assert!(fs.open("/new.txt", O_CREAT | O_EXCL).is_err());  // Already exists

    // Test permissions
    fs.chmod("/new.txt", 0o644).unwrap();
    let stat = fs.stat("/new.txt").unwrap();
    assert_eq!(stat.mode & 0o777, 0o644);

    // Test hard links
    fs.link("/new.txt", "/link.txt").unwrap();
    let stat = fs.stat("/new.txt").unwrap();
    assert_eq!(stat.nlink, 2);

    // Test unlink
    fs.unlink("/link.txt").unwrap();
    assert_eq!(fs.stat("/new.txt").unwrap().nlink, 1);
}
```

#### Test 12: Crash Consistency

```rust
#[test]
fn test_crash_consistency() {
    // Simulate crash during write
    let fs = GpuFilesystem::new().unwrap();

    fs.create("/test.txt").unwrap();
    let fd = fs.open("/test.txt", O_WRONLY).unwrap();

    // Write partial data
    fs.write(fd, b"Hello").unwrap();

    // Simulate crash (don't close FD)
    drop(fs);

    // Remount
    let fs = GpuFilesystem::mount_existing().unwrap();

    // Verify file is either:
    // 1. Empty (write not committed), or
    // 2. Contains "Hello" (write committed)
    let fd = fs.open("/test.txt", O_RDONLY).unwrap();
    let data = fs.read(fd, 100).unwrap();
    assert!(data.is_empty() || data == b"Hello");
}
```

---

## 6. Performance Requirements

### 6.1 Metadata Operations

| Operation | Count | Target Latency | Traditional | Speedup |
|-----------|-------|----------------|-------------|---------|
| Path lookup (deep) | - | <50μs | 500μs | 10x |
| Path lookup (shallow) | - | <20μs | 100μs | 5x |
| Directory listing | 1K files | <500μs | 5ms | 10x |
| Directory listing | 100K files | <5ms | 500ms | 100x |
| Directory listing | 1M files | <50ms | 30s | 600x |
| File create | - | <100μs | 1ms | 10x |
| File delete | - | <50μs | 500μs | 10x |
| Global search | 1M files | <100ms | 30s | 300x |

### 6.2 Data Operations

| Operation | Throughput Target | Notes |
|-----------|------------------|-------|
| Sequential read | >6 GB/s | 80% of M4 Max NVMe (7.5 GB/s) |
| Sequential write | >5 GB/s | 67% of NVMe (accounting for compression) |
| Random read (4KB) | >500K IOPS | Limited by NVMe, not GPU |
| Random write (4KB) | >300K IOPS | CoW overhead |
| Compression (LZ4) | >30 GB/s | Research shows 36 GB/s possible |
| Decompression (LZ4) | >100 GB/s | Research shows 118 GB/s |
| Encryption (AES-256) | >100 GB/s | M4 Max: 546 GB/s memory BW |
| Checksumming (CRC32) | >50 GB/s | 2x CPU baseline |

### 6.3 Memory Usage

| Component | Budget | Actual (1M files) |
|-----------|--------|-------------------|
| Inode table | <64MB | 64MB (1M × 64B) |
| Directory entries | <320MB | 320MB (10M × 32B) |
| Hash tables | <50MB | 16MB |
| B-trees | <500MB | 400MB |
| Block bitmap | <100MB | 32MB |
| File cache | <20GB | Variable |
| **Total metadata** | **<1GB** | **~850MB** |

### 6.4 Scalability

| Metric | Target | Notes |
|--------|--------|-------|
| Max files | 100M | Limited by memory (6.4GB inodes) |
| Max file size | 16TB | 4B blocks × 4KB |
| Max directory size | 10M files | Hash table + B-tree |
| Max path depth | 256 | Standard limit |
| Max filename length | 255 bytes | Standard limit |
| Concurrent operations | 1024 | One per GPU thread |

---

## 7. Success Metrics

### 7.1 Phase 1 (MVP) Success Criteria

**Must Have:**
- ✅ Mounts successfully in macOS Finder
- ✅ Create/read/write/delete files work
- ✅ Directory listing works
- ✅ Path lookup <100μs
- ✅ Read/write within 2x of tmpfs
- ✅ Zero kernel panics in 24h stress test

**Nice to Have:**
- Directory listing <10ms for 10K files
- Basic compression support

### 7.2 Phase 2 (Performance) Success Criteria

**Must Have:**
- ✅ Directory listing (100K files) <5ms
- ✅ Global search (1M files) <100ms
- ✅ Compression >30 GB/s
- ✅ Sequential I/O >80% NVMe bandwidth
- ✅ Memory usage <1GB for 1M files

**Nice to Have:**
- Prefetching reduces read latency by 50%
- Block cache hit rate >80% for typical workloads

### 7.3 Phase 3 (Features) Success Criteria

**Must Have:**
- ✅ Snapshots in <10ms
- ✅ Real-time deduplication <5% overhead
- ✅ Encryption >100 GB/s
- ✅ Data integrity (checksums) verified
- ✅ POSIX compliance test suite passes

**Nice to Have:**
- Content-addressable storage
- B-trees for 10M+ file directories

### 7.4 Production Readiness Criteria

- **Stability:** 0 crashes in 30-day test
- **Performance:** All benchmarks within 10% of targets
- **Compatibility:** Works with major macOS apps (Xcode, Finder, Terminal)
- **Documentation:** Complete API docs and user guide
- **Testing:** >90% code coverage, all POSIX tests pass
- **Security:** Pass security audit, no privilege escalation

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **CPU I/O bottleneck** | Medium | High | - Double buffering<br>- Prefetching<br>- Async I/O with multiple threads<br>- Measure and optimize |
| **FSKit overhead >100μs** | High | Medium | - Batch operations<br>- Cache aggressively<br>- Benchmark early |
| **Unified memory limits** | Low | High | - Design for 65% RAM usage<br>- Implement eviction policy<br>- Stream large files |
| **GPU kernel bugs** | Medium | High | - Extensive unit tests<br>- Validation on CPU first<br>- Metal shader debugger |
| **Atomics performance** | Medium | Medium | - Use lock-free algorithms<br>- Minimize contention<br>- Benchmark alternatives |

### 8.2 Product Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **macOS API changes** | Low | Medium | - Use stable FSKit APIs<br>- Test on beta releases<br>- Abstract OS layer |
| **User adoption** | Medium | High | - Strong benchmarks<br>- Clear value prop<br>- Easy installation |
| **Competition (APFS)** | High | Medium | - Differentiate on perf<br>- Target GPU-heavy users<br>- Open source |
| **Code signing issues** | Low | High | - Test notarization early<br>- Follow Apple guidelines |

### 8.3 Performance Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Small file overhead** | High | Medium | - Inline small files in inodes<br>- Aggressive caching<br>- Pack multiple files per block |
| **Random access slow** | Medium | Medium | - Prefetch on patterns<br>- Cache hot blocks<br>- Accept limitation |
| **Write amplification** | Medium | High | - Tune compression ratio<br>- CoW granularity<br>- Log-structured writes |
| **Dedup overhead** | Medium | Medium | - Make optional<br>- Background processing<br>- Adaptive based on duplicates found |

---

## Appendix A: Data Structure Sizes

```
InodeCompact:       64 bytes
DirEntryCompact:    32 bytes
BlockMapEntry:      8 bytes
HashBucket:         16 bytes

1M files filesystem:
- Inodes:           64 MB
- Dir entries:      320 MB (avg 10 per dir)
- Block map:        varies with file size
- Hash tables:      16 MB
- Total metadata:   ~400 MB

Memory efficiency: 400 bytes per file (metadata only)
```

## Appendix B: Metal Kernel Launch Configuration

```rust
// Threadgroup size
const THREADGROUP_SIZE: u64 = 1024;

// Workload distribution
match operation {
    PathLookup => {
        // Single threadgroup, all threads cooperate
        threadgroups: (1, 1, 1),
        threads_per_group: (1024, 1, 1)
    },

    DirectoryListing => {
        // Single threadgroup, parallel filter + sort
        threadgroups: (1, 1, 1),
        threads_per_group: (1024, 1, 1)
    },

    GlobalSearch => {
        // Multiple threadgroups for >1024 files
        let groups = (file_count + 1023) / 1024;
        threadgroups: (groups, 1, 1),
        threads_per_group: (1024, 1, 1)
    },

    Compression => {
        // One thread per block
        let groups = (block_count + 1023) / 1024;
        threadgroups: (groups, 1, 1),
        threads_per_group: (1024, 1, 1)
    }
}
```

## Appendix C: References

1. **M4 GPU Architecture Research** - `/docs/M4_GPU_ARCHITECTURE_RESEARCH.md`
2. **Filesystem Architecture Research** - `/docs/FILESYSTEM_ARCHITECTURE_RESEARCH.md`
3. **GPU Direct Storage Research** - `/docs/GPU_DIRECT_STORAGE_RESEARCH.md`
4. **macOS Integration Research** - `/docs/MACOS_FILESYSTEM_INTEGRATION_RESEARCH.md`
5. **GPU Algorithms Research** - `/docs/GPU_FILESYSTEM_ALGORITHMS_RESEARCH.md`

## Appendix D: Implementation Checklist

### Phase 1: MVP
- [ ] Data structures (Inode, DirEntry, BlockMap)
- [ ] GPU kernels (path lookup, dir listing)
- [ ] CPU I/O coordinator
- [ ] FSKit integration
- [ ] Basic tests
- [ ] Benchmark vs tmpfs

### Phase 2: Performance
- [ ] Hash tables (Hive implementation)
- [ ] Radix sort
- [ ] LZ4 compression
- [ ] Block cache
- [ ] Prefetching
- [ ] Performance tests

### Phase 3: Features
- [ ] Copy-on-write
- [ ] Snapshots
- [ ] Deduplication
- [ ] Encryption
- [ ] Checksums
- [ ] B-trees

### Production
- [ ] Documentation
- [ ] Security audit
- [ ] Stress testing
- [ ] Code signing & notarization
- [ ] User guide
- [ ] Benchmark suite

---

**End of PRD**
