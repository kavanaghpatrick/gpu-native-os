# PRD: GPU Text Buffer - TRUE GPU-Native Text Editing

## Problem Statement

### Current State: CPU-Centric "GPU Text"

The existing `text_editor.rs` and the original PRD have fundamental anti-patterns:

| Anti-Pattern | Why It's Wrong |
|--------------|----------------|
| `if (tid != 0) return;` | Only 1 thread works, 63 threads wasted |
| Single-char operations | GPU hates single-element work |
| `barrier()` in hot path | All threads wait = catastrophic latency |
| Rust manages pipelines | CPU in the loop every frame |
| `GpuTextBuffer::insert_char()` | CPU call per keystroke |

**THE GPU IS THE COMPUTER.** Text editing must be designed for GPU parallelism from the ground up.

### Target State: TRUE GPU-Native Text Buffer

A text buffer where:
- **Batch all edits** - Never single-char, always batched
- **N threads process N elements** - Parallel by default
- **GPU owns edit log** - CPU never touches text data
- **No barriers in hot path** - Data structures avoid synchronization
- **Megakernel integration** - Text buffer accessed within persistent kernel

---

## GPU-First Architecture

### Core Insight: Edit Log, Not Gap Buffer

Gap buffers are CPU-optimized (O(1) at cursor, O(n) elsewhere). For GPU:

**Edit Log Architecture**:
```
Original:  "Hello World"
Edit log:  [INSERT(5, " Beautiful"), DELETE(11, 1)]
Snapshot:  Periodically compact log → new original
```

Why edit log wins on GPU:
1. **Append-only** - No data movement, lockless
2. **Parallel replay** - N threads replay N edits
3. **Trivial undo** - Just truncate log
4. **Batch-friendly** - Batch of edits = batch append

### Data Structures (Metal)

```metal
// ═══════════════════════════════════════════════════════════════════
// TEXT CHUNK (4KB aligned for cache efficiency)
// ═══════════════════════════════════════════════════════════════════

#define CHUNK_SIZE 4096
#define MAX_CHUNKS 16384  // 64MB max document

struct TextChunk {
    uchar data[CHUNK_SIZE];
};

// ═══════════════════════════════════════════════════════════════════
// EDIT ENTRY (32 bytes, cache-line friendly)
// ═══════════════════════════════════════════════════════════════════

#define EDIT_INSERT 1
#define EDIT_DELETE 2
#define EDIT_BATCH_INSERT 3  // Batch of chars from staging buffer

struct EditEntry {
    uint edit_type;         // EDIT_INSERT | EDIT_DELETE | EDIT_BATCH_INSERT
    uint position;          // Byte position in logical document
    uint length;            // For delete: chars to delete; for insert: chars inserted
    uint data_offset;       // Offset into staging buffer for inserted text
    uint version;           // Monotonic version number
    uint flags;             // UNDO_BOUNDARY | SELECTION_EDIT
    uint _padding[2];
};

#define MAX_EDITS 65536     // 64K edit entries before compaction

// ═══════════════════════════════════════════════════════════════════
// TEXT BUFFER STATE (256 bytes, single cache line group)
// ═══════════════════════════════════════════════════════════════════

struct TextBufferState {
    // Edit log state (16 bytes)
    atomic_uint edit_count;     // Number of edits in log
    atomic_uint edit_version;   // Current version number
    uint staging_size;          // Bytes in staging buffer
    uint compacted_version;     // Last compacted version

    // Document metrics (16 bytes)
    uint total_bytes;           // Total logical bytes after all edits
    uint total_lines;           // Total line count
    uint chunk_count;           // Active chunks in chunk array
    uint _pad0;

    // Cursor state (16 bytes) - GPU-readable, megakernel updates
    uint cursor_byte;
    uint cursor_line;
    uint cursor_column;
    uint _pad1;

    // Selection state (16 bytes)
    uint selection_anchor;
    uint selection_active;      // 0 = no selection
    uint selection_start_line;
    uint selection_end_line;

    // Undo state (16 bytes)
    uint undo_edit_index;       // Edit index for current undo point
    uint redo_edit_index;       // Edit index for redo limit
    uint undo_boundary_count;   // Number of undo boundaries
    uint _pad2;

    // Search results (16 bytes)
    atomic_uint match_count;
    uint max_matches;
    uint last_match_position;
    uint _pad3;

    // Line index (16 bytes)
    uint line_index_version;    // Version when line index was built
    uint line_index_valid;      // 1 if line index matches current version
    uint _pad4[2];

    // Performance counters (16 bytes)
    atomic_uint total_inserts;
    atomic_uint total_deletes;
    atomic_uint compaction_count;
    uint _pad5;

    // Reserved (96 bytes for future use)
    uint _reserved[24];
};

// ═══════════════════════════════════════════════════════════════════
// LINE INDEX (O(1) line lookup after parallel build)
// ═══════════════════════════════════════════════════════════════════

// Line start offsets - rebuilt in parallel after edits
// line_offsets[i] = byte offset where line i starts
// Max 4M lines (16MB buffer)
#define MAX_LINES 4194304

// ═══════════════════════════════════════════════════════════════════
// MATCH RESULT (for parallel find)
// ═══════════════════════════════════════════════════════════════════

struct MatchResult {
    uint position;      // Logical byte position
    uint line;          // Line number (if line index valid)
    uint column;        // Column (if line index valid)
    uint _padding;
};

#define MAX_MATCH_RESULTS 65536
```

### Buffer Layout

```
Buffer 0: text_state (256 bytes)
Buffer 1: chunks[] (64MB max)
Buffer 2: edit_log[] (2MB = 64K entries)
Buffer 3: staging_buffer (1MB = pending insertions)
Buffer 4: line_offsets[] (16MB = 4M lines)
Buffer 5: match_results[] (1MB)
Buffer 6: compaction_temp (64MB = temp buffer for compaction)
```

**Total GPU Memory**: ~150MB for 64MB documents

---

## Kernel Design: Batch Everything

### Kernel 1: BATCH_INSERT (N threads for N chars)

```metal
// Insert a batch of characters at various positions
// Called from megakernel when input buffer has pending chars
kernel void batch_insert(
    device TextBufferState* state [[buffer(0)]],
    device EditEntry* edit_log [[buffer(1)]],
    device uchar* staging [[buffer(2)]],

    // Input: batch of insertions
    device const uint* insert_positions [[buffer(3)]],
    device const uchar* insert_chars [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],

    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Phase 1: Each thread claims a slot in edit log and staging buffer
    if (tid >= batch_size) return;

    uint position = insert_positions[tid];
    uchar ch = insert_chars[tid];

    // Atomic claim slot in edit log
    uint edit_slot = atomic_fetch_add_explicit(&state->edit_count, 1, memory_order_relaxed);
    if (edit_slot >= MAX_EDITS) {
        // Edit log full - signal compaction needed
        return;
    }

    // Atomic claim slot in staging buffer
    uint staging_slot = atomic_fetch_add_explicit(&state->staging_size, 1, memory_order_relaxed);
    staging[staging_slot] = ch;

    // Write edit entry (no barriers needed - each thread writes own slot)
    edit_log[edit_slot].edit_type = EDIT_INSERT;
    edit_log[edit_slot].position = position;
    edit_log[edit_slot].length = 1;
    edit_log[edit_slot].data_offset = staging_slot;
    edit_log[edit_slot].version = atomic_fetch_add_explicit(&state->edit_version, 1, memory_order_relaxed);
    edit_log[edit_slot].flags = 0;

    // Invalidate line index
    state->line_index_valid = 0;
}
```

### Kernel 2: BATCH_DELETE (N threads for N deletions)

```metal
kernel void batch_delete(
    device TextBufferState* state [[buffer(0)]],
    device EditEntry* edit_log [[buffer(1)]],

    // Input: batch of deletions
    device const uint* delete_positions [[buffer(2)]],
    device const uint* delete_lengths [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],

    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_size) return;

    uint position = delete_positions[tid];
    uint length = delete_lengths[tid];

    // Atomic claim slot
    uint edit_slot = atomic_fetch_add_explicit(&state->edit_count, 1, memory_order_relaxed);
    if (edit_slot >= MAX_EDITS) return;

    edit_log[edit_slot].edit_type = EDIT_DELETE;
    edit_log[edit_slot].position = position;
    edit_log[edit_slot].length = length;
    edit_log[edit_slot].data_offset = 0;
    edit_log[edit_slot].version = atomic_fetch_add_explicit(&state->edit_version, 1, memory_order_relaxed);
    edit_log[edit_slot].flags = 0;

    state->line_index_valid = 0;
}
```

### Kernel 3: PARALLEL_COMPACTION (N threads copy N bytes)

When edit log exceeds threshold, compact to new chunk array:

```metal
// Phase 1: Calculate output positions with parallel prefix sum
kernel void compaction_phase1(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],

    // Output: per-byte output positions
    device uint* output_positions [[buffer(4)]],

    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Each thread processes a chunk of the original content
    // applying edits to calculate output position
    // Uses Hillis-Steele parallel prefix sum

    // ... (same pattern as parallel_alloc.rs)
}

// Phase 2: Write output content
kernel void compaction_phase2(
    device const TextChunk* old_chunks [[buffer(0)]],
    device TextChunk* new_chunks [[buffer(1)]],
    device const uint* output_positions [[buffer(2)]],
    device const EditEntry* edit_log [[buffer(3)]],
    device const uchar* staging [[buffer(4)]],
    constant uint& old_size [[buffer(5)]],

    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Each thread writes one byte to correct output position
    if (tid >= old_size) return;

    uint out_pos = output_positions[tid];
    if (out_pos == 0xFFFFFFFF) return;  // Byte was deleted

    // Find source byte (original or staging)
    // Copy to new_chunks[out_pos / CHUNK_SIZE].data[out_pos % CHUNK_SIZE]
}
```

### Kernel 4: PARALLEL_FIND (N threads check N positions)

```metal
kernel void parallel_find(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],

    // Pattern
    device const uchar* pattern [[buffer(4)]],
    constant uint& pattern_len [[buffer(5)]],
    constant uint& flags [[buffer(6)]],  // CASE_INSENSITIVE, etc.

    // Output
    device MatchResult* matches [[buffer(7)]],
    device atomic_uint* match_count [[buffer(8)]],

    uint tid [[thread_position_in_grid]]
) {
    // Calculate logical content size
    uint content_len = state->total_bytes;
    if (tid + pattern_len > content_len) return;

    // Each thread checks if pattern matches at position tid
    // Reads through edit log to get actual bytes
    bool match = true;
    for (uint i = 0; i < pattern_len && match; i++) {
        uchar byte = read_logical_byte(state, chunks, edit_log, staging, tid + i);
        uchar pattern_byte = pattern[i];

        // Case-insensitive comparison
        if (flags & 1) {
            if (byte >= 'A' && byte <= 'Z') byte += 32;
            if (pattern_byte >= 'A' && pattern_byte <= 'Z') pattern_byte += 32;
        }

        if (byte != pattern_byte) match = false;
    }

    if (match) {
        uint idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        if (idx < MAX_MATCH_RESULTS) {
            matches[idx].position = tid;
            matches[idx].line = 0;      // Filled by line index lookup
            matches[idx].column = 0;
        }
    }
}

// Helper: Read logical byte through edit log
// This is O(edits) but typically small; compaction keeps edits bounded
inline uchar read_logical_byte(
    device const TextBufferState* state,
    device const TextChunk* chunks,
    device const EditEntry* edit_log,
    device const uchar* staging,
    uint logical_pos
) {
    // Start with compacted content
    uint physical_pos = logical_pos;
    uint edit_count = atomic_load_explicit(&state->edit_count, memory_order_relaxed);

    // Replay edits to find actual position
    for (uint e = 0; e < edit_count; e++) {
        EditEntry edit = edit_log[e];
        if (edit.edit_type == EDIT_INSERT && edit.position <= logical_pos) {
            if (logical_pos < edit.position + edit.length) {
                // Byte is in this insertion - return from staging
                return staging[edit.data_offset + (logical_pos - edit.position)];
            }
            physical_pos -= edit.length;  // Inserted bytes shift position
        } else if (edit.edit_type == EDIT_DELETE && edit.position <= logical_pos) {
            physical_pos += edit.length;  // Deleted bytes shift position
        }
    }

    // Read from compacted chunks
    uint chunk_idx = physical_pos / CHUNK_SIZE;
    uint byte_idx = physical_pos % CHUNK_SIZE;
    return chunks[chunk_idx].data[byte_idx];
}
```

### Kernel 5: BUILD_LINE_INDEX (Parallel line counting)

```metal
// Two-phase line index build:
// Phase 1: Each thread counts newlines in its chunk
// Phase 2: Prefix sum to get global line offsets

kernel void build_line_index_phase1(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],

    device uint* chunk_line_counts [[buffer(4)]],  // Per-chunk newline count

    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->chunk_count) return;

    // Count newlines in chunk tid
    uint count = 0;
    for (uint i = 0; i < CHUNK_SIZE; i++) {
        uint logical_pos = tid * CHUNK_SIZE + i;
        if (logical_pos >= state->total_bytes) break;

        uchar byte = read_logical_byte(state, chunks, edit_log, staging, logical_pos);
        if (byte == '\n') count++;
    }

    chunk_line_counts[tid] = count;
}

kernel void build_line_index_phase2(
    device TextBufferState* state [[buffer(0)]],
    device const uint* chunk_line_counts [[buffer(1)]],
    device const uint* chunk_line_prefix [[buffer(2)]],  // Prefix sum of counts

    // Write line offsets
    device uint* line_offsets [[buffer(3)]],

    device const TextChunk* chunks [[buffer(4)]],
    device const EditEntry* edit_log [[buffer(5)]],
    device const uchar* staging [[buffer(6)]],

    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->chunk_count) return;

    // Starting line number for this chunk
    uint line_num = (tid == 0) ? 0 : chunk_line_prefix[tid - 1];

    // Scan chunk, record line offsets
    for (uint i = 0; i < CHUNK_SIZE; i++) {
        uint logical_pos = tid * CHUNK_SIZE + i;
        if (logical_pos >= state->total_bytes) break;

        uchar byte = read_logical_byte(state, chunks, edit_log, staging, logical_pos);
        if (byte == '\n') {
            line_offsets[line_num] = logical_pos + 1;  // Line starts after newline
            line_num++;
        }
    }

    // Thread 0 finalizes
    if (tid == 0) {
        line_offsets[0] = 0;  // First line at offset 0
        state->total_lines = chunk_line_prefix[state->chunk_count - 1];
        state->line_index_version = state->edit_version;
        state->line_index_valid = 1;
    }
}
```

### Kernel 6: BATCH_REPLACE_ALL (Two-phase parallel replace)

```metal
// Phase 1: Find all matches and calculate size delta
kernel void replace_phase1(
    device TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],

    device const uchar* find_pattern [[buffer(4)]],
    constant uint& find_len [[buffer(5)]],
    constant uint& replace_len [[buffer(6)]],

    device uint* is_match_start [[buffer(7)]],  // 1 if match starts at position
    device atomic_uint* match_count [[buffer(8)]],

    uint tid [[thread_position_in_grid]]
) {
    uint content_len = state->total_bytes;
    if (tid >= content_len) {
        is_match_start[tid] = 0;
        return;
    }

    // Check if pattern matches starting at tid
    is_match_start[tid] = 0;

    if (tid + find_len > content_len) return;

    bool match = true;
    for (uint i = 0; i < find_len && match; i++) {
        uchar byte = read_logical_byte(state, chunks, edit_log, staging, tid + i);
        if (byte != find_pattern[i]) match = false;
    }

    if (match) {
        is_match_start[tid] = 1;
        atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
    }
}

// Phase 2: Build new content with replacements
// Uses prefix sum of is_match_start to calculate output positions
kernel void replace_phase2(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* old_chunks [[buffer(1)]],
    device TextChunk* new_chunks [[buffer(2)]],

    device const uint* is_match_start [[buffer(3)]],
    device const uint* match_prefix_sum [[buffer(4)]],  // Exclusive prefix sum

    device const uchar* find_pattern [[buffer(5)]],
    device const uchar* replace_pattern [[buffer(6)]],
    constant uint& find_len [[buffer(7)]],
    constant uint& replace_len [[buffer(8)]],

    device const EditEntry* edit_log [[buffer(9)]],
    device const uchar* staging [[buffer(10)]],

    uint tid [[thread_position_in_grid]]
) {
    uint content_len = state->total_bytes;
    if (tid >= content_len) return;

    // Check if this byte is inside a matched region (skip it)
    bool inside_match = false;
    for (uint i = 1; i < find_len && tid >= i; i++) {
        if (is_match_start[tid - i]) {
            inside_match = true;
            break;
        }
    }

    if (inside_match) return;  // Skip - replaced by earlier match

    // Calculate output position
    int size_delta = int(replace_len) - int(find_len);
    uint matches_before = match_prefix_sum[tid];
    uint out_pos = tid + matches_before * size_delta;

    if (is_match_start[tid]) {
        // Write replacement pattern
        for (uint i = 0; i < replace_len; i++) {
            uint chunk_idx = (out_pos + i) / CHUNK_SIZE;
            uint byte_idx = (out_pos + i) % CHUNK_SIZE;
            new_chunks[chunk_idx].data[byte_idx] = replace_pattern[i];
        }
    } else {
        // Copy original byte
        uchar byte = read_logical_byte(state, old_chunks, edit_log, staging, tid);
        uint chunk_idx = out_pos / CHUNK_SIZE;
        uint byte_idx = out_pos % CHUNK_SIZE;
        new_chunks[chunk_idx].data[byte_idx] = byte;
    }
}
```

### Kernel 7: UNDO (Truncate edit log)

Undo is trivial with edit log:

```metal
kernel void undo_to_version(
    device TextBufferState* state [[buffer(0)]],
    constant uint& target_version [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single-threaded undo pointer update

    // Find edit index for target version
    uint current = atomic_load_explicit(&state->edit_count, memory_order_relaxed);

    // Binary search or linear scan to find edit at target_version
    // Truncate edit log to that point

    state->redo_edit_index = current;  // Remember for redo
    state->undo_edit_index = target_version;
    atomic_store_explicit(&state->edit_count, target_version, memory_order_release);
    state->line_index_valid = 0;
}
```

### Kernel 8: REDO (Extend edit log)

```metal
kernel void redo_to_version(
    device TextBufferState* state [[buffer(0)]],
    constant uint& target_version [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Restore edit count up to target (which was saved before undo)
    if (target_version <= state->redo_edit_index) {
        atomic_store_explicit(&state->edit_count, target_version, memory_order_release);
        state->line_index_valid = 0;
    }
}
```

---

## Integration with Megakernel

The text buffer is accessed within the persistent megakernel, not via CPU calls:

```metal
// In megakernel main loop
void process_text_input(
    device TextBufferState* text_state,
    device EditEntry* edit_log,
    device uchar* staging,
    device const InputEvent* input_events,
    uint input_count,
    uint tid,
    uint threads
) {
    // Batch all pending keystrokes
    threadgroup uint batch_insert_count = 0;
    threadgroup uint batch_insert_positions[64];
    threadgroup uchar batch_insert_chars[64];

    // Each thread processes some input events
    for (uint i = tid; i < input_count; i += threads) {
        InputEvent evt = input_events[i];
        if (evt.event_type == EVENT_KEY_DOWN && is_printable(evt.key_code)) {
            uint slot = atomic_fetch_add_explicit(&batch_insert_count, 1, memory_order_relaxed);
            if (slot < 64) {
                batch_insert_positions[slot] = text_state->cursor_byte;
                batch_insert_chars[slot] = key_to_char(evt.key_code);
                // Advance cursor for next keystroke
                text_state->cursor_byte++;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now batch_insert all collected chars
    for (uint i = tid; i < batch_insert_count; i += threads) {
        // ... (same as batch_insert kernel)
    }
}
```

---

## Rust Side: Minimal Binding (NOT State Management)

Rust does NOT manage text buffer state. Rust only:
1. Allocates GPU buffers at startup
2. Loads initial file content via GPU IO
3. Binds buffers to megakernel

```rust
/// GPU text buffer - Rust side only binds buffers, GPU owns state
pub struct GpuTextBuffer {
    // GPU buffers (allocated once at startup)
    state_buffer: Buffer,           // TextBufferState (256 bytes)
    chunks_buffer: Buffer,          // TextChunk[] (64MB)
    edit_log_buffer: Buffer,        // EditEntry[] (2MB)
    staging_buffer: Buffer,         // Pending insertions (1MB)
    line_offsets_buffer: Buffer,    // Line index (16MB)
    match_results_buffer: Buffer,   // Search results (1MB)
    compaction_temp_buffer: Buffer, // Compaction temp (64MB)
}

impl GpuTextBuffer {
    /// Create new text buffer with given capacity
    pub fn new(device: &Device, capacity_bytes: usize) -> Self {
        let state_buffer = device.new_buffer(
            256,
            MTLResourceOptions::StorageModeShared
        );

        let chunk_count = (capacity_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let chunks_buffer = device.new_buffer(
            chunk_count * CHUNK_SIZE,
            MTLResourceOptions::StorageModePrivate  // GPU-only
        );

        // ... allocate other buffers ...

        // Initialize state to zero
        unsafe {
            let state_ptr = state_buffer.contents() as *mut TextBufferState;
            std::ptr::write_bytes(state_ptr, 0, 1);
        }

        Self { state_buffer, chunks_buffer, /* ... */ }
    }

    /// Load file content directly to GPU chunks (zero CPU involvement)
    pub fn load_file(&self, io_queue: &GpuIOQueue, path: &Path) -> Result<(), Error> {
        // Use GPU IO to load file directly into chunks_buffer
        let file_handle = GpuIOFileHandle::open(io_queue.device(), path)?;
        let cmd = io_queue.command_buffer()?;

        let file_size = std::fs::metadata(path)?.len();
        cmd.load_buffer(&self.chunks_buffer, 0, file_size, &file_handle, 0);
        cmd.commit();
        // Don't wait - GPU will start using content when ready

        Ok(())
    }

    /// Get buffers for megakernel binding
    pub fn bind_to_megakernel(&self, encoder: &ComputeCommandEncoder, base_index: u32) {
        encoder.set_buffer(base_index + 0, Some(&self.state_buffer), 0);
        encoder.set_buffer(base_index + 1, Some(&self.chunks_buffer), 0);
        encoder.set_buffer(base_index + 2, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(base_index + 3, Some(&self.staging_buffer), 0);
        encoder.set_buffer(base_index + 4, Some(&self.line_offsets_buffer), 0);
        encoder.set_buffer(base_index + 5, Some(&self.match_results_buffer), 0);
    }

    /// Read state for display (GPU → CPU, use sparingly)
    pub fn read_state(&self) -> TextBufferState {
        unsafe {
            let ptr = self.state_buffer.contents() as *const TextBufferState;
            ptr.read()
        }
    }
}
```

---

## Performance Analysis

### Why Edit Log Beats Gap Buffer on GPU

| Operation | Gap Buffer | Edit Log | Winner |
|-----------|------------|----------|--------|
| Insert at cursor | O(1) | O(1) atomic | Tie |
| Insert elsewhere | O(n) data move | O(1) atomic | **Edit Log** |
| Delete | O(1) gap expand | O(1) atomic | Tie |
| Parallel insert | Sequential | N threads parallel | **Edit Log** |
| Find | O(n/threads) | O(n/threads) | Tie |
| Undo | Complex state restore | Truncate pointer | **Edit Log** |
| Memory locality | Gap fragments reads | Compaction restores | Edit Log (with compaction) |

### Compaction Amortization

Compaction is O(n) but happens infrequently:
- Trigger when edit log > 50% full (32K edits)
- Typical typing: 1000 edits/minute
- Compaction every ~30 minutes of continuous typing
- Compaction is fully parallel: 64MB in ~2ms on M1

### Thread Utilization

| Operation | Threads Used | Utilization |
|-----------|--------------|-------------|
| Batch insert (100 chars) | 100 | 100% |
| Find (1MB document) | 1M | 100% |
| Replace all | 2 passes, 1M threads each | 100% |
| Line index build | chunk_count threads | 100% |
| Undo/Redo | 1 thread | 1.5% (acceptable - O(1) op) |

---

## Success Criteria

| Metric | Target | Verification |
|--------|--------|--------------|
| Batch insert throughput | >100M chars/sec | Benchmark |
| Find in 1MB | <5ms | Benchmark |
| Replace all (1MB) | <20ms | Benchmark |
| Line index rebuild | <2ms | Benchmark |
| Compaction (64MB) | <5ms | Benchmark |
| CPU involvement per frame | 0 | Profiler |
| SIMD divergence | <5% | Metal profiler |

---

## Implementation Plan

### Week 1: Core Edit Log
- [ ] `TextBufferState`, `EditEntry`, `TextChunk` structs
- [ ] `batch_insert` kernel
- [ ] `batch_delete` kernel
- [ ] Basic Rust buffer binding

### Week 2: Compaction & Search
- [ ] `compaction_phase1/2` kernels
- [ ] `parallel_find` kernel
- [ ] Integration with prefix sum from `parallel_alloc.rs`

### Week 3: Replace & Line Index
- [ ] `replace_phase1/2` kernels
- [ ] `build_line_index_phase1/2` kernels
- [ ] Line number lookup for matches

### Week 4: Undo/Redo & Integration
- [ ] `undo_to_version`, `redo_to_version` kernels
- [ ] Megakernel integration
- [ ] Integration with existing text_editor app

### Week 5: Polish & Benchmarks
- [ ] Performance benchmarks
- [ ] GPU profiling for SIMD divergence
- [ ] Documentation

---

## Anti-Patterns Avoided

| Anti-Pattern | Why Wrong | What We Do Instead |
|--------------|-----------|-------------------|
| `if (tid != 0) return;` | Wastes 63/64 threads | Parallel everything |
| Single-char operations | GPU hates scalar work | Batch all operations |
| CPU manages state | CPU in the loop | GPU owns state |
| Gap buffer | Sequential cursor dependency | Edit log (append-only) |
| Barriers in hot path | All threads wait | Lockless atomics |
| CPU file loading | CPU copies bytes | GPU IO direct to chunks |

---

## References

- [Operational Transformation](https://en.wikipedia.org/wiki/Operational_transformation) - Edit log theory
- [CRDT](https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type) - Distributed edit log
- [GPU Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) - For compaction
- [Undo/Redo with Command Pattern](https://en.wikipedia.org/wiki/Command_pattern) - Edit log as command history
