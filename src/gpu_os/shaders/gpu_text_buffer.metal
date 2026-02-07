//! GPU Text Buffer - Edit Log Architecture
//!
//! Issue #166 - GPU-Native Text Editing
//!
//! Architecture:
//! - Edit log instead of gap buffer (append-only, lockless)
//! - N threads process N elements (parallel by default)
//! - Compaction when edit log full
//! - Trivial undo/redo (truncate/extend edit log)

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

#define CHUNK_SIZE 4096
#define MAX_CHUNKS 16384      // 64MB max document
#define MAX_EDITS 65536       // 64K edit entries
#define MAX_LINES 4194304     // 4M lines
#define MAX_MATCH_RESULTS 65536

// Edit types
constant uint EDIT_INSERT = 1;
constant uint EDIT_DELETE = 2;
constant uint EDIT_BATCH_INSERT = 3;

// Flags
constant uint FLAG_UNDO_BOUNDARY = 1;
constant uint FLAG_SELECTION_EDIT = 2;

// Search flags
constant uint SEARCH_CASE_INSENSITIVE = 1;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Text chunk (4KB aligned)
struct TextChunk {
    uchar data[CHUNK_SIZE];
};

// Edit entry (32 bytes)
struct EditEntry {
    uint edit_type;         // EDIT_INSERT | EDIT_DELETE | EDIT_BATCH_INSERT
    uint position;          // Byte position in logical document
    uint length;            // Chars to delete or chars inserted
    uint data_offset;       // Offset into staging buffer for inserted text
    uint version;           // Monotonic version number
    uint flags;             // UNDO_BOUNDARY | SELECTION_EDIT
    uint _padding[2];
};

// Text buffer state (256 bytes)
struct TextBufferState {
    // Edit log state (16 bytes)
    atomic_uint edit_count;     // Number of edits in log
    atomic_uint edit_version;   // Current version number
    atomic_uint staging_head;   // Next staging slot
    uint compacted_version;     // Last compacted version

    // Document metrics (16 bytes)
    uint total_bytes;           // Total logical bytes after all edits
    uint total_lines;           // Total line count
    uint chunk_count;           // Active chunks in chunk array
    uint max_edits;             // MAX_EDITS

    // Cursor state (16 bytes)
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
    uint undo_edit_index;
    uint redo_edit_index;
    uint undo_boundary_count;
    uint _pad2;

    // Search results (16 bytes)
    atomic_uint match_count;
    uint max_matches;
    uint last_match_position;
    uint _pad3;

    // Line index (16 bytes)
    uint line_index_version;
    uint line_index_valid;
    uint staging_size;          // Total staging buffer size
    uint _pad4;

    // Performance counters (16 bytes)
    atomic_uint total_inserts;
    atomic_uint total_deletes;
    atomic_uint compaction_count;
    uint _pad5;

    // Reserved (96 bytes)
    uint _reserved[24];
};

// Match result (16 bytes)
struct MatchResult {
    uint position;
    uint line;
    uint column;
    uint length;
};

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

kernel void text_buffer_init(
    device TextBufferState* state [[buffer(0)]],
    constant uint& total_bytes [[buffer(1)]],
    constant uint& staging_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    atomic_store_explicit(&state->edit_count, 0, memory_order_relaxed);
    atomic_store_explicit(&state->edit_version, 0, memory_order_relaxed);
    atomic_store_explicit(&state->staging_head, 0, memory_order_relaxed);
    state->compacted_version = 0;

    state->total_bytes = total_bytes;
    state->total_lines = 1;  // At least one line
    state->chunk_count = (total_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
    state->max_edits = MAX_EDITS;

    state->cursor_byte = 0;
    state->cursor_line = 0;
    state->cursor_column = 0;

    state->selection_anchor = 0;
    state->selection_active = 0;
    state->selection_start_line = 0;
    state->selection_end_line = 0;

    state->undo_edit_index = 0;
    state->redo_edit_index = 0;
    state->undo_boundary_count = 0;

    atomic_store_explicit(&state->match_count, 0, memory_order_relaxed);
    state->max_matches = MAX_MATCH_RESULTS;
    state->last_match_position = 0;

    state->line_index_version = 0;
    state->line_index_valid = 0;
    state->staging_size = staging_size;

    atomic_store_explicit(&state->total_inserts, 0, memory_order_relaxed);
    atomic_store_explicit(&state->total_deletes, 0, memory_order_relaxed);
    atomic_store_explicit(&state->compaction_count, 0, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: Read logical byte through edit log
// ═══════════════════════════════════════════════════════════════════════════════

inline uchar read_logical_byte(
    device const TextBufferState* state,
    device const TextChunk* chunks,
    device const EditEntry* edit_log,
    device const uchar* staging,
    uint logical_pos
) {
    // Issue #271 fix: Use seqlock pattern to detect concurrent modifications
    // Retry up to 3 times if version changes during read
    for (uint retry = 0; retry < 3; retry++) {
        uint version_before = atomic_load_explicit(&state->edit_version, memory_order_relaxed);
        uint edit_count = atomic_load_explicit(&state->edit_count, memory_order_relaxed);
        uint physical_pos = logical_pos;
        uchar result = 0;
        bool found_in_staging = false;

        // Replay edits to find actual position
        for (uint e = 0; e < edit_count; e++) {
            EditEntry edit = edit_log[e];
            if (edit.edit_type == EDIT_INSERT && edit.position <= logical_pos) {
                if (logical_pos < edit.position + edit.length) {
                    // Byte is in this insertion - return from staging
                    result = staging[edit.data_offset + (logical_pos - edit.position)];
                    found_in_staging = true;
                    break;
                }
                physical_pos -= edit.length;  // Inserted bytes shift position
            } else if (edit.edit_type == EDIT_DELETE && edit.position <= logical_pos) {
                physical_pos += edit.length;  // Deleted bytes shift position
            }
        }

        // Check version didn't change during read
        uint version_after = atomic_load_explicit(&state->edit_version, memory_order_relaxed);
        if (version_before == version_after) {
            // Consistent read
            if (found_in_staging) {
                return result;
            }
            // Read from compacted chunks
            if (state->chunk_count == 0) return 0;
            uint chunk_idx = physical_pos / CHUNK_SIZE;
            uint byte_idx = physical_pos % CHUNK_SIZE;
            if (chunk_idx >= state->chunk_count) return 0;
            return chunks[chunk_idx].data[byte_idx];
        }
        // Version changed - retry
    }

    // After 3 retries, return 0 (best effort)
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH INSERT
// ═══════════════════════════════════════════════════════════════════════════════

kernel void batch_insert(
    device TextBufferState* state [[buffer(0)]],
    device EditEntry* edit_log [[buffer(1)]],
    device uchar* staging [[buffer(2)]],
    device const uint* insert_positions [[buffer(3)]],
    device const uchar* insert_chars [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_size) return;

    uint position = insert_positions[tid];
    uchar ch = insert_chars[tid];

    // Claim edit log slot
    uint edit_slot = atomic_fetch_add_explicit(&state->edit_count, 1, memory_order_relaxed);
    if (edit_slot >= state->max_edits) {
        atomic_fetch_sub_explicit(&state->edit_count, 1, memory_order_relaxed);
        return;
    }

    // Claim staging slot
    uint staging_slot = atomic_fetch_add_explicit(&state->staging_head, 1, memory_order_relaxed);
    if (staging_slot >= state->staging_size) {
        atomic_fetch_sub_explicit(&state->staging_head, 1, memory_order_relaxed);
        atomic_fetch_sub_explicit(&state->edit_count, 1, memory_order_relaxed);
        return;
    }
    staging[staging_slot] = ch;

    // Write edit entry
    edit_log[edit_slot].edit_type = EDIT_INSERT;
    edit_log[edit_slot].position = position;
    edit_log[edit_slot].length = 1;
    edit_log[edit_slot].data_offset = staging_slot;
    edit_log[edit_slot].version = atomic_fetch_add_explicit(&state->edit_version, 1, memory_order_relaxed);
    edit_log[edit_slot].flags = 0;

    // Update stats
    atomic_fetch_add_explicit(&state->total_inserts, 1, memory_order_relaxed);

    // Invalidate line index
    state->line_index_valid = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH DELETE
// ═══════════════════════════════════════════════════════════════════════════════

kernel void batch_delete(
    device TextBufferState* state [[buffer(0)]],
    device EditEntry* edit_log [[buffer(1)]],
    device const uint* delete_positions [[buffer(2)]],
    device const uint* delete_lengths [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_size) return;

    uint position = delete_positions[tid];
    uint length = delete_lengths[tid];

    // Claim edit log slot
    uint edit_slot = atomic_fetch_add_explicit(&state->edit_count, 1, memory_order_relaxed);
    if (edit_slot >= state->max_edits) {
        atomic_fetch_sub_explicit(&state->edit_count, 1, memory_order_relaxed);
        return;
    }

    // Write edit entry
    edit_log[edit_slot].edit_type = EDIT_DELETE;
    edit_log[edit_slot].position = position;
    edit_log[edit_slot].length = length;
    edit_log[edit_slot].data_offset = 0;
    edit_log[edit_slot].version = atomic_fetch_add_explicit(&state->edit_version, 1, memory_order_relaxed);
    edit_log[edit_slot].flags = 0;

    // Update stats
    atomic_fetch_add_explicit(&state->total_deletes, 1, memory_order_relaxed);

    // Invalidate line index
    state->line_index_valid = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARALLEL FIND
// ═══════════════════════════════════════════════════════════════════════════════

kernel void parallel_find(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],
    device const uchar* pattern [[buffer(4)]],
    constant uint& pattern_len [[buffer(5)]],
    constant uint& flags [[buffer(6)]],
    device MatchResult* matches [[buffer(7)]],
    device atomic_uint* match_count [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint content_len = state->total_bytes;
    if (tid + pattern_len > content_len) return;

    // Check if pattern matches at position tid
    bool match = true;
    for (uint i = 0; i < pattern_len && match; i++) {
        uchar byte = read_logical_byte(state, chunks, edit_log, staging, tid + i);
        uchar pattern_byte = pattern[i];

        // Case-insensitive comparison
        if (flags & SEARCH_CASE_INSENSITIVE) {
            if (byte >= 'A' && byte <= 'Z') byte += 32;
            if (pattern_byte >= 'A' && pattern_byte <= 'Z') pattern_byte += 32;
        }

        if (byte != pattern_byte) match = false;
    }

    if (match) {
        uint idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        if (idx < MAX_MATCH_RESULTS) {
            matches[idx].position = tid;
            matches[idx].line = 0;
            matches[idx].column = 0;
            matches[idx].length = pattern_len;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNDO / REDO
// ═══════════════════════════════════════════════════════════════════════════════

kernel void undo_last(
    device TextBufferState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint current = atomic_load_explicit(&state->edit_count, memory_order_relaxed);
    if (current == 0) return;

    // Save redo point
    state->redo_edit_index = current;

    // Truncate edit log by 1
    atomic_store_explicit(&state->edit_count, current - 1, memory_order_relaxed);
    state->line_index_valid = 0;
}

kernel void undo_to_version(
    device TextBufferState* state [[buffer(0)]],
    constant uint& target_version [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint current = atomic_load_explicit(&state->edit_count, memory_order_relaxed);
    if (target_version >= current) return;

    state->redo_edit_index = current;
    atomic_store_explicit(&state->edit_count, target_version, memory_order_relaxed);
    state->line_index_valid = 0;
}

kernel void redo_last(
    device TextBufferState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint current = atomic_load_explicit(&state->edit_count, memory_order_relaxed);
    if (current >= state->redo_edit_index) return;

    // Restore one edit
    atomic_store_explicit(&state->edit_count, current + 1, memory_order_relaxed);
    state->line_index_valid = 0;
}

kernel void redo_to_version(
    device TextBufferState* state [[buffer(0)]],
    constant uint& target_version [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    if (target_version > state->redo_edit_index) return;

    atomic_store_explicit(&state->edit_count, target_version, memory_order_relaxed);
    state->line_index_valid = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINE INDEX BUILD
// ═══════════════════════════════════════════════════════════════════════════════

// Phase 1: Count newlines in each chunk
kernel void build_line_index_phase1(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],
    device uint* chunk_line_counts [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->chunk_count) return;

    uint count = 0;
    uint base = tid * CHUNK_SIZE;

    for (uint i = 0; i < CHUNK_SIZE; i++) {
        uint logical_pos = base + i;
        if (logical_pos >= state->total_bytes) break;

        uchar byte = read_logical_byte(state, chunks, edit_log, staging, logical_pos);
        if (byte == '\n') count++;
    }

    chunk_line_counts[tid] = count;
}

// Phase 2: Write line offsets (after prefix sum computed)
kernel void build_line_index_phase2(
    device TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],
    device const uint* chunk_line_prefix [[buffer(4)]],
    device uint* line_offsets [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->chunk_count) return;

    uint line_num = (tid == 0) ? 1 : chunk_line_prefix[tid - 1] + 1;
    uint base = tid * CHUNK_SIZE;

    // First line starts at 0
    if (tid == 0) {
        line_offsets[0] = 0;
    }

    for (uint i = 0; i < CHUNK_SIZE; i++) {
        uint logical_pos = base + i;
        if (logical_pos >= state->total_bytes) break;

        uchar byte = read_logical_byte(state, chunks, edit_log, staging, logical_pos);
        if (byte == '\n' && line_num < MAX_LINES) {
            line_offsets[line_num] = logical_pos + 1;
            line_num++;
        }
    }

    // Thread 0 finalizes
    if (tid == 0) {
        state->total_lines = chunk_line_prefix[state->chunk_count - 1] + 1;
        state->line_index_version = atomic_load_explicit(&state->edit_version, memory_order_relaxed);
        state->line_index_valid = 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPACTION (Apply edits to create clean chunk array)
// ═══════════════════════════════════════════════════════════════════════════════

// Calculate logical size after all edits
kernel void calculate_logical_size(
    device TextBufferState* state [[buffer(0)]],
    device const EditEntry* edit_log [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint edit_count = atomic_load_explicit(&state->edit_count, memory_order_relaxed);
    int size_delta = 0;

    for (uint e = 0; e < edit_count; e++) {
        EditEntry edit = edit_log[e];
        if (edit.edit_type == EDIT_INSERT) {
            size_delta += int(edit.length);
        } else if (edit.edit_type == EDIT_DELETE) {
            size_delta -= int(edit.length);
        }
    }

    // Update total bytes based on original size plus delta
    // Note: This assumes original total_bytes is the compacted size
    state->total_bytes = uint(int(state->total_bytes) + size_delta);
}

// Write compacted content (each thread writes one byte)
kernel void write_compacted_content(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* old_chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],
    device TextChunk* new_chunks [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->total_bytes) return;

    // Read logical byte at this position
    uchar byte = read_logical_byte(state, old_chunks, edit_log, staging, tid);

    // Write to new chunks
    uint chunk_idx = tid / CHUNK_SIZE;
    uint byte_idx = tid % CHUNK_SIZE;
    new_chunks[chunk_idx].data[byte_idx] = byte;
}

// Finalize compaction (reset edit log)
kernel void finalize_compaction(
    device TextBufferState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Reset edit log
    atomic_store_explicit(&state->edit_count, 0, memory_order_relaxed);
    atomic_store_explicit(&state->staging_head, 0, memory_order_relaxed);
    state->compacted_version = atomic_load_explicit(&state->edit_version, memory_order_relaxed);
    state->chunk_count = (state->total_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Increment compaction count
    atomic_fetch_add_explicit(&state->compaction_count, 1, memory_order_relaxed);

    // Invalidate line index (will be rebuilt as needed)
    state->line_index_valid = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// READ CONTENT (For external use - get logical content)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void read_content_range(
    device const TextBufferState* state [[buffer(0)]],
    device const TextChunk* chunks [[buffer(1)]],
    device const EditEntry* edit_log [[buffer(2)]],
    device const uchar* staging [[buffer(3)]],
    device uchar* output [[buffer(4)]],
    constant uint& start_pos [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= length) return;

    uint logical_pos = start_pos + tid;
    if (logical_pos >= state->total_bytes) {
        output[tid] = 0;
        return;
    }

    output[tid] = read_logical_byte(state, chunks, edit_log, staging, logical_pos);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURSOR / SELECTION
// ═══════════════════════════════════════════════════════════════════════════════

kernel void set_cursor(
    device TextBufferState* state [[buffer(0)]],
    constant uint& byte_pos [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    state->cursor_byte = byte_pos;
    // Line/column will be updated when line index is valid
}

kernel void set_selection(
    device TextBufferState* state [[buffer(0)]],
    constant uint& anchor [[buffer(1)]],
    constant uint& active [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    state->selection_anchor = anchor;
    state->selection_active = active;
}
