//! GPU Text Buffer - Edit Log Architecture
//!
//! Issue #166 - GPU-Native Text Editing
//!
//! Architecture:
//! - Edit log instead of gap buffer (append-only, lockless)
//! - N threads process N elements (parallel by default)
//! - Compaction when edit log full
//! - Trivial undo/redo (truncate/extend edit log)

use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::sync::atomic::{fence, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

pub const CHUNK_SIZE: u32 = 4096;
pub const MAX_CHUNKS: u32 = 16384;      // 64MB max document
pub const MAX_EDITS: u32 = 65536;       // 64K edit entries
pub const MAX_LINES: u32 = 4194304;     // 4M lines
pub const MAX_MATCH_RESULTS: u32 = 65536;

pub const DEFAULT_STAGING_SIZE: u32 = 1024 * 1024; // 1MB

// Edit types
pub const EDIT_INSERT: u32 = 1;
pub const EDIT_DELETE: u32 = 2;
pub const EDIT_BATCH_INSERT: u32 = 3;

// Search flags
pub const SEARCH_CASE_INSENSITIVE: u32 = 1;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════════

/// Edit entry (32 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EditEntry {
    pub edit_type: u32,
    pub position: u32,
    pub length: u32,
    pub data_offset: u32,
    pub version: u32,
    pub flags: u32,
    pub _padding: [u32; 2],
}

/// Text buffer state (256 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TextBufferState {
    // Edit log state (16 bytes)
    pub edit_count: u32,
    pub edit_version: u32,
    pub staging_head: u32,
    pub compacted_version: u32,

    // Document metrics (16 bytes)
    pub total_bytes: u32,
    pub total_lines: u32,
    pub chunk_count: u32,
    pub max_edits: u32,

    // Cursor state (16 bytes)
    pub cursor_byte: u32,
    pub cursor_line: u32,
    pub cursor_column: u32,
    pub _pad1: u32,

    // Selection state (16 bytes)
    pub selection_anchor: u32,
    pub selection_active: u32,
    pub selection_start_line: u32,
    pub selection_end_line: u32,

    // Undo state (16 bytes)
    pub undo_edit_index: u32,
    pub redo_edit_index: u32,
    pub undo_boundary_count: u32,
    pub _pad2: u32,

    // Search results (16 bytes)
    pub match_count: u32,
    pub max_matches: u32,
    pub last_match_position: u32,
    pub _pad3: u32,

    // Line index (16 bytes)
    pub line_index_version: u32,
    pub line_index_valid: u32,
    pub staging_size: u32,
    pub _pad4: u32,

    // Performance counters (16 bytes)
    pub total_inserts: u32,
    pub total_deletes: u32,
    pub compaction_count: u32,
    pub _pad5: u32,

    // Reserved (128 bytes) - Pad to 256 total
    pub _reserved: [u32; 32],
}

impl Default for TextBufferState {
    fn default() -> Self {
        Self {
            edit_count: 0,
            edit_version: 0,
            staging_head: 0,
            compacted_version: 0,
            total_bytes: 0,
            total_lines: 1,
            chunk_count: 0,
            max_edits: MAX_EDITS,
            cursor_byte: 0,
            cursor_line: 0,
            cursor_column: 0,
            _pad1: 0,
            selection_anchor: 0,
            selection_active: 0,
            selection_start_line: 0,
            selection_end_line: 0,
            undo_edit_index: 0,
            redo_edit_index: 0,
            undo_boundary_count: 0,
            _pad2: 0,
            match_count: 0,
            max_matches: MAX_MATCH_RESULTS,
            last_match_position: 0,
            _pad3: 0,
            line_index_version: 0,
            line_index_valid: 0,
            staging_size: DEFAULT_STAGING_SIZE,
            _pad4: 0,
            total_inserts: 0,
            total_deletes: 0,
            compaction_count: 0,
            _pad5: 0,
            _reserved: [0; 32],
        }
    }
}

/// Match result (16 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchResult {
    pub position: u32,
    pub line: u32,
    pub column: u32,
    pub length: u32,
}

/// Statistics for debugging
#[derive(Clone, Debug, Default)]
pub struct TextBufferStats {
    pub edit_count: u32,
    pub edit_version: u32,
    pub total_bytes: u32,
    pub total_lines: u32,
    pub chunk_count: u32,
    pub total_inserts: u32,
    pub total_deletes: u32,
    pub compaction_count: u32,
    pub match_count: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU TEXT BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU Text Buffer with edit log architecture
pub struct GpuTextBuffer {
    // GPU buffers
    state_buffer: Buffer,
    chunks_buffer: Buffer,
    edit_log_buffer: Buffer,
    staging_buffer: Buffer,
    line_offsets_buffer: Buffer,
    match_results_buffer: Buffer,
    match_count_buffer: Buffer,
    compaction_temp_buffer: Buffer,
    chunk_line_counts_buffer: Buffer,

    // Compute pipelines
    init_pipeline: ComputePipelineState,
    batch_insert_pipeline: ComputePipelineState,
    batch_delete_pipeline: ComputePipelineState,
    parallel_find_pipeline: ComputePipelineState,
    undo_last_pipeline: ComputePipelineState,
    undo_to_version_pipeline: ComputePipelineState,
    redo_last_pipeline: ComputePipelineState,
    redo_to_version_pipeline: ComputePipelineState,
    build_line_index_phase1_pipeline: ComputePipelineState,
    build_line_index_phase2_pipeline: ComputePipelineState,
    calculate_logical_size_pipeline: ComputePipelineState,
    write_compacted_content_pipeline: ComputePipelineState,
    finalize_compaction_pipeline: ComputePipelineState,
    read_content_range_pipeline: ComputePipelineState,
    set_cursor_pipeline: ComputePipelineState,
    set_selection_pipeline: ComputePipelineState,

    // Configuration
    capacity_bytes: u32,
    staging_size: u32,
}

impl GpuTextBuffer {
    /// Create a new GPU text buffer
    pub fn new(device: &Device, capacity_bytes: u32) -> Result<Self, String> {
        let staging_size = DEFAULT_STAGING_SIZE;
        let chunk_count = (capacity_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;

        // Allocate buffers
        let state_buffer = device.new_buffer(
            std::mem::size_of::<TextBufferState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let chunks_buffer = device.new_buffer(
            (chunk_count * CHUNK_SIZE) as u64,
            MTLResourceOptions::StorageModeShared, // Shared for testing; Private in production
        );

        let edit_log_buffer = device.new_buffer(
            (MAX_EDITS as usize * std::mem::size_of::<EditEntry>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let staging_buffer = device.new_buffer(
            staging_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let line_offsets_buffer = device.new_buffer(
            (MAX_LINES * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let match_results_buffer = device.new_buffer(
            (MAX_MATCH_RESULTS as usize * std::mem::size_of::<MatchResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let match_count_buffer = device.new_buffer(
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let compaction_temp_buffer = device.new_buffer(
            (chunk_count * CHUNK_SIZE) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let chunk_line_counts_buffer = device.new_buffer(
            (chunk_count * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Compile shader
        let shader_source = include_str!("shaders/gpu_text_buffer.metal");
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile text buffer shader: {}", e))?;

        // Create all compute pipelines
        let init_pipeline = Self::create_pipeline(&library, device, "text_buffer_init")?;
        let batch_insert_pipeline = Self::create_pipeline(&library, device, "batch_insert")?;
        let batch_delete_pipeline = Self::create_pipeline(&library, device, "batch_delete")?;
        let parallel_find_pipeline = Self::create_pipeline(&library, device, "parallel_find")?;
        let undo_last_pipeline = Self::create_pipeline(&library, device, "undo_last")?;
        let undo_to_version_pipeline = Self::create_pipeline(&library, device, "undo_to_version")?;
        let redo_last_pipeline = Self::create_pipeline(&library, device, "redo_last")?;
        let redo_to_version_pipeline = Self::create_pipeline(&library, device, "redo_to_version")?;
        let build_line_index_phase1_pipeline = Self::create_pipeline(&library, device, "build_line_index_phase1")?;
        let build_line_index_phase2_pipeline = Self::create_pipeline(&library, device, "build_line_index_phase2")?;
        let calculate_logical_size_pipeline = Self::create_pipeline(&library, device, "calculate_logical_size")?;
        let write_compacted_content_pipeline = Self::create_pipeline(&library, device, "write_compacted_content")?;
        let finalize_compaction_pipeline = Self::create_pipeline(&library, device, "finalize_compaction")?;
        let read_content_range_pipeline = Self::create_pipeline(&library, device, "read_content_range")?;
        let set_cursor_pipeline = Self::create_pipeline(&library, device, "set_cursor")?;
        let set_selection_pipeline = Self::create_pipeline(&library, device, "set_selection")?;

        Ok(Self {
            state_buffer,
            chunks_buffer,
            edit_log_buffer,
            staging_buffer,
            line_offsets_buffer,
            match_results_buffer,
            match_count_buffer,
            compaction_temp_buffer,
            chunk_line_counts_buffer,
            init_pipeline,
            batch_insert_pipeline,
            batch_delete_pipeline,
            parallel_find_pipeline,
            undo_last_pipeline,
            undo_to_version_pipeline,
            redo_last_pipeline,
            redo_to_version_pipeline,
            build_line_index_phase1_pipeline,
            build_line_index_phase2_pipeline,
            calculate_logical_size_pipeline,
            write_compacted_content_pipeline,
            finalize_compaction_pipeline,
            read_content_range_pipeline,
            set_cursor_pipeline,
            set_selection_pipeline,
            capacity_bytes,
            staging_size,
        })
    }

    fn create_pipeline(
        library: &metal::Library,
        device: &Device,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let func = library
            .get_function(name, None)
            .map_err(|e| format!("Failed to get function {}: {}", name, e))?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("Failed to create pipeline {}: {}", name, e))
    }

    /// Initialize the text buffer with content
    pub fn initialize(&self, device: &Device, content: &[u8]) {
        // Copy content to chunks
        let chunk_ptr = self.chunks_buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), chunk_ptr, content.len());
        }

        // Initialize state via GPU
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let total_bytes = content.len() as u32;
        let total_bytes_buf = device.new_buffer_with_data(
            &total_bytes as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let staging_size_buf = device.new_buffer_with_data(
            &self.staging_size as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&total_bytes_buf), 0);
        encoder.set_buffer(2, Some(&staging_size_buf), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Read current state
    /// Issue #256: Add memory fence to ensure we see GPU writes
    pub fn read_state(&self) -> TextBufferState {
        fence(Ordering::Acquire);
        let ptr = self.state_buffer.contents() as *const TextBufferState;
        unsafe { std::ptr::read_volatile(ptr) }
    }

    /// Read statistics
    pub fn read_stats(&self) -> TextBufferStats {
        let state = self.read_state();
        TextBufferStats {
            edit_count: state.edit_count,
            edit_version: state.edit_version,
            total_bytes: state.total_bytes,
            total_lines: state.total_lines,
            chunk_count: state.chunk_count,
            total_inserts: state.total_inserts,
            total_deletes: state.total_deletes,
            compaction_count: state.compaction_count,
            match_count: state.match_count,
        }
    }

    /// Batch insert characters
    pub fn batch_insert(&self, device: &Device, positions: &[u32], chars: &[u8]) {
        if positions.is_empty() || chars.is_empty() {
            return;
        }

        let count = positions.len().min(chars.len());

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let positions_buf = device.new_buffer_with_data(
            positions.as_ptr() as *const _,
            (count * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let chars_buf = device.new_buffer_with_data(
            chars.as_ptr() as *const _,
            count as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count_u32 = count as u32;
        let count_buf = device.new_buffer_with_data(
            &count_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.batch_insert_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(2, Some(&self.staging_buffer), 0);
        encoder.set_buffer(3, Some(&positions_buf), 0);
        encoder.set_buffer(4, Some(&chars_buf), 0);
        encoder.set_buffer(5, Some(&count_buf), 0);

        let threads = MTLSize::new(count as u64, 1, 1);
        let threadgroup = MTLSize::new(64.min(count as u64), 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Batch delete characters
    pub fn batch_delete(&self, device: &Device, positions: &[u32], lengths: &[u32]) {
        if positions.is_empty() || lengths.is_empty() {
            return;
        }

        let count = positions.len().min(lengths.len());

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let positions_buf = device.new_buffer_with_data(
            positions.as_ptr() as *const _,
            (count * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let lengths_buf = device.new_buffer_with_data(
            lengths.as_ptr() as *const _,
            (count * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count_u32 = count as u32;
        let count_buf = device.new_buffer_with_data(
            &count_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.batch_delete_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(2, Some(&positions_buf), 0);
        encoder.set_buffer(3, Some(&lengths_buf), 0);
        encoder.set_buffer(4, Some(&count_buf), 0);

        let threads = MTLSize::new(count as u64, 1, 1);
        let threadgroup = MTLSize::new(64.min(count as u64), 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Parallel find - search for pattern in document
    pub fn find(&self, device: &Device, pattern: &[u8], case_insensitive: bool) -> Vec<MatchResult> {
        if pattern.is_empty() {
            return Vec::new();
        }

        // Reset match count
        let zero: u32 = 0;
        let match_count_ptr = self.match_count_buffer.contents() as *mut u32;
        unsafe { *match_count_ptr = zero; }

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let pattern_buf = device.new_buffer_with_data(
            pattern.as_ptr() as *const _,
            pattern.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pattern_len = pattern.len() as u32;
        let pattern_len_buf = device.new_buffer_with_data(
            &pattern_len as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let flags = if case_insensitive { SEARCH_CASE_INSENSITIVE } else { 0 };
        let flags_buf = device.new_buffer_with_data(
            &flags as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let state = self.read_state();
        let content_len = state.total_bytes;

        encoder.set_compute_pipeline_state(&self.parallel_find_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.chunks_buffer), 0);
        encoder.set_buffer(2, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(3, Some(&self.staging_buffer), 0);
        encoder.set_buffer(4, Some(&pattern_buf), 0);
        encoder.set_buffer(5, Some(&pattern_len_buf), 0);
        encoder.set_buffer(6, Some(&flags_buf), 0);
        encoder.set_buffer(7, Some(&self.match_results_buffer), 0);
        encoder.set_buffer(8, Some(&self.match_count_buffer), 0);

        if content_len > 0 {
            let threads = MTLSize::new(content_len as u64, 1, 1);
            let threadgroup = MTLSize::new(64, 1, 1);
            encoder.dispatch_threads(threads, threadgroup);
        }

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read results
        let match_count = unsafe { *match_count_ptr };
        let count = (match_count as usize).min(MAX_MATCH_RESULTS as usize);

        let results_ptr = self.match_results_buffer.contents() as *const MatchResult;
        let mut results = vec![MatchResult::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(results_ptr, results.as_mut_ptr(), count);
        }
        results
    }

    /// Undo last edit
    pub fn undo(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.undo_last_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Redo last undone edit
    pub fn redo(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.redo_last_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Read content range
    pub fn read_content(&self, device: &Device, start: u32, length: u32) -> Vec<u8> {
        if length == 0 {
            return Vec::new();
        }

        let output_buf = device.new_buffer(
            length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let start_buf = device.new_buffer_with_data(
            &start as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let length_buf = device.new_buffer_with_data(
            &length as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.read_content_range_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.chunks_buffer), 0);
        encoder.set_buffer(2, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(3, Some(&self.staging_buffer), 0);
        encoder.set_buffer(4, Some(&output_buf), 0);
        encoder.set_buffer(5, Some(&start_buf), 0);
        encoder.set_buffer(6, Some(&length_buf), 0);

        let threads = MTLSize::new(length as u64, 1, 1);
        let threadgroup = MTLSize::new(64.min(length as u64), 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read output
        let output_ptr = output_buf.contents() as *const u8;
        let mut result = vec![0u8; length as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(output_ptr, result.as_mut_ptr(), length as usize);
        }
        result
    }

    /// Get buffers for binding to megakernel
    pub fn bind_to_encoder(&self, encoder: &metal::ComputeCommandEncoderRef, base_index: u64) {
        encoder.set_buffer(base_index, Some(&self.state_buffer), 0);
        encoder.set_buffer(base_index + 1, Some(&self.chunks_buffer), 0);
        encoder.set_buffer(base_index + 2, Some(&self.edit_log_buffer), 0);
        encoder.set_buffer(base_index + 3, Some(&self.staging_buffer), 0);
        encoder.set_buffer(base_index + 4, Some(&self.line_offsets_buffer), 0);
        encoder.set_buffer(base_index + 5, Some(&self.match_results_buffer), 0);
    }

    /// Get state buffer (for external use)
    pub fn state_buffer(&self) -> &Buffer {
        &self.state_buffer
    }

    /// Get chunks buffer (for external use)
    pub fn chunks_buffer(&self) -> &Buffer {
        &self.chunks_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<EditEntry>(), 32);
        assert_eq!(std::mem::size_of::<TextBufferState>(), 256);
        assert_eq!(std::mem::size_of::<MatchResult>(), 16);
    }
}
