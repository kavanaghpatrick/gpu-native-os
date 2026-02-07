//! Terminal Launch Command - GPU-Native App Discovery and Launch
//!
//! Issue #169 - Terminal Launch Command
//!
//! Architecture:
//! - GPU parses commands, searches index, queues loads
//! - CPU is only I/O coprocessor
//!
//! THE GPU IS THE COMPUTER. GPU owns the terminal.

use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::mem;
use std::sync::atomic::{fence, Ordering};

use crate::gpu_os::gpu_app_loader::INVALID_SLOT;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

pub const INPUT_BUFFER_SIZE: usize = 4096;
pub const OUTPUT_BUFFER_SIZE: usize = 65536;  // 64KB
pub const COMMAND_SIZE: usize = 256;
pub const HISTORY_SIZE: usize = 32;

// Command handlers
pub const HANDLER_UNKNOWN: u32 = 0;
pub const HANDLER_LAUNCH: u32 = 1;
pub const HANDLER_LS: u32 = 2;
pub const HANDLER_CD: u32 = 3;
pub const HANDLER_HELP: u32 = 4;
pub const HANDLER_CLEAR: u32 = 5;
pub const HANDLER_APPS: u32 = 6;

pub const INVALID_HANDLE: u32 = 0xFFFFFFFF;
pub const INVALID_PATH: u32 = 0xFFFFFFFF;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════════

/// Terminal State (GPU-resident)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TerminalState {
    // Input buffer (4096 bytes)
    pub input_buffer: [u8; INPUT_BUFFER_SIZE],

    // Atomic input pointers (8 bytes)
    pub input_head: u32,
    pub input_tail: u32,

    // Command parsing (264 bytes)
    pub current_command: [u8; COMMAND_SIZE],
    pub command_length: u32,
    pub command_ready: u32,

    // Output buffer (64KB)
    pub output_buffer: [u8; OUTPUT_BUFFER_SIZE],
    pub output_length: u32,
    pub scroll_offset: u32,

    // Command history (8KB)
    pub history: [[u8; COMMAND_SIZE]; HISTORY_SIZE],
    pub history_count: u32,
    pub history_index: u32,

    // Cursor position
    pub cursor_x: u32,
    pub cursor_y: u32,

    // Pending operations
    pub pending_app_load: u32,   // IO handle if loading
    pub pending_path_idx: u32,   // Which app being loaded

    // Padding to align nicely
    pub _pad: [u32; 2],
}

impl Default for TerminalState {
    fn default() -> Self {
        Self {
            input_buffer: [0; INPUT_BUFFER_SIZE],
            input_head: 0,
            input_tail: 0,
            current_command: [0; COMMAND_SIZE],
            command_length: 0,
            command_ready: 0,
            output_buffer: [0; OUTPUT_BUFFER_SIZE],
            output_length: 0,
            scroll_offset: 0,
            history: [[0; COMMAND_SIZE]; HISTORY_SIZE],
            history_count: 0,
            history_index: 0,
            cursor_x: 0,
            cursor_y: 0,
            pending_app_load: INVALID_HANDLE,
            pending_path_idx: INVALID_PATH,
            _pad: [0; 2],
        }
    }
}

/// Command entry for built-in commands
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandEntry {
    pub name_hash: u32,
    pub handler_id: u32,
}

/// Parsed command
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ParsedCommand {
    pub handler_id: u32,
    pub args_offset: u32,
    pub args_length: u32,
    pub _pad: u32,
}

impl Default for ParsedCommand {
    fn default() -> Self {
        Self {
            handler_id: HANDLER_UNKNOWN,
            args_offset: 0,
            args_length: 0,
            _pad: 0,
        }
    }
}

/// Terminal statistics
#[derive(Clone, Debug, Default)]
pub struct TerminalStats {
    pub input_head: u32,
    pub input_tail: u32,
    pub command_length: u32,
    pub command_ready: u32,
    pub output_length: u32,
    pub pending_app_load: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HASH FUNCTION (DJB2)
// ═══════════════════════════════════════════════════════════════════════════════

/// DJB2 hash function (same as in Metal shader)
pub const fn hash_command_const(s: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    let mut i = 0;
    while i < s.len() {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(s[i] as u32);
        i += 1;
    }
    h
}

/// Runtime hash function for dynamic strings
pub fn hash_command(s: &str) -> u32 {
    hash_command_const(s.as_bytes())
}

// Pre-computed command hashes (const, no runtime overhead)
pub const HASH_LAUNCH: u32 = hash_command_const(b"launch");
pub const HASH_LS: u32 = hash_command_const(b"ls");
pub const HASH_CD: u32 = hash_command_const(b"cd");
pub const HASH_HELP: u32 = hash_command_const(b"help");
pub const HASH_CLEAR: u32 = hash_command_const(b"clear");
pub const HASH_APPS: u32 = hash_command_const(b"apps");

// ═══════════════════════════════════════════════════════════════════════════════
// TERMINAL LAUNCH
// ═══════════════════════════════════════════════════════════════════════════════

/// Terminal Launch Command Handler
pub struct TerminalLaunch {
    // GPU buffers
    state_buffer: Buffer,
    command_table_buffer: Buffer,
    results_buffer: Buffer,

    // Compute pipelines
    init_pipeline: ComputePipelineState,
    parse_pipeline: ComputePipelineState,
    append_input_pipeline: ComputePipelineState,
    append_output_pipeline: ComputePipelineState,
}

impl TerminalLaunch {
    /// Create a new Terminal Launch handler
    pub fn new(device: &Device) -> Result<Self, String> {
        // Create state buffer
        let state_buffer = device.new_buffer(
            mem::size_of::<TerminalState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command table with built-in commands
        let mut command_table = vec![CommandEntry::default(); 64];
        command_table[0] = CommandEntry { name_hash: HASH_LAUNCH, handler_id: HANDLER_LAUNCH };
        command_table[1] = CommandEntry { name_hash: HASH_LS, handler_id: HANDLER_LS };
        command_table[2] = CommandEntry { name_hash: HASH_CD, handler_id: HANDLER_CD };
        command_table[3] = CommandEntry { name_hash: HASH_HELP, handler_id: HANDLER_HELP };
        command_table[4] = CommandEntry { name_hash: HASH_CLEAR, handler_id: HANDLER_CLEAR };
        command_table[5] = CommandEntry { name_hash: HASH_APPS, handler_id: HANDLER_APPS };

        let command_table_buffer = device.new_buffer_with_data(
            command_table.as_ptr() as *const _,
            (command_table.len() * mem::size_of::<CommandEntry>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Results buffer
        let results_buffer = device.new_buffer(
            256,
            MTLResourceOptions::StorageModeShared,
        );

        // Compile shader
        let shader_source = include_str!("shaders/terminal_launch.metal");
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile terminal_launch shader: {}", e))?;

        let init_pipeline = {
            let func = library
                .get_function("terminal_init", None)
                .map_err(|e| format!("Failed to get init function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create init pipeline: {}", e))?
        };

        let parse_pipeline = {
            let func = library
                .get_function("terminal_parse_command", None)
                .map_err(|e| format!("Failed to get parse function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create parse pipeline: {}", e))?
        };

        let append_input_pipeline = {
            let func = library
                .get_function("terminal_append_input", None)
                .map_err(|e| format!("Failed to get append_input function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create append_input pipeline: {}", e))?
        };

        let append_output_pipeline = {
            let func = library
                .get_function("terminal_append_output", None)
                .map_err(|e| format!("Failed to get append_output function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create append_output pipeline: {}", e))?
        };

        Ok(Self {
            state_buffer,
            command_table_buffer,
            results_buffer,
            init_pipeline,
            parse_pipeline,
            append_input_pipeline,
            append_output_pipeline,
        })
    }

    /// Initialize terminal state (GPU-side)
    pub fn initialize(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Append characters to input buffer (GPU-side)
    pub fn gpu_append_input(&self, device: &Device, chars: &[u8]) {
        if chars.is_empty() {
            return;
        }

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let chars_buffer = device.new_buffer_with_data(
            chars.as_ptr() as *const _,
            chars.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count = chars.len() as u32;
        let count_buffer = device.new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.append_input_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&chars_buffer), 0);
        encoder.set_buffer(2, Some(&count_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Parse current command (GPU-side)
    pub fn gpu_parse_command(&self, device: &Device) -> ParsedCommand {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.parse_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.command_table_buffer), 0);
        encoder.set_buffer(2, Some(&self.results_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read result
        let ptr = self.results_buffer.contents() as *const ParsedCommand;
        unsafe { *ptr }
    }

    /// Append string to output buffer (GPU-side)
    pub fn gpu_append_output(&self, device: &Device, text: &str) {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return;
        }

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let text_buffer = device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count = bytes.len() as u32;
        let count_buffer = device.new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.append_output_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&text_buffer), 0);
        encoder.set_buffer(2, Some(&count_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Read terminal statistics
    pub fn read_stats(&self) -> TerminalStats {
        let state = self.read_state();
        TerminalStats {
            input_head: state.input_head,
            input_tail: state.input_tail,
            command_length: state.command_length,
            command_ready: state.command_ready,
            output_length: state.output_length,
            pending_app_load: state.pending_app_load,
        }
    }

    /// Read full terminal state (CPU readback)
    /// Issue #256: Add memory fence to ensure we see GPU writes
    pub fn read_state(&self) -> TerminalState {
        fence(Ordering::Acquire);
        let ptr = self.state_buffer.contents() as *const TerminalState;
        unsafe { std::ptr::read_volatile(ptr) }
    }

    /// Read current command as string
    pub fn read_current_command(&self) -> String {
        let state = self.read_state();
        let len = state.command_length.min(COMMAND_SIZE as u32) as usize;
        String::from_utf8_lossy(&state.current_command[..len]).to_string()
    }

    /// Read output buffer as string
    pub fn read_output(&self) -> String {
        let state = self.read_state();
        let len = state.output_length.min(OUTPUT_BUFFER_SIZE as u32) as usize;
        String::from_utf8_lossy(&state.output_buffer[..len]).to_string()
    }

    /// Get buffer references
    pub fn state_buffer(&self) -> &Buffer {
        &self.state_buffer
    }

    pub fn command_table_buffer(&self) -> &Buffer {
        &self.command_table_buffer
    }
}

/// CPU-side command parsing (for testing/comparison)
pub fn cpu_parse_command(input: &str) -> (String, String) {
    let input = input.trim();
    let mut parts = input.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();
    (cmd.to_string(), args.to_string())
}

/// CPU-side command lookup (for testing/comparison)
pub fn cpu_lookup_command(cmd: &str) -> u32 {
    let hash = hash_command(cmd);
    if hash == HASH_LAUNCH { HANDLER_LAUNCH }
    else if hash == HASH_LS { HANDLER_LS }
    else if hash == HASH_CD { HANDLER_CD }
    else if hash == HASH_HELP { HANDLER_HELP }
    else if hash == HASH_CLEAR { HANDLER_CLEAR }
    else if hash == HASH_APPS { HANDLER_APPS }
    else { HANDLER_UNKNOWN }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_function() {
        // Test that hash is deterministic
        assert_eq!(hash_command("launch"), hash_command("launch"));
        assert_eq!(hash_command("apps"), hash_command("apps"));

        // Test that different commands have different hashes
        assert_ne!(hash_command("launch"), hash_command("apps"));
        assert_ne!(hash_command("ls"), hash_command("cd"));
    }

    #[test]
    fn test_cpu_parse_command() {
        assert_eq!(cpu_parse_command("launch myapp"), ("launch".to_string(), "myapp".to_string()));
        assert_eq!(cpu_parse_command("  launch  myapp  "), ("launch".to_string(), "myapp".to_string()));
        assert_eq!(cpu_parse_command("apps"), ("apps".to_string(), "".to_string()));
        assert_eq!(cpu_parse_command("help"), ("help".to_string(), "".to_string()));
    }

    #[test]
    fn test_cpu_lookup_command() {
        assert_eq!(cpu_lookup_command("launch"), HANDLER_LAUNCH);
        assert_eq!(cpu_lookup_command("apps"), HANDLER_APPS);
        assert_eq!(cpu_lookup_command("clear"), HANDLER_CLEAR);
        assert_eq!(cpu_lookup_command("unknown"), HANDLER_UNKNOWN);
    }

    #[test]
    fn test_terminal_state_size() {
        // Terminal state should be a reasonable size
        let size = std::mem::size_of::<TerminalState>();
        println!("TerminalState size: {} bytes", size);
        // Should be INPUT_BUFFER_SIZE + OUTPUT_BUFFER_SIZE + history + overhead
        // 4096 + 65536 + 8192 + ~400 = ~78KB
        assert!(size > 70000 && size < 90000);
    }
}
