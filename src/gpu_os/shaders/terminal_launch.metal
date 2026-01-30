//! Terminal Launch Command - Metal Kernels
//!
//! Issue #169 - Terminal Launch Command
//!
//! THE GPU IS THE COMPUTER.
//! GPU parses commands, searches index, launches apps.
//! CPU is only I/O coprocessor.

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

constant uint INPUT_BUFFER_SIZE = 4096;
constant uint OUTPUT_BUFFER_SIZE = 65536;
constant uint COMMAND_SIZE = 256;
constant uint HISTORY_SIZE = 32;
constant uint COMMAND_TABLE_SIZE = 64;

// Command handlers
constant uint HANDLER_UNKNOWN = 0;
constant uint HANDLER_LAUNCH = 1;
constant uint HANDLER_LS = 2;
constant uint HANDLER_CD = 3;
constant uint HANDLER_HELP = 4;
constant uint HANDLER_CLEAR = 5;
constant uint HANDLER_APPS = 6;

constant uint INVALID_HANDLE = 0xFFFFFFFF;
constant uint INVALID_PATH = 0xFFFFFFFF;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Terminal state (matches Rust struct)
struct TerminalState {
    // Input buffer (4096 bytes)
    uchar input_buffer[INPUT_BUFFER_SIZE];

    // Atomic input pointers (8 bytes)
    uint input_head;
    uint input_tail;

    // Command parsing (264 bytes)
    uchar current_command[COMMAND_SIZE];
    uint command_length;
    uint command_ready;

    // Output buffer (64KB)
    uchar output_buffer[OUTPUT_BUFFER_SIZE];
    uint output_length;
    uint scroll_offset;

    // Command history (8KB)
    uchar history[HISTORY_SIZE][COMMAND_SIZE];
    uint history_count;
    uint history_index;

    // Cursor position
    uint cursor_x;
    uint cursor_y;

    // Pending operations
    uint pending_app_load;
    uint pending_path_idx;

    // Padding
    uint _pad[2];
};

// Command entry
struct CommandEntry {
    uint name_hash;
    uint handler_id;
};

// Parsed command result
struct ParsedCommand {
    uint handler_id;
    uint args_offset;
    uint args_length;
    uint _pad;
};

// ═══════════════════════════════════════════════════════════════════════════════
// HASH FUNCTION (DJB2)
// ═══════════════════════════════════════════════════════════════════════════════

// DJB2 hash - O(n) but typically very short strings (< 10 chars)
inline uint hash_bytes(device const uchar* str, uint len) {
    uint h = 5381;
    for (uint i = 0; i < len; i++) {
        h = ((h << 5) + h) + str[i];
    }
    return h;
}

// Hash from threadgroup memory
inline uint hash_bytes_tg(threadgroup const uchar* str, uint len) {
    uint h = 5381;
    for (uint i = 0; i < len; i++) {
        h = ((h << 5) + h) + str[i];
    }
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TERMINAL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize terminal state
kernel void terminal_init(
    device TerminalState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    state->input_head = 0;
    state->input_tail = 0;
    state->command_length = 0;
    state->command_ready = 0;
    state->output_length = 0;
    state->scroll_offset = 0;
    state->history_count = 0;
    state->history_index = 0;
    state->cursor_x = 0;
    state->cursor_y = 0;
    state->pending_app_load = INVALID_HANDLE;
    state->pending_path_idx = INVALID_PATH;

    // Clear command buffer
    for (uint i = 0; i < COMMAND_SIZE; i++) {
        state->current_command[i] = 0;
    }

    // Write welcome message
    char welcome[] = "GPU Terminal v1.0\nType 'help' for commands.\n> ";
    uint len = 0;
    while (welcome[len] != 0 && len < 100) len++;

    for (uint i = 0; i < len; i++) {
        state->output_buffer[i] = welcome[i];
    }
    state->output_length = len;
}

// Append characters to input buffer
kernel void terminal_append_input(
    device TerminalState* state [[buffer(0)]],
    device const uchar* chars [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    for (uint i = 0; i < count; i++) {
        uchar ch = chars[i];

        if (ch == '\n' || ch == '\r') {
            // Command ready to process
            state->command_ready = 1;
        } else if (ch == 0x7F || ch == 0x08) {
            // Backspace
            if (state->command_length > 0) {
                state->command_length--;
                state->current_command[state->command_length] = 0;
            }
        } else if (ch >= 32 && ch < 127) {
            // Printable character
            if (state->command_length < COMMAND_SIZE - 1) {
                state->current_command[state->command_length] = ch;
                state->command_length++;

                // Echo to output
                if (state->output_length < OUTPUT_BUFFER_SIZE - 1) {
                    state->output_buffer[state->output_length] = ch;
                    state->output_length++;
                }
            }
        }
    }
}

// Parse current command and look up handler
kernel void terminal_parse_command(
    device TerminalState* state [[buffer(0)]],
    device const CommandEntry* command_table [[buffer(1)]],
    device ParsedCommand* result [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    result->handler_id = HANDLER_UNKNOWN;
    result->args_offset = 0;
    result->args_length = 0;

    if (state->command_length == 0) return;

    // Skip leading whitespace
    uint i = 0;
    while (i < state->command_length && (state->current_command[i] == ' ' || state->current_command[i] == '\t')) {
        i++;
    }

    // Extract command name
    uint cmd_start = i;
    while (i < state->command_length && state->current_command[i] != ' ' && state->current_command[i] != '\t') {
        i++;
    }
    uint cmd_len = i - cmd_start;

    if (cmd_len == 0) return;

    // Hash command name
    uint cmd_hash = hash_bytes(&state->current_command[cmd_start], cmd_len);

    // Look up in command table
    for (uint j = 0; j < COMMAND_TABLE_SIZE; j++) {
        if (command_table[j].name_hash == cmd_hash) {
            result->handler_id = command_table[j].handler_id;
            break;
        }
        if (command_table[j].name_hash == 0) break;  // End of table
    }

    // Skip whitespace to find args
    while (i < state->command_length && (state->current_command[i] == ' ' || state->current_command[i] == '\t')) {
        i++;
    }

    // Args are from here to end
    if (i < state->command_length) {
        result->args_offset = i;
        result->args_length = state->command_length - i;
    }
}

// Append string to output buffer
kernel void terminal_append_output(
    device TerminalState* state [[buffer(0)]],
    device const uchar* text [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint start = state->output_length;

    for (uint i = 0; i < count; i++) {
        if (start + i >= OUTPUT_BUFFER_SIZE) break;
        state->output_buffer[start + i] = text[i];
    }

    state->output_length = min(start + count, (uint)OUTPUT_BUFFER_SIZE);
}

// Clear command after processing
kernel void terminal_clear_command(
    device TerminalState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Save to history if non-empty
    if (state->command_length > 0 && state->history_count < HISTORY_SIZE) {
        uint hist_idx = state->history_count % HISTORY_SIZE;
        for (uint i = 0; i < state->command_length && i < COMMAND_SIZE; i++) {
            state->history[hist_idx][i] = state->current_command[i];
        }
        state->history[hist_idx][state->command_length] = 0;
        state->history_count++;
    }

    // Clear command
    for (uint i = 0; i < COMMAND_SIZE; i++) {
        state->current_command[i] = 0;
    }
    state->command_length = 0;
    state->command_ready = 0;
}

// Handle 'help' command - print help text
kernel void terminal_handle_help(
    device TerminalState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    char help_text[] = "\nAvailable commands:\n"
                                "  launch <app>  - Launch a .gpuapp application\n"
                                "  apps          - List available applications\n"
                                "  clear         - Clear the screen\n"
                                "  help          - Show this help message\n"
                                "\n> ";

    uint len = 0;
    while (help_text[len] != 0 && len < 512) len++;

    uint start = state->output_length;
    for (uint i = 0; i < len; i++) {
        if (start + i >= OUTPUT_BUFFER_SIZE) break;
        state->output_buffer[start + i] = help_text[i];
    }
    state->output_length = min(start + len, (uint)OUTPUT_BUFFER_SIZE);
}

// Handle 'clear' command
kernel void terminal_handle_clear(
    device TerminalState* state [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    state->output_length = 0;
    state->scroll_offset = 0;

    // Write prompt
    char prompt[] = "> ";
    state->output_buffer[0] = '>';
    state->output_buffer[1] = ' ';
    state->output_length = 2;
}

// Search filesystem index for app (O(1) hash lookup)
// Returns path_idx or INVALID_PATH
kernel void terminal_search_app(
    device const TerminalState* state [[buffer(0)]],
    device const uint* fs_buckets [[buffer(1)]],        // Hash buckets
    device const uint* fs_entry_hashes [[buffer(2)]],   // Entry name hashes
    device const uint* fs_entry_next [[buffer(3)]],     // Linked list next
    constant uint& bucket_count [[buffer(4)]],
    constant uint& args_offset [[buffer(5)]],
    constant uint& args_length [[buffer(6)]],
    device uint* result_path_idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    *result_path_idx = INVALID_PATH;

    if (args_length == 0) return;

    // Hash the app name + ".gpuapp"
    uint app_hash = hash_bytes(&state->current_command[args_offset], args_length);

    // Add ".gpuapp" hash (pre-computed: hash_command(".gpuapp") = some value)
    // For simplicity, we'll search for the raw name hash
    // In production, we'd append ".gpuapp" and hash that

    // Look up in filesystem index hash table
    // Max chain length to prevent unbounded iteration (hash table should be well-distributed)
    constant uint MAX_CHAIN_LENGTH = 256;

    uint bucket = app_hash % bucket_count;
    uint entry_idx = fs_buckets[bucket];

    for (uint chain_step = 0; chain_step < MAX_CHAIN_LENGTH && entry_idx != INVALID_PATH; chain_step++) {
        if (fs_entry_hashes[entry_idx] == app_hash) {
            *result_path_idx = entry_idx;
            return;
        }
        entry_idx = fs_entry_next[entry_idx];
    }
    // Not found or chain too long (indicates poor hash distribution)
}

// Queue app load request (called after search succeeds)
kernel void terminal_queue_load(
    device TerminalState* state [[buffer(0)]],
    device uint* io_request_queue [[buffer(1)]],
    device atomic_uint* io_queue_head [[buffer(2)]],
    constant uint& path_idx [[buffer(3)]],
    device uint* result_handle [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Simple queue: just store path_idx, real implementation uses IO pipeline
    uint slot = atomic_fetch_add_explicit(io_queue_head, 1, memory_order_relaxed);
    io_request_queue[slot % 128] = path_idx;

    state->pending_app_load = slot;
    state->pending_path_idx = path_idx;

    *result_handle = slot;
}
