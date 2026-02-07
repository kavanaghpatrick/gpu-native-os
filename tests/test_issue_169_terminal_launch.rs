//! Issue #169: Terminal Launch Command Tests
//!
//! Tests for GPU-native terminal command processing.
//! THE GPU IS THE COMPUTER - GPU parses, searches, launches.

use metal::Device;

use rust_experiment::gpu_os::terminal_launch::{
    TerminalLaunch, TerminalState, ParsedCommand,
    hash_command, cpu_parse_command, cpu_lookup_command,
    HANDLER_UNKNOWN, HANDLER_LAUNCH, HANDLER_LS, HANDLER_CD,
    HANDLER_HELP, HANDLER_CLEAR, HANDLER_APPS,
    INPUT_BUFFER_SIZE, OUTPUT_BUFFER_SIZE, COMMAND_SIZE,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CPU-SIDE TESTS (Hash and Parse)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_hash_deterministic() {
    assert_eq!(hash_command("launch"), hash_command("launch"));
    assert_eq!(hash_command("apps"), hash_command("apps"));
    assert_eq!(hash_command("help"), hash_command("help"));
}

#[test]
fn test_hash_different_strings() {
    assert_ne!(hash_command("launch"), hash_command("apps"));
    assert_ne!(hash_command("ls"), hash_command("cd"));
    assert_ne!(hash_command("help"), hash_command("clear"));
}

#[test]
fn test_cpu_parse_simple() {
    let (cmd, args) = cpu_parse_command("launch myapp");
    assert_eq!(cmd, "launch");
    assert_eq!(args, "myapp");
}

#[test]
fn test_cpu_parse_with_whitespace() {
    let (cmd, args) = cpu_parse_command("  launch  myapp  ");
    assert_eq!(cmd, "launch");
    assert_eq!(args, "myapp");
}

#[test]
fn test_cpu_parse_no_args() {
    let (cmd, args) = cpu_parse_command("apps");
    assert_eq!(cmd, "apps");
    assert_eq!(args, "");
}

#[test]
fn test_cpu_parse_empty() {
    let (cmd, args) = cpu_parse_command("");
    assert_eq!(cmd, "");
    assert_eq!(args, "");
}

#[test]
fn test_cpu_lookup_known_commands() {
    assert_eq!(cpu_lookup_command("launch"), HANDLER_LAUNCH);
    assert_eq!(cpu_lookup_command("ls"), HANDLER_LS);
    assert_eq!(cpu_lookup_command("cd"), HANDLER_CD);
    assert_eq!(cpu_lookup_command("help"), HANDLER_HELP);
    assert_eq!(cpu_lookup_command("clear"), HANDLER_CLEAR);
    assert_eq!(cpu_lookup_command("apps"), HANDLER_APPS);
}

#[test]
fn test_cpu_lookup_unknown_command() {
    assert_eq!(cpu_lookup_command("unknown"), HANDLER_UNKNOWN);
    assert_eq!(cpu_lookup_command("foo"), HANDLER_UNKNOWN);
    assert_eq!(cpu_lookup_command(""), HANDLER_UNKNOWN);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU-SIDE TESTS (Terminal Operations)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_terminal_creation() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    // Should compile successfully
    println!("Terminal created successfully");
}

#[test]
fn test_terminal_initialization() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    let stats = terminal.read_stats();
    assert_eq!(stats.command_length, 0);
    assert_eq!(stats.command_ready, 0);
    assert!(stats.output_length > 0, "Should have welcome message");

    let output = terminal.read_output();
    assert!(output.contains("GPU Terminal"), "Should have welcome message");
    assert!(output.contains(">"), "Should have prompt");

    println!("Terminal initialized with output: {:?}", output);
}

#[test]
fn test_terminal_append_input() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    // Append some characters
    terminal.gpu_append_input(&device, b"hello");

    let cmd = terminal.read_current_command();
    assert_eq!(cmd, "hello");

    let stats = terminal.read_stats();
    assert_eq!(stats.command_length, 5);
    assert_eq!(stats.command_ready, 0);

    println!("Command after input: {:?}", cmd);
}

#[test]
fn test_terminal_append_newline() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    // Type command and press enter
    terminal.gpu_append_input(&device, b"help\n");

    let stats = terminal.read_stats();
    assert_eq!(stats.command_ready, 1, "Command should be ready after newline");
    assert_eq!(stats.command_length, 4, "Command length should be 4 (without newline)");

    let cmd = terminal.read_current_command();
    assert_eq!(cmd, "help");

    println!("Command ready: {:?}", cmd);
}

#[test]
fn test_terminal_parse_launch_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"launch myapp\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_LAUNCH);
    assert!(parsed.args_length > 0, "Should have args");

    println!("Parsed: handler={}, args_offset={}, args_len={}",
             parsed.handler_id, parsed.args_offset, parsed.args_length);
}

#[test]
fn test_terminal_parse_apps_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"apps\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_APPS);
    assert_eq!(parsed.args_length, 0, "Should have no args");

    println!("Parsed apps command: handler={}", parsed.handler_id);
}

#[test]
fn test_terminal_parse_help_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"help\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_HELP);

    println!("Parsed help command: handler={}", parsed.handler_id);
}

#[test]
fn test_terminal_parse_clear_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"clear\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_CLEAR);

    println!("Parsed clear command: handler={}", parsed.handler_id);
}

#[test]
fn test_terminal_parse_unknown_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"unknown_cmd\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_UNKNOWN);

    println!("Parsed unknown command: handler={}", parsed.handler_id);
}

#[test]
fn test_terminal_parse_with_whitespace() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"  launch   myapp  \n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_LAUNCH);
    assert!(parsed.args_length > 0, "Should extract args");

    println!("Parsed with whitespace: handler={}, args_len={}",
             parsed.handler_id, parsed.args_length);
}

#[test]
fn test_terminal_append_output() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    let initial_output = terminal.read_output();
    let initial_len = initial_output.len();

    terminal.gpu_append_output(&device, "Hello, World!\n");

    let output = terminal.read_output();
    assert!(output.len() > initial_len, "Output should grow");
    assert!(output.ends_with("Hello, World!\n"), "Should end with appended text");

    println!("Output after append: {:?}", output);
}

#[test]
fn test_terminal_backspace() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    // Type "hello" then backspace (single backspace works)
    terminal.gpu_append_input(&device, b"hello");
    terminal.gpu_append_input(&device, &[0x7F]);  // One backspace

    let cmd = terminal.read_current_command();
    assert_eq!(cmd, "hell");

    // Another backspace
    terminal.gpu_append_input(&device, &[0x7F]);
    let cmd = terminal.read_current_command();
    assert_eq!(cmd, "hel");

    println!("Command after backspaces: {:?}", cmd);
}

#[test]
fn test_terminal_empty_command() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);
    terminal.gpu_append_input(&device, b"\n");

    let parsed = terminal.gpu_parse_command(&device);
    assert_eq!(parsed.handler_id, HANDLER_UNKNOWN);
    assert_eq!(parsed.args_length, 0);

    println!("Empty command parsed: handler={}", parsed.handler_id);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_terminal_batch_input() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    // Send a long command in one batch
    let long_input = "launch some_very_long_app_name_that_has_many_characters\n";
    terminal.gpu_append_input(&device, long_input.as_bytes());

    let cmd = terminal.read_current_command();
    let stats = terminal.read_stats();

    assert_eq!(stats.command_ready, 1);
    assert!(!cmd.is_empty());

    println!("Long command: {} chars", cmd.len());
}

#[test]
fn test_terminal_multiple_commands() {
    let device = Device::system_default().expect("No Metal device");
    let terminal = TerminalLaunch::new(&device).expect("Failed to create terminal");

    terminal.initialize(&device);

    // Process multiple commands in sequence
    let commands = ["help\n", "apps\n", "clear\n", "launch test\n"];
    let expected_handlers = [HANDLER_HELP, HANDLER_APPS, HANDLER_CLEAR, HANDLER_LAUNCH];

    for (i, cmd) in commands.iter().enumerate() {
        // Re-initialize to clear state between commands
        terminal.initialize(&device);
        terminal.gpu_append_input(&device, cmd.as_bytes());

        let parsed = terminal.gpu_parse_command(&device);
        assert_eq!(parsed.handler_id, expected_handlers[i],
                   "Command '{}' should have handler {}", cmd.trim(), expected_handlers[i]);
    }

    println!("All {} commands parsed correctly", commands.len());
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRUCT SIZE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_terminal_state_size() {
    let size = std::mem::size_of::<TerminalState>();
    println!("TerminalState size: {} bytes ({:.2} KB)", size, size as f64 / 1024.0);

    // Should be roughly:
    // INPUT_BUFFER_SIZE (4096) + OUTPUT_BUFFER_SIZE (65536) + history (32*256=8192) + overhead
    assert!(size >= 70000, "State should be at least 70KB");
    assert!(size <= 100000, "State should be at most 100KB");
}

#[test]
fn test_parsed_command_size() {
    let size = std::mem::size_of::<ParsedCommand>();
    assert_eq!(size, 16, "ParsedCommand should be 16 bytes");
}
