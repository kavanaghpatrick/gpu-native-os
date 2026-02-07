# PRD: Terminal as Megakernel App (Issue #165)

## Overview

Port the terminal to run as an app inside GpuAppSystem's megakernel, using existing GPU-native infrastructure for file I/O, text rendering, and string processing.

## Infrastructure Used

| Need | Infrastructure | File |
|------|---------------|------|
| Keyboard input | GPU ring buffer | `input.rs` |
| Text output | Bitmap font renderer | `text_render.rs` |
| Command parsing | GPU tokenizer | `gpu_string.rs` |
| File operations | GPU-direct I/O | `gpu_io.rs` |
| Path completion | GPU filesystem index | `shared_index.rs` |
| File listing | GPU-resident index | `gpu_index.rs` |

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| `split_whitespace()` | `gpu_string::tokenize_query()` |
| `std::fs::read_dir()` | `shared_index.search()` |
| `std::fs::read()` | `GpuIOQueue::load_file_async()` |
| CPU cursor blink | GPU time-based toggle |

## Design

### Terminal State Structure

```metal
struct TerminalState {
    uint cursor_x;
    uint cursor_y;
    uint scroll_offset;
    uint line_count;
    uint max_lines;
    uint cols;
    uint rows;
    uint input_start;           // Start of editable region
    float cursor_blink_time;
    uint cursor_visible;

    // GPU-direct I/O integration
    uint pending_io_handle;     // GpuIOFileHandle for async loads
    uint io_status;             // IOStatus enum

    // Shared index integration
    uint completion_count;
    uint completion_selected;

    uint _pad[2];
    // char line_buffer[MAX_LINES * COLS] follows
    // char input_buffer[MAX_INPUT] follows
    // GpuPathEntry completions[MAX_COMPLETIONS] follows
};
```

### Terminal Update - Uses Existing Infrastructure

```metal
inline void terminal_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device InputEvent* input_queue,
    uint input_head,
    uint input_tail,
    device GpuPathEntry* fs_index,       // SHARED INDEX
    uint fs_index_count,
    device GpuIOCommandBuffer* io_queue, // GPU-DIRECT I/O
    float time,
    uint tid,
    uint tg_size
) {
    device TerminalState* state = (device TerminalState*)(unified_state + app->state_offset);
    device char* lines = (device char*)(state + 1);
    device char* input_buf = lines + MAX_LINES * COLS;
    device GpuPathEntry* completions = (device GpuPathEntry*)(input_buf + MAX_INPUT);

    // Thread 0: Process input from GPU ring buffer
    if (tid == 0) {
        // Cursor blink
        state->cursor_blink_time += 0.016;
        if (state->cursor_blink_time > 0.5) {
            state->cursor_visible = 1 - state->cursor_visible;
            state->cursor_blink_time = 0;
        }

        // Process keyboard from input_queue (INPUT.RS INFRASTRUCTURE)
        for (uint i = input_tail; i != input_head; i = (i + 1) % INPUT_QUEUE_SIZE) {
            InputEvent event = input_queue[i];
            if (event.event_type != EVENT_KEY_DOWN) continue;
            if (event.target_app != app->slot) continue;

            uint key = event.key_code;

            if (key == KEY_TAB) {
                // PATH COMPLETION using SHARED_INDEX.RS
                device char* partial = input_buf + state->input_start;
                uint partial_len = state->cursor_x - state->input_start;

                // GPU parallel search through fs_index
                state->completion_count = 0;
                for (uint j = 0; j < fs_index_count && state->completion_count < MAX_COMPLETIONS; j++) {
                    if (path_matches_prefix(fs_index[j].path, partial, partial_len)) {
                        completions[state->completion_count++] = fs_index[j];
                    }
                }
            } else if (key == KEY_ENTER) {
                // Parse command using GPU_STRING.RS infrastructure
                device GpuWord* words = (device GpuWord*)(unified_state + SCRATCH_OFFSET);
                uint word_count = gpu_tokenize(input_buf, state->cursor_x - state->input_start, words);

                // Execute command based on first word
                execute_terminal_command(state, words, word_count, fs_index, fs_index_count, io_queue);

                // New prompt
                state->cursor_x = 0;
                state->cursor_y++;
                write_prompt(lines, state);
            } else if (key >= 32 && key < 127) {
                lines[state->cursor_y * COLS + state->cursor_x] = (char)key;
                state->cursor_x++;
            }
        }

        // Check async I/O status (GPU_IO.RS INFRASTRUCTURE)
        if (state->pending_io_handle != INVALID_HANDLE) {
            IOStatus status = io_get_status(io_queue, state->pending_io_handle);
            if (status == IO_COMPLETE) {
                // File loaded - output contents
                device uchar* data = io_get_data(io_queue, state->pending_io_handle);
                uint size = io_get_size(io_queue, state->pending_io_handle);
                terminal_output_bytes(state, lines, data, size);
                state->pending_io_handle = INVALID_HANDLE;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // All threads: Render text using TEXT_RENDER.RS infrastructure
    // (Use existing write_glyph helper)
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    uint total_chars = state->rows * COLS;
    uint chars_per_thread = (total_chars + tg_size - 1) / tg_size;
    uint start = tid * chars_per_thread;
    uint end = min(start + chars_per_thread, total_chars);

    for (uint i = start; i < end; i++) {
        uint row = i / COLS;
        uint col = i % COLS;
        uint line_idx = state->scroll_offset + row;

        if (line_idx > state->line_count) continue;

        char c = lines[line_idx * COLS + col];
        if (c == 0 || c == ' ') continue;

        write_glyph(verts + i * 6,
            float2(col * 8.0, row * 16.0),
            c, app->window_depth,
            float4(0.9, 0.9, 0.9, 1.0));
    }

    if (tid == 0) {
        app->vertex_count = total_chars * 6;
    }
}
```

### Command Execution - Uses GPU Infrastructure

```metal
inline void execute_terminal_command(
    device TerminalState* state,
    device GpuWord* words,
    uint word_count,
    device GpuPathEntry* fs_index,
    uint fs_index_count,
    device GpuIOCommandBuffer* io_queue
) {
    if (word_count == 0) return;

    // Compare first word to known commands
    GpuWord cmd = words[0];

    if (word_equals(cmd, "ls")) {
        // LIST FILES - uses GPU_INDEX.RS
        // Search fs_index for entries in current directory
        uint result_count = 0;
        for (uint i = 0; i < fs_index_count && result_count < 100; i++) {
            if (fs_index[i].parent_idx == state->current_dir_idx) {
                // Write filename to output
                terminal_output_path(state, fs_index[i]);
                result_count++;
            }
        }
    } else if (word_equals(cmd, "cat") && word_count > 1) {
        // READ FILE - uses GPU_IO.RS
        // Find path in index
        for (uint i = 0; i < fs_index_count; i++) {
            if (path_matches(fs_index[i].path, words[1])) {
                // Queue async load via GPU-direct I/O
                state->pending_io_handle = io_queue_load(io_queue, fs_index[i].path);
                state->io_status = IO_PENDING;
                break;
            }
        }
    } else if (word_equals(cmd, "cd") && word_count > 1) {
        // CHANGE DIRECTORY - index lookup only
        for (uint i = 0; i < fs_index_count; i++) {
            if (path_matches(fs_index[i].path, words[1]) &&
                (fs_index[i].flags & FLAG_IS_DIR)) {
                state->current_dir_idx = i;
                break;
            }
        }
    } else if (word_equals(cmd, "find")) {
        // SEARCH - uses SHARED_INDEX.RS parallel search
        // GPU parallel search already exists!
        // Just invoke existing search infrastructure
    }
}
```

### Rust Integration - Minimal CPU

```rust
impl GpuOs {
    /// Terminal uses GPU infrastructure - minimal CPU involvement
    pub fn create_terminal(&mut self) -> Option<u32> {
        // Launch as megakernel app
        let slot = self.system.launch_by_type(app_type::TERMINAL)?;

        // Share filesystem index (already GPU-resident)
        // Share I/O queue (already exists)
        // No CPU infrastructure needed!

        Some(slot)
    }

    /// Only CPU involvement: external process execution
    /// (Eventually even this could be GPU-initiated)
    pub fn terminal_external_command(&mut self, slot: u32, cmd: &str) -> Option<String> {
        // Only for commands that MUST involve CPU processes
        // e.g., spawning external programs
        match cmd {
            cmd if cmd.starts_with("!") => {
                // Shell escape - truly needs CPU
                let output = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&cmd[1..])
                    .output()
                    .ok()?;
                Some(String::from_utf8_lossy(&output.stdout).into_owned())
            }
            _ => None // Most commands handled by GPU
        }
    }
}
```

## Tests

```rust
#[test]
fn test_terminal_uses_shared_index() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    // Terminal should have access to shared filesystem index
    let app = os.system.get_app(terminal).unwrap();
    assert!(app.has_shared_resource(SHARED_FS_INDEX));
}

#[test]
fn test_terminal_gpu_tokenization() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    // Type command
    for c in "cat README.md".chars() {
        os.system.queue_input(InputEvent::key_down(c as u32));
    }
    os.system.queue_input(InputEvent::key_down(KEY_ENTER));
    os.system.process_input();
    os.system.run_frame();

    // Command should be tokenized on GPU
    // Result should be: ["cat", "README.md"]
    let state = os.get_terminal_state(terminal);
    // Verify GPU processed command
}

#[test]
fn test_terminal_gpu_direct_file_load() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    // Type "cat test.txt"
    for c in "cat test.txt".chars() {
        os.system.queue_input(InputEvent::key_down(c as u32));
    }
    os.system.queue_input(InputEvent::key_down(KEY_ENTER));
    os.system.process_input();
    os.system.run_frame();

    // File should be loading via GPU-direct I/O
    let state = os.get_terminal_state(terminal);
    assert!(state.pending_io_handle != INVALID_HANDLE);

    // Run more frames until complete
    for _ in 0..100 {
        os.system.run_frame();
        if os.get_terminal_state(terminal).io_status == IO_COMPLETE {
            break;
        }
    }

    // File contents should be in terminal output
}

#[test]
fn test_terminal_path_completion() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    // Type partial path and tab
    for c in "src/".chars() {
        os.system.queue_input(InputEvent::key_down(c as u32));
    }
    os.system.queue_input(InputEvent::key_down(KEY_TAB));
    os.system.process_input();
    os.system.run_frame();

    // Should have completions from shared index
    let state = os.get_terminal_state(terminal);
    assert!(state.completion_count > 0);
}
```

## Benchmarks

```rust
#[test]
fn bench_terminal_ls_command() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();
    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    // Warm up
    for _ in 0..10 {
        os.system.run_frame();
    }

    let start = Instant::now();
    for _ in 0..100 {
        // Execute ls via GPU index search
        for c in "ls\n".chars() {
            os.system.queue_input(InputEvent::key_down(c as u32));
        }
        os.system.process_input();
        os.system.run_frame();
    }
    let duration = start.elapsed();

    println!("ls command: {}us average", duration.as_micros() / 100);
    // Should be <1ms since it's GPU index search, not CPU read_dir
}

#[test]
fn bench_terminal_file_load() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();
    let terminal = os.launch_app(app_type::TERMINAL).unwrap();

    let start = Instant::now();

    // cat a file using GPU-direct I/O
    for c in "cat Cargo.toml\n".chars() {
        os.system.queue_input(InputEvent::key_down(c as u32));
    }
    os.system.process_input();

    // Wait for GPU-direct load
    while os.get_terminal_state(terminal).io_status != IO_COMPLETE {
        os.system.run_frame();
    }

    let duration = start.elapsed();
    println!("File load (GPU-direct): {:?}", duration);
}
```

## Success Metrics

1. **Zero CPU in hot path**: ls, cat, cd all GPU-native
2. **<1ms file listing**: GPU index search vs 100ms read_dir
3. **GPU-direct file load**: No CPU involvement for cat
4. **GPU path completion**: Search index in parallel
5. **60 FPS typing**: No lag between keypress and display
