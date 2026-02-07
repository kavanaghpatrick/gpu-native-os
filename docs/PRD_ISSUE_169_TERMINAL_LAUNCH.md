# PRD: Terminal Launch Command - GPU-Native App Discovery and Launch

**Issue**: #169 - Terminal Launch Command
**Priority**: High (User-facing app launching)
**Status**: Design
**Depends On**: #168 (GPU App Loader)

---

## THE GPU IS THE COMPUTER

**The CPU does NOT process commands. The GPU parses, searches, and launches.**

### CPU-Centric Thinking (WRONG)
```
User types "launch myapp"
CPU: read terminal buffer
CPU: parse command
CPU: search filesystem
CPU: load app file
CPU: initialize app
CPU: tell GPU "new app ready"
GPU: "ok"
```

### GPU-Native Thinking (RIGHT)
```
User types "launch myapp"
GPU: terminal_update() sees newline in input buffer
GPU: parse_command() extracts "launch" and "myapp"
GPU: search_filesystem_index() finds myapp.gpuapp (O(1) hash lookup)
GPU: queue_app_load() requests file via I/O queue
GPU: continues running (never waits)
GPU: poll_pending_loads() sees file ready
GPU: initialize_app_from_gpuapp() creates app
GPU: app runs

CPU I/O thread: (meanwhile) loads file bytes async
```

**The GPU owns the terminal. GPU parses commands. GPU searches index. GPU launches apps.**

---

## Architecture

### Terminal State (GPU-Owned)

```metal
struct TerminalState {
    // Input buffer (keyboard input accumulates here)
    uchar input_buffer[INPUT_BUFFER_SIZE];  // 4096 bytes
    atomic_uint input_head;
    atomic_uint input_tail;

    // Command parsing
    uchar current_command[256];
    uint command_length;
    uint command_ready;  // 1 when newline received

    // Output buffer (terminal displays this)
    uchar output_buffer[OUTPUT_BUFFER_SIZE];  // 64KB
    uint output_length;
    uint scroll_offset;

    // Command history
    uchar history[HISTORY_SIZE][256];
    uint history_count;
    uint history_index;

    // Cursor
    uint cursor_x;
    uint cursor_y;

    // Pending operations
    uint pending_app_load;  // IO handle if loading app
    uint pending_path_idx;  // Which app being loaded
};
```

### Command Table (GPU-Resident)

```metal
// Built-in commands - O(1) lookup via hash
struct CommandEntry {
    uint name_hash;         // hash("launch"), hash("ls"), etc.
    uint handler_id;        // HANDLER_LAUNCH, HANDLER_LS, etc.
};

constant uint HANDLER_LAUNCH = 1;
constant uint HANDLER_LS = 2;
constant uint HANDLER_CD = 3;
constant uint HANDLER_HELP = 4;
constant uint HANDLER_CLEAR = 5;
constant uint HANDLER_APPS = 6;  // List available apps

#define COMMAND_TABLE_SIZE 64

// Hash function for command lookup
inline uint hash_command(device const uchar* str, uint len) {
    uint h = 5381;
    for (uint i = 0; i < len; i++) {
        h = ((h << 5) + h) + str[i];
    }
    return h;
}
```

### Data Flow

```
1. INPUT HANDLING (Every frame)
   ┌─────────────────────────────────────────────────────────────┐
   │ // In megakernel, terminal processes input                   │
   │ inline void terminal_process_input(                          │
   │     device TerminalState* term,                              │
   │     device InputEvent* events,                               │
   │     uint event_count                                         │
   │ ) {                                                          │
   │     for (uint i = 0; i < event_count; i++) {                │
   │         if (events[i].event_type == KEY_DOWN) {             │
   │             uchar ch = keycode_to_char(events[i].key);      │
   │             if (ch == '\n') {                                │
   │                 term->command_ready = 1;                     │
   │             } else {                                         │
   │                 append_to_input(term, ch);                   │
   │             }                                                 │
   │         }                                                     │
   │     }                                                         │
   │ }                                                             │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. COMMAND PARSING (When newline received)
   ┌─────────────────────────────────────────────────────────────┐
   │ if (term->command_ready) {                                   │
   │     // Extract command name and args                         │
   │     uchar cmd_name[32];                                      │
   │     uchar cmd_args[224];                                     │
   │     parse_command(term->current_command, cmd_name, cmd_args);│
   │                                                              │
   │     // Hash lookup for handler                               │
   │     uint hash = hash_command(cmd_name, strlen(cmd_name));   │
   │     uint handler = lookup_command_handler(hash);            │
   │                                                              │
   │     // Dispatch                                              │
   │     switch (handler) {                                       │
   │         case HANDLER_LAUNCH: handle_launch(term, cmd_args); │
   │         case HANDLER_APPS: handle_apps(term);               │
   │         // ...                                               │
   │     }                                                         │
   │                                                              │
   │     term->command_ready = 0;                                 │
   │     term->command_length = 0;                                │
   │ }                                                             │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. LAUNCH HANDLER (Searches index, queues load)
   ┌─────────────────────────────────────────────────────────────┐
   │ inline void handle_launch(                                   │
   │     device TerminalState* term,                              │
   │     device uchar* app_name,                                  │
   │     device FilesystemIndex* fs_index,                        │
   │     device PipelineState* io_state,                          │
   │     device IORequest* io_queue,                              │
   │     device FileHandle* io_handles                            │
   │ ) {                                                          │
   │     // Search for app.gpuapp in filesystem index             │
   │     uint path_idx = search_index_for_app(fs_index, app_name);│
   │                                                              │
   │     if (path_idx == INVALID_PATH) {                         │
   │         terminal_print(term, "App not found: ");            │
   │         terminal_print(term, app_name);                     │
   │         return;                                              │
   │     }                                                         │
   │                                                              │
   │     // Request file load                                     │
   │     uint handle = request_read(io_state, io_queue, io_handles,│
   │                                path_idx, TERMINAL_APP_ID,    │
   │                                IO_PRIORITY_HIGH);            │
   │                                                              │
   │     if (handle == INVALID_HANDLE) {                         │
   │         terminal_print(term, "Failed to start load");       │
   │         return;                                              │
   │     }                                                         │
   │                                                              │
   │     // Track pending load                                    │
   │     term->pending_app_load = handle;                        │
   │     term->pending_path_idx = path_idx;                      │
   │     terminal_print(term, "Loading ");                       │
   │     terminal_print(term, app_name);                         │
   │     terminal_print(term, "...\n");                          │
   │ }                                                             │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. POLL PENDING LOAD (Every frame)
   ┌─────────────────────────────────────────────────────────────┐
   │ if (term->pending_app_load != INVALID_HANDLE) {             │
   │     uint status = check_status(io_handles, term->pending_app_load); │
   │                                                              │
   │     if (status == STATUS_READY) {                           │
   │         // Initialize app from loaded bytes                  │
   │         uint slot = initialize_app_from_gpuapp(...);        │
   │                                                              │
   │         if (slot != INVALID_SLOT) {                         │
   │             terminal_print(term, "Launched in slot ");      │
   │             terminal_print_uint(term, slot);                │
   │             terminal_print(term, "\n");                     │
   │         } else {                                             │
   │             terminal_print(term, "Failed to initialize\n"); │
   │         }                                                     │
   │                                                              │
   │         term->pending_app_load = INVALID_HANDLE;            │
   │     } else if (status == STATUS_ERROR) {                    │
   │         terminal_print(term, "Load failed\n");              │
   │         term->pending_app_load = INVALID_HANDLE;            │
   │     }                                                         │
   │     // STATUS_LOADING: keep waiting                          │
   │ }                                                             │
   └─────────────────────────────────────────────────────────────┘
```

---

## GPU-Side Implementation (Metal)

### Filesystem Index Search (O(1))

```metal
// Search for app by name (uses GPU-resident filesystem index from Issue #135)
inline uint search_index_for_app(
    device FilesystemIndex* index,
    device const uchar* app_name,
    uint name_len
) {
    // Compute hash of "appname.gpuapp"
    uint hash = hash_string(app_name, name_len);
    hash = hash_combine(hash, hash_string(".gpuapp", 7));

    // O(1) hash table lookup
    uint bucket = hash & (index->bucket_count - 1);
    uint entry_idx = index->buckets[bucket];

    while (entry_idx != INVALID_ENTRY) {
        device FileEntry* entry = &index->entries[entry_idx];

        // Compare name hash (fast reject)
        if (entry->name_hash == hash) {
            // Full name compare if hash matches
            if (compare_names(entry, app_name, name_len)) {
                return entry_idx;  // Found!
            }
        }

        entry_idx = entry->next;  // Chain to next
    }

    return INVALID_PATH;  // Not found
}
```

### Terminal Print (Append to Output Buffer)

```metal
inline void terminal_print(device TerminalState* term, constant char* str) {
    uint len = 0;
    while (str[len] != 0 && len < 256) len++;

    uint start = term->output_length;
    if (start + len > OUTPUT_BUFFER_SIZE) {
        // Buffer full - scroll or wrap
        start = 0;
    }

    for (uint i = 0; i < len; i++) {
        term->output_buffer[start + i] = str[i];
    }
    term->output_length = start + len;
}

inline void terminal_print_uint(device TerminalState* term, uint value) {
    uchar buf[16];
    int i = 15;
    buf[i--] = 0;

    do {
        buf[i--] = '0' + (value % 10);
        value /= 10;
    } while (value > 0 && i >= 0);

    terminal_print(term, (constant char*)(buf + i + 1));
}
```

### Apps Command (List Available)

```metal
inline void handle_apps(
    device TerminalState* term,
    device FilesystemIndex* index
) {
    terminal_print(term, "Available apps:\n");

    // Scan index for .gpuapp files
    uint count = 0;
    for (uint i = 0; i < index->entry_count && count < 20; i++) {
        device FileEntry* entry = &index->entries[i];

        // Check if ends with .gpuapp
        if (entry_ends_with_gpuapp(entry)) {
            terminal_print(term, "  ");
            terminal_print_entry_name(term, entry);
            terminal_print(term, "\n");
            count++;
        }
    }

    if (count == 0) {
        terminal_print(term, "  (no apps found)\n");
    }
}
```

### Command Parsing (GPU String Processing)

```metal
inline void parse_command(
    device const uchar* input,
    uint input_len,
    device uchar* cmd_name,
    device uint* cmd_name_len,
    device uchar* cmd_args,
    device uint* cmd_args_len
) {
    uint i = 0;

    // Skip leading whitespace
    while (i < input_len && (input[i] == ' ' || input[i] == '\t')) i++;

    // Extract command name
    uint name_start = i;
    while (i < input_len && input[i] != ' ' && input[i] != '\t' && input[i] != '\n') {
        cmd_name[i - name_start] = input[i];
        i++;
    }
    *cmd_name_len = i - name_start;
    cmd_name[*cmd_name_len] = 0;

    // Skip whitespace
    while (i < input_len && (input[i] == ' ' || input[i] == '\t')) i++;

    // Rest is args
    uint args_start = i;
    while (i < input_len && input[i] != '\n') {
        cmd_args[i - args_start] = input[i];
        i++;
    }
    *cmd_args_len = i - args_start;
    cmd_args[*cmd_args_len] = 0;
}
```

---

## Integration with Megakernel

```metal
// In terminal_update (called every frame for terminal app)
inline void terminal_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device FilesystemIndex* fs_index,
    device PipelineState* io_state,
    device IORequest* io_queue,
    device FileHandle* io_handles,
    device uchar* content_pool,
    device AppTableHeader* app_table,
    device GpuAppDescriptor* all_apps,
    device InputEvent* input_events,
    uint input_count,
    uint tid,
    uint tg_size
) {
    device TerminalState* term = (device TerminalState*)(unified_state + app->state_offset);

    // Process keyboard input
    if (tid == 0) {
        terminal_process_input(term, input_events, input_count);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Handle command if ready
    if (tid == 0 && term->command_ready) {
        uchar cmd_name[32];
        uint cmd_name_len;
        uchar cmd_args[224];
        uint cmd_args_len;

        parse_command(term->current_command, term->command_length,
                     cmd_name, &cmd_name_len, cmd_args, &cmd_args_len);

        uint handler = lookup_command(cmd_name, cmd_name_len);

        switch (handler) {
            case HANDLER_LAUNCH:
                handle_launch(term, cmd_args, cmd_args_len,
                             fs_index, io_state, io_queue, io_handles);
                break;
            case HANDLER_APPS:
                handle_apps(term, fs_index);
                break;
            case HANDLER_CLEAR:
                term->output_length = 0;
                break;
            case HANDLER_HELP:
                terminal_print(term, "Commands: launch, apps, clear, help\n");
                break;
            default:
                terminal_print(term, "Unknown command: ");
                terminal_print(term, cmd_name);
                terminal_print(term, "\n");
        }

        term->command_ready = 0;
        term->command_length = 0;
    }

    // Poll pending app load
    if (tid == 0) {
        poll_pending_app_load(term, io_handles, content_pool,
                              app_table, all_apps, unified_state);
    }

    // Render terminal (all threads participate)
    terminal_render(app, term, unified_vertices, tid, tg_size);
}
```

---

## Test Plan

### Test 1: Command Parsing

```rust
#[test]
fn test_gpu_command_parsing() {
    let device = Device::system_default().unwrap();

    let test_cases = [
        ("launch myapp", "launch", "myapp"),
        ("  launch  myapp  ", "launch", "myapp"),
        ("apps", "apps", ""),
        ("help", "help", ""),
    ];

    for (input, expected_cmd, expected_args) in test_cases {
        let (cmd, args) = gpu_parse_command(&device, input);
        assert_eq!(cmd, expected_cmd);
        assert_eq!(args, expected_args);
    }
}
```

### Test 2: Command Hash Lookup

```rust
#[test]
fn test_command_hash_lookup() {
    let device = Device::system_default().unwrap();

    assert_eq!(gpu_lookup_command(&device, "launch"), HANDLER_LAUNCH);
    assert_eq!(gpu_lookup_command(&device, "apps"), HANDLER_APPS);
    assert_eq!(gpu_lookup_command(&device, "clear"), HANDLER_CLEAR);
    assert_eq!(gpu_lookup_command(&device, "unknown"), HANDLER_UNKNOWN);
}
```

### Test 3: App Search in Index

```rust
#[test]
fn test_search_for_gpuapp() {
    let device = Device::system_default().unwrap();

    // Create filesystem index with test apps
    let mut index = GpuFilesystemIndex::new(&device);
    index.add_entry("/apps/hello.gpuapp", 1024);
    index.add_entry("/apps/game.gpuapp", 2048);
    index.add_entry("/apps/readme.txt", 100);  // Not an app

    // GPU search
    let path_idx = gpu_search_for_app(&device, &index, "hello");
    assert_ne!(path_idx, INVALID_PATH);

    let path_idx = gpu_search_for_app(&device, &index, "game");
    assert_ne!(path_idx, INVALID_PATH);

    let path_idx = gpu_search_for_app(&device, &index, "readme");
    assert_eq!(path_idx, INVALID_PATH);  // Not .gpuapp

    let path_idx = gpu_search_for_app(&device, &index, "nonexistent");
    assert_eq!(path_idx, INVALID_PATH);
}
```

### Test 4: Full Launch Flow

```rust
#[test]
fn test_terminal_launch_flow() {
    let device = Device::system_default().unwrap();

    // Setup: Create .gpuapp file, populate index
    let app_path = create_test_gpuapp("testapp");
    let mut os = GpuOs::boot(&device);
    os.index_directory(&app_path.parent().unwrap());

    // Simulate typing "launch testapp" in terminal
    os.key_event(/* l */);
    os.key_event(/* a */);
    // ... more keys ...
    os.key_event(/* \n */);

    // Run frames until app loads
    for _ in 0..100 {
        os.run_frame();
        os.process_io();

        // Check if app appeared
        if os.app_count() > 4 {  // 4 system apps + 1 user app
            break;
        }
    }

    assert!(os.app_count() > 4);
}
```

### Test 5: Apps Command Lists Apps

```rust
#[test]
fn test_apps_command() {
    let device = Device::system_default().unwrap();

    // Create multiple .gpuapp files
    create_test_gpuapp("app1");
    create_test_gpuapp("app2");
    create_test_gpuapp("app3");

    let mut os = GpuOs::boot(&device);
    os.index_directory(test_apps_dir());

    // Type "apps" command
    simulate_type_command(&mut os, "apps\n");
    os.run_frame();

    // Check terminal output contains app names
    let output = os.terminal_output();
    assert!(output.contains("app1"));
    assert!(output.contains("app2"));
    assert!(output.contains("app3"));
}
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Command parse time | <100 GPU cycles |
| App search time | O(1) hash lookup |
| Launch latency | <15ms (I/O bound) |
| CPU involvement | 0 (except I/O thread) |

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `src/gpu_os/shaders/terminal.metal` | Terminal update with launch command |
| `src/gpu_os/gpu_app_system.rs` | Add terminal state, command handlers |
| `tests/test_issue_169_terminal_launch.rs` | Tests |

---

## Anti-Patterns Avoided

| Bad Pattern | Why Bad | Good Pattern |
|-------------|---------|--------------|
| CPU parses command | CPU in loop | GPU string processing |
| CPU searches filesystem | CPU work | GPU hash table lookup |
| CPU loads app | CPU coordinates | GPU queues, CPU async I/O |
| CPU notifies GPU | Round-trip | GPU polls status |
| String comparison loops | SIMD divergence | Hash-based lookup |
