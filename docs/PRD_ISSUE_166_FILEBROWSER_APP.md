# PRD: File Browser as Megakernel App (Issue #166)

## Overview

Port the file browser to run as a megakernel app, using existing GPU-resident filesystem index, GPU-direct I/O, and document layout engine.

## Infrastructure Used

| Need | Infrastructure | File |
|------|---------------|------|
| File listing | GPU-resident index | `shared_index.rs`, `gpu_index.rs` |
| File loading | GPU-direct I/O | `gpu_io.rs`, `batch_io.rs` |
| File preview | Zero-copy mmap | `mmap_buffer.rs` |
| Layout | Level-parallel flexbox | `document/layout.metal` |
| Scrolling | Prefix sum positioning | `parallel_compact.rs` |
| Text | Bitmap font | `text_render.rs` |
| Click handling | GPU event dispatch | `event_loop.rs` |

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| `std::fs::read_dir()` | `shared_index.search()` |
| CPU scroll calculation | Prefix sum for visible items |
| CPU click detection | `event_loop.rs` hit testing |
| CPU file type detection | Index flags (FLAG_IS_DIR, etc.) |

## Design

### FileBrowser State Structure

```metal
struct FileBrowserState {
    uint current_dir_idx;       // Index in fs_index
    uint entry_count;           // Files in current dir
    uint scroll_offset;         // First visible item
    uint visible_count;         // Items that fit in view
    uint selected_idx;          // Currently selected file
    uint hovered_idx;           // Mouse hover target
    float scroll_y;             // Smooth scroll position
    float target_scroll_y;      // Scroll target for animation

    // Preview state
    uint preview_file_idx;      // File being previewed
    uint preview_io_handle;     // GPU-direct I/O handle
    uint preview_status;        // IOStatus

    // View settings
    uint view_mode;             // LIST, GRID, COLUMNS
    float item_height;
    float item_width;

    uint _pad[2];
    // uint visible_entries[MAX_VISIBLE] follows - indices into fs_index
    // PreviewBuffer preview follows
};

constant uint VIEW_LIST = 0;
constant uint VIEW_GRID = 1;
constant uint VIEW_COLUMNS = 2;
```

### FileBrowser Update - Uses Infrastructure

```metal
inline void filebrowser_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuPathEntry* fs_index,           // SHARED_INDEX.RS
    uint fs_index_count,
    device GpuEventLoopState* event_state,   // EVENT_LOOP.RS
    device GpuIOCommandBuffer* io_queue,     // GPU_IO.RS
    uint tid,
    uint tg_size
) {
    device FileBrowserState* state = (device FileBrowserState*)(unified_state + app->state_offset);
    device uint* visible = (device uint*)(state + 1);

    // Thread 0: Update state based on events (EVENT_LOOP.RS INFRASTRUCTURE)
    if (tid == 0) {
        // Check for navigation events from event_loop
        if (event_state->dispatch_target == DISPATCH_APP_INPUT &&
            event_state->target_app == app->slot) {

            // Handle click
            if (event_state->click_detected) {
                float click_y = event_state->last_mouse_pos.y - app->window_y;
                uint clicked_item = (uint)(click_y / state->item_height) + state->scroll_offset;

                if (clicked_item < state->entry_count) {
                    GpuPathEntry entry = fs_index[visible[clicked_item]];

                    if (entry.flags & FLAG_IS_DIR) {
                        // Navigate into directory
                        state->current_dir_idx = visible[clicked_item];
                        state->scroll_offset = 0;
                        state->selected_idx = 0;
                        // Will rebuild visible list on next frame
                    } else {
                        // Select file and start preview
                        state->selected_idx = clicked_item;
                        state->preview_file_idx = visible[clicked_item];

                        // Queue GPU-direct I/O for preview (GPU_IO.RS)
                        state->preview_io_handle = io_queue_load(io_queue, entry.path);
                        state->preview_status = IO_PENDING;
                    }
                }
            }

            // Handle scroll
            if (event_state->scroll_delta_y != 0) {
                state->target_scroll_y -= event_state->scroll_delta_y * 20.0;
                state->target_scroll_y = max(0.0f, state->target_scroll_y);
            }
        }

        // Smooth scroll animation
        float diff = state->target_scroll_y - state->scroll_y;
        state->scroll_y += diff * 0.2;
        state->scroll_offset = (uint)(state->scroll_y / state->item_height);

        // Check preview I/O status (GPU_IO.RS)
        if (state->preview_status == IO_PENDING) {
            IOStatus status = io_get_status(io_queue, state->preview_io_handle);
            if (status == IO_COMPLETE) {
                state->preview_status = IO_COMPLETE;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Parallel: Build visible entries list from fs_index (SHARED_INDEX.RS)
    // Each thread checks a portion of the index
    uint entries_per_thread = (fs_index_count + tg_size - 1) / tg_size;
    uint start = tid * entries_per_thread;
    uint end = min(start + entries_per_thread, fs_index_count);

    // Use atomic to append matching entries
    for (uint i = start; i < end; i++) {
        if (fs_index[i].parent_idx == state->current_dir_idx) {
            uint slot = atomic_fetch_add_explicit(
                (device atomic_uint*)&state->entry_count, 1, memory_order_relaxed);
            if (slot < MAX_VISIBLE) {
                visible[slot] = i;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Parallel: Generate vertices for visible items (TEXT_RENDER.RS infrastructure)
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    uint visible_start = state->scroll_offset;
    uint visible_end = min(visible_start + state->visible_count, state->entry_count);
    uint items_to_render = visible_end - visible_start;

    uint items_per_thread = (items_to_render + tg_size - 1) / tg_size;
    uint my_start = tid * items_per_thread;
    uint my_end = min(my_start + items_per_thread, items_to_render);

    for (uint i = my_start; i < my_end; i++) {
        uint entry_idx = visible[visible_start + i];
        GpuPathEntry entry = fs_index[entry_idx];

        float y = i * state->item_height;
        float depth = app->window_depth;

        // Background (selection/hover)
        float4 bg_color = float4(0.0);
        if (visible_start + i == state->selected_idx) {
            bg_color = float4(0.2, 0.4, 0.8, 1.0); // Selected
        } else if (visible_start + i == state->hovered_idx) {
            bg_color = float4(0.3, 0.3, 0.3, 0.5); // Hovered
        }

        uint vert_base = i * (6 + 6 * 32); // Background + up to 32 chars

        if (bg_color.a > 0) {
            write_quad(verts + vert_base, float2(0, y),
                float2(app->window_width, state->item_height),
                depth, bg_color);
        }

        // Icon (folder or file)
        float4 icon_color = (entry.flags & FLAG_IS_DIR)
            ? float4(0.3, 0.6, 0.9, 1.0)  // Folder blue
            : float4(0.7, 0.7, 0.7, 1.0); // File gray

        write_quad(verts + vert_base + 6, float2(4, y + 2),
            float2(16, 16), depth, icon_color);

        // Filename text (using existing glyph infrastructure)
        float text_x = 24.0;
        for (uint c = 0; c < 32 && entry.path[c] != 0; c++) {
            // Find filename portion of path
            // (In practice, extract from GpuPathEntry)
            write_glyph(verts + vert_base + 12 + c * 6,
                float2(text_x + c * 8.0, y + 2.0),
                entry.path[c], depth, float4(0.9, 0.9, 0.9, 1.0));
        }
    }

    // Thread 0: Set vertex count
    if (tid == 0) {
        app->vertex_count = items_to_render * (6 + 6 * 32);
    }
}
```

### Directory Navigation - All GPU

```metal
inline void navigate_to_directory(
    device FileBrowserState* state,
    device GpuPathEntry* fs_index,
    uint new_dir_idx
) {
    state->current_dir_idx = new_dir_idx;
    state->scroll_offset = 0;
    state->scroll_y = 0;
    state->target_scroll_y = 0;
    state->selected_idx = UINT_MAX;
    state->entry_count = 0; // Will be rebuilt on next frame
}
```

### Rust Integration - Minimal CPU

```rust
impl GpuOs {
    /// File browser uses GPU infrastructure exclusively
    pub fn create_file_browser(&mut self) -> Option<u32> {
        let slot = self.system.launch_by_type(app_type::FILEBROWSER)?;

        // Initial directory from shared index (already GPU-resident)
        let home_idx = self.shared_index.find_directory("~")?;
        self.set_filebrowser_directory(slot, home_idx);

        Some(slot)
    }

    /// Open file in appropriate app (GPU decides type from index flags)
    pub fn filebrowser_open_selected(&mut self, slot: u32) {
        let state = self.get_filebrowser_state(slot)?;
        let entry_idx = state.visible_entries[state.selected_idx];
        let entry = self.shared_index.get_entry(entry_idx)?;

        // File type already in index - no CPU stat needed
        let app_type = if entry.flags & FLAG_IS_TEXT {
            app_type::EDITOR
        } else if entry.flags & FLAG_IS_IMAGE {
            app_type::VIEWER
        } else {
            app_type::PREVIEW
        };

        self.launch_app_with_file(app_type, entry_idx);
    }
}
```

## Tests

```rust
#[test]
fn test_filebrowser_uses_shared_index() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();

    // Should have entries from shared index
    os.system.run_frame();

    let state = os.get_filebrowser_state(browser).unwrap();
    assert!(state.entry_count > 0);
}

#[test]
fn test_filebrowser_navigation() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();
    os.system.run_frame();

    // Find a directory in the index
    let dir_idx = os.shared_index.find_directory("src").unwrap();

    // Navigate to it
    os.set_filebrowser_directory(browser, dir_idx);
    os.system.run_frame();

    let state = os.get_filebrowser_state(browser).unwrap();
    assert_eq!(state.current_dir_idx, dir_idx);
}

#[test]
fn test_filebrowser_preview_gpu_direct() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();
    os.system.run_frame();

    // Select a file
    os.system.queue_input(InputEvent::mouse_click(100.0, 50.0, 0));
    os.system.process_input();
    os.system.run_frame();

    // Preview should be loading via GPU-direct I/O
    let state = os.get_filebrowser_state(browser).unwrap();
    assert!(state.preview_io_handle != INVALID_HANDLE);
}

#[test]
fn test_filebrowser_scroll() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();
    os.system.run_frame();

    let initial_offset = os.get_filebrowser_state(browser).unwrap().scroll_offset;

    // Scroll down
    os.system.queue_input(InputEvent::scroll(0.0, -60.0));
    os.system.process_input();

    // Run frames for smooth scroll
    for _ in 0..10 {
        os.system.run_frame();
    }

    let new_offset = os.get_filebrowser_state(browser).unwrap().scroll_offset;
    assert!(new_offset > initial_offset);
}
```

## Benchmarks

```rust
#[test]
fn bench_filebrowser_large_directory() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();

    // Navigate to a directory with many files
    // (The index already contains all files)
    let large_dir = os.shared_index.find_directory("node_modules").unwrap_or(0);
    os.set_filebrowser_directory(browser, large_dir);

    let start = Instant::now();
    for _ in 0..100 {
        os.system.mark_dirty(browser);
        os.system.run_frame();
    }
    let duration = start.elapsed();

    println!("Large directory render: {}us/frame", duration.as_micros() / 100);
    // Should be <1ms - GPU parallel entry building
}

#[test]
fn bench_filebrowser_scroll_1000_files() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let browser = os.launch_app(app_type::FILEBROWSER).unwrap();
    os.system.run_frame();

    let start = Instant::now();
    for i in 0..1000 {
        // Continuous scrolling
        os.system.queue_input(InputEvent::scroll(0.0, -3.0));
        os.system.process_input();
        os.system.run_frame();
    }
    let duration = start.elapsed();

    println!("Scroll 1000 frames: {:?} ({:.0} FPS)",
        duration, 1000.0 / duration.as_secs_f64());
}
```

## Success Metrics

1. **<1ms directory listing**: GPU parallel index search
2. **GPU-direct preview**: No CPU for file loading
3. **60 FPS scrolling**: Smooth scroll with GPU animation
4. **Zero CPU in navigation**: All directory changes via index lookup
5. **Parallel entry building**: All threads contribute to visible list
