# GPU Infrastructure Map for Megakernel Apps

## Available Infrastructure

### File I/O (NO CPU IN HOT PATH)

| Capability | File | Use Case |
|------------|------|----------|
| Zero-copy mmap | `mmap_buffer.rs` | Load any file to GPU with no copies |
| GPU-direct I/O | `gpu_io.rs` | MTLIOCommandQueue bypasses CPU entirely |
| Batch I/O | `batch_io.rs` | Load 1000s of files in single command |
| Streaming I/O | `streaming_search.rs` | Search while loading (quad-buffered) |

### Filesystem Index (GPU-RESIDENT)

| Capability | File | Use Case |
|------------|------|----------|
| GPU index | `gpu_index.rs` | 3M+ paths searchable by GPU |
| Shared index | `shared_index.rs` | <10ms load via mmap, shared across tools |

### Text & Documents

| Capability | File | Use Case |
|------------|------|----------|
| Bitmap font | `text_render.rs` | Terminal, file browser text |
| Document pipeline | `document/` | HTML-like: tokenize→parse→style→layout→paint |
| Text shaping | `document/text.metal` | Wrapping, line breaking, baseline alignment |

### Memory Management (GPU-PARALLEL)

| Capability | File | Use Case |
|------------|------|----------|
| Parallel allocator | `parallel_alloc.rs` | 1024 allocations in 600ns (O(log N)) |
| Parallel compaction | `parallel_compact.rs` | Scatter→contiguous via Blelloch scan |

### Input & Events (GPU-DRIVEN)

| Capability | File | Use Case |
|------------|------|----------|
| Input pipeline | `input.rs` | HID→GPU ring buffer, click detection |
| Event dispatch | `event_loop.rs` | GPU hit-test, window focus, all interactions |

### Persistent Execution

| Capability | File | Use Case |
|------------|------|----------|
| Work queue | `work_queue.rs` | GPU pulls work, decides what to do |
| Persistent kernel | `persistent_search.rs` | GPU runs continuously, polls for work |
| GPU cache | `gpu_cache.rs` | LRU cache with cuckoo hash, GPU-managed |

### String Processing (GPU-PARALLEL)

| Capability | File | Use Case |
|------------|------|----------|
| Tokenization | `gpu_string.rs` | "foo BAR" → ["foo","bar"] on GPU |
| Path parsing | `gpu_string.rs` | Extract filename/extension/depth |

### Graphics

| Capability | File | Use Case |
|------------|------|----------|
| Vector paths | `vector/` | Bezier tessellation, gradients |
| Hybrid render | `render.rs` | Compute vertices → fragment pixels |

---

## App-to-Infrastructure Mapping

### Terminal App → Uses:
- `input.rs` - Keyboard events from GPU ring buffer
- `text_render.rs` - Render text output
- `gpu_string.rs` - Tokenize commands on GPU
- `gpu_io.rs` - Execute file operations GPU-direct
- `shared_index.rs` - File completion via GPU search

### File Browser App → Uses:
- `shared_index.rs` - Browse filesystem from GPU index
- `gpu_io.rs` - Load file contents GPU-direct
- `text_render.rs` - Render filenames
- `event_loop.rs` - Click/scroll handling
- `mmap_buffer.rs` - Preview file contents zero-copy

### Dock App → Uses:
- `input.rs` - Mouse hover/click from ring buffer
- `event_loop.rs` - Hit testing for icons
- `render.rs` - Generate icon quads

### MenuBar App → Uses:
- `input.rs` - Mouse events
- `event_loop.rs` - Menu hit testing
- `text_render.rs` - Menu item text

### Window Chrome App → Uses:
- `input.rs` - Drag/resize from ring buffer
- `event_loop.rs` - Title bar hit testing
- `render.rs` - Chrome geometry generation

### Compositor → Uses:
- `render.rs` - Unified vertex buffer concept
- All apps write to same buffer with depth values
- Single draw call, hardware z-test

---

## Key Anti-Patterns

### WRONG: CPU Command Execution
```rust
// BAD - CPU in hot path
fn execute_command(&self, cmd: &str) {
    std::fs::read_dir(path)  // CPU filesystem!
}
```

### RIGHT: GPU-Direct Execution
```rust
// GOOD - GPU owns everything
fn execute_command(&self, cmd: &str) {
    // Parse command on GPU (gpu_string.rs)
    // Search filesystem on GPU (shared_index.rs)
    // Load files GPU-direct (gpu_io.rs)
    // Render results on GPU (text_render.rs)
}
```

### WRONG: CPU Text Processing
```rust
// BAD - CPU tokenization
let words: Vec<&str> = query.split_whitespace().collect();
```

### RIGHT: GPU Tokenization
```metal
// GOOD - GPU parallel tokenization
kernel void tokenize_query(
    device uchar* query,
    device GpuWord* words,
    device TokenizeResult* result
) {
    // Each thread handles portion of string
}
```

---

## Infrastructure Integration Points

### For Megakernel Apps

1. **State Buffer**: Apps get unified_state offset from GpuAppDescriptor
2. **Vertex Buffer**: Apps write to unified_vertices at vertex_offset
3. **Input Events**: Apps read from shared InputQueue
4. **Filesystem**: Apps use shared GpuFilesystemIndex
5. **File Loading**: Apps use GpuIOQueue for async loads
6. **Text Rendering**: Apps call write_glyph() helper

### Shared Resources in Megakernel

```metal
// All apps have access to:
device uchar* unified_state,          // All app state
device RenderVertex* unified_vertices, // All app vertices
device InputEvent* input_queue,       // HID events
device GpuPathEntry* fs_index,        // Filesystem (3M+ entries)
device GpuIOCommandBuffer* io_queue,  // File loading queue
device GpuWord* string_scratch        // String processing workspace
```

---

## Success Metrics

| Metric | CPU Pattern | GPU Pattern |
|--------|-------------|-------------|
| File list | 100ms (read_dir) | <1ms (index search) |
| Command parse | 10μs (CPU split) | <100ns (GPU parallel) |
| File load | Round-trip | Zero-copy mmap |
| Event dispatch | CPU callback | GPU hit-test |
| Text render | CPU layout | GPU prefix-sum |
