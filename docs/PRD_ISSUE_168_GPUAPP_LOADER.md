# PRD: GPU App Loader - Load .gpuapp from Filesystem

**Issue**: #168 - GPU App Loader
**Priority**: Critical (Apps must load from disk)
**Status**: Design

---

## THE GPU IS THE COMPUTER

**The CPU does NOT load apps. The CPU is an I/O coprocessor that fetches bytes.**

### CPU-Centric Thinking (WRONG)
```
CPU: parse manifest.toml
CPU: allocate buffers
CPU: copy bytecode to GPU
CPU: initialize app descriptor
CPU: tell GPU "app is ready"
GPU: "thanks CPU, running now"
```

### GPU-Native Thinking (RIGHT)
```
GPU: writes load request to I/O queue (path_idx, destination)
GPU: continues running other apps
GPU: polls handle status
GPU: when READY, parses header directly from buffer
GPU: initializes app slot atomically
GPU: app runs on next frame

Meanwhile (async):
CPU I/O thread: sees request
CPU I/O thread: dispatches MTLIOCommandQueue
Metal: loads bytes directly to GPU buffer
CPU I/O thread: updates status = READY
```

**The GPU parses, validates, and initializes. CPU just moves bytes.**

---

## Architecture

### .gpuapp File Format (GPU-Parseable)

```
┌─────────────────────────────────────────────────────────────────┐
│ GpuAppFileHeader (64 bytes) - Fixed, GPU reads directly         │
│   magic: [u8; 6] = "GPUAPP"                                     │
│   version: u16 = 1                                              │
│   flags: u32                                                    │
│   code_offset: u32 (offset to bytecode from file start)         │
│   code_size: u32 (number of instructions)                       │
│   state_size: u32 (bytes needed for app state)                  │
│   vertex_budget: u32 (max vertices)                             │
│   thread_count: u32 (recommended threads)                       │
│   entry_point: u32 (starting PC)                                │
│   name: [u8; 24] (null-terminated)                              │
│   _reserved: [u8; 8]                                            │
├─────────────────────────────────────────────────────────────────┤
│ Bytecode Instructions (8 bytes each)                            │
│   [BytecodeInst; code_size]                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Why this format?**
- Fixed-size header = O(1) GPU parse
- No string parsing, no TOML, no JSON
- GPU can validate magic bytes directly
- All offsets are byte offsets from file start

### Memory Layout

```
Content Pipeline Buffer (from Issue #165):
┌──────────────────────────────────────────────────────────────┐
│ Content Pool                                                   │
│   [FileHandle 0] → .gpuapp file bytes loaded here             │
│   [FileHandle 1] → another .gpuapp                            │
│   ...                                                          │
└──────────────────────────────────────────────────────────────┘

App System Buffers:
┌──────────────────────────────────────────────────────────────┐
│ unified_state                                                  │
│   [App 0 state: BytecodeHeader + instructions + app data]     │
│   [App 1 state: ...]                                          │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. GPU REQUESTS LOAD
   ┌─────────────────────────────────────────────────────────────┐
   │ Terminal/Dock receives "launch myapp" command               │
   │                                                              │
   │ // In megakernel (GPU):                                      │
   │ uint path_idx = search_filesystem_index("myapp.gpuapp");    │
   │ if (path_idx != INVALID) {                                  │
   │     uint handle = request_read(io_state, path_idx, ...);    │
   │     pending_app_loads[slot] = handle;                       │
   │ }                                                            │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. CPU I/O COPROCESSOR (async, never blocks GPU)
   ┌─────────────────────────────────────────────────────────────┐
   │ // On dedicated thread:                                      │
   │ process_requests();  // Dispatches MTLIOCommandQueue        │
   │ // File bytes land directly in content_pool                 │
   │ // Status updated to READY                                   │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. GPU INITIALIZES APP
   ┌─────────────────────────────────────────────────────────────┐
   │ // In megakernel (GPU), polling:                             │
   │ uint status = check_status(io_handles, handle);             │
   │ if (status == STATUS_READY) {                               │
   │     device uchar* file_data = content_pool + handle.offset; │
   │     device GpuAppFileHeader* hdr = (GpuAppFileHeader*)file_data; │
   │                                                              │
   │     // Validate                                              │
   │     if (!validate_gpuapp_header(hdr)) { error(); return; }  │
   │                                                              │
   │     // Allocate app slot (atomic)                           │
   │     uint slot = allocate_app_slot(app_table);               │
   │                                                              │
   │     // Copy bytecode to app state                           │
   │     copy_bytecode_to_state(file_data, apps[slot]);          │
   │                                                              │
   │     // Initialize descriptor                                 │
   │     apps[slot].app_type = APP_TYPE_BYTECODE;                │
   │     apps[slot].flags = ACTIVE | VISIBLE | DIRTY;            │
   │                                                              │
   │     // App runs on next frame!                               │
   │ }                                                            │
   └─────────────────────────────────────────────────────────────┘
```

---

## GPU-Side Implementation (Metal)

### Header Validation (O(1))

```metal
inline bool validate_gpuapp_header(device GpuAppFileHeader* header) {
    // Check magic (unrolled, no loop)
    if (header->magic[0] != 'G') return false;
    if (header->magic[1] != 'P') return false;
    if (header->magic[2] != 'U') return false;
    if (header->magic[3] != 'A') return false;
    if (header->magic[4] != 'P') return false;
    if (header->magic[5] != 'P') return false;

    // Check version
    if (header->version != 1) return false;

    // Check sizes are reasonable
    if (header->code_size > MAX_BYTECODE_SIZE) return false;
    if (header->state_size > MAX_APP_STATE) return false;
    if (header->vertex_budget > MAX_APP_VERTICES) return false;

    return true;
}
```

### App Initialization (Parallel Copy)

```metal
// Called when file load completes - initializes app from loaded bytes
inline void initialize_app_from_gpuapp(
    device AppTableHeader* table,
    device GpuAppDescriptor* apps,
    device uchar* unified_state,
    device uchar* file_data,
    uint file_size,
    uint tid,
    uint tg_size
) {
    device GpuAppFileHeader* header = (device GpuAppFileHeader*)file_data;

    // Thread 0 validates and allocates
    if (tid == 0) {
        if (!validate_gpuapp_header(header)) {
            // Signal error somehow
            return;
        }

        // Allocate slot (atomic bitmap)
        uint slot = allocate_app_slot_atomic(table);
        if (slot == INVALID_SLOT) return;

        // Store slot for other threads
        // (use threadgroup memory or atomic)
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads participate in copy
    // Source: file bytecode section
    device uchar* src = file_data + header->code_offset;
    uint copy_size = header->code_size * sizeof(BytecodeInst);

    // Destination: app state buffer
    device uchar* dst = unified_state + apps[slot].state_offset + sizeof(BytecodeHeader);

    // Parallel copy: each thread copies a chunk
    uint bytes_per_thread = (copy_size + tg_size - 1) / tg_size;
    uint my_start = tid * bytes_per_thread;
    uint my_end = min(my_start + bytes_per_thread, copy_size);

    for (uint i = my_start; i < my_end; i++) {
        dst[i] = src[i];
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 finalizes descriptor
    if (tid == 0) {
        apps[slot].app_type = APP_TYPE_BYTECODE;
        apps[slot].flags = ACTIVE_FLAG | VISIBLE_FLAG | DIRTY_FLAG;
        apps[slot].vertex_budget = header->vertex_budget;
        // ... other fields

        // Write BytecodeHeader at start of state
        device BytecodeHeader* bc_header = (device BytecodeHeader*)(unified_state + apps[slot].state_offset);
        bc_header->code_size = header->code_size;
        bc_header->entry_point = header->entry_point;
        bc_header->vertex_budget = header->vertex_budget;
    }
}
```

### Pending Load Tracking

```metal
// In system state, track pending app loads
struct PendingAppLoad {
    uint io_handle;      // From content pipeline
    uint requesting_app; // Who requested (terminal, dock)
    uint target_slot;    // Pre-allocated slot (or INVALID)
    uint flags;
};

#define MAX_PENDING_LOADS 16

// In megakernel, poll pending loads
inline void poll_pending_app_loads(
    device PendingAppLoad* pending,
    device atomic_uint* pending_count,
    device FileHandle* io_handles,
    device uchar* content_pool,
    device AppTableHeader* app_table,
    device GpuAppDescriptor* apps,
    device uchar* unified_state,
    uint tid,
    uint tg_size
) {
    uint count = atomic_load_explicit(pending_count, memory_order_relaxed);
    if (count == 0) return;

    // Each pending load checked by thread 0
    // (Could parallelize if many pending)
    if (tid != 0) return;

    for (uint i = 0; i < count; i++) {
        uint handle = pending[i].io_handle;
        uint status = check_status(io_handles, handle);

        if (status == STATUS_READY) {
            // Initialize app
            uint offset = io_handles[handle].content_offset;
            uint size = io_handles[handle].file_size;
            device uchar* file_data = content_pool + offset;

            initialize_app_from_gpuapp(
                app_table, apps, unified_state,
                file_data, size,
                tid, tg_size
            );

            // Remove from pending (swap with last)
            pending[i] = pending[count - 1];
            atomic_fetch_sub_explicit(pending_count, 1, memory_order_relaxed);
            i--;  // Recheck this slot
            count--;
        } else if (status == STATUS_ERROR) {
            // Handle error - remove from pending
            pending[i] = pending[count - 1];
            atomic_fetch_sub_explicit(pending_count, 1, memory_order_relaxed);
            i--;
            count--;
        }
        // STATUS_LOADING: keep waiting
    }
}
```

---

## Rust-Side Implementation

### GpuAppLoader (Minimal CPU Role)

```rust
/// GPU App Loader - CPU just provides I/O, GPU does everything else
pub struct GpuAppLoader {
    content_pipeline: Arc<ContentPipeline>,

    // Pending loads buffer (GPU-owned)
    pending_loads_buffer: Buffer,
    pending_count_buffer: Buffer,
}

impl GpuAppLoader {
    /// Register a .gpuapp path for loading
    /// Returns path_idx for GPU to use
    pub fn register_app_path(&mut self, path: PathBuf) -> u32 {
        self.content_pipeline.register_path(path)
    }

    /// Called by I/O coprocessor thread
    pub fn process_io(&self) {
        self.content_pipeline.process_requests();
    }

    /// Get buffers for megakernel binding
    pub fn bind_to_encoder(&self, encoder: &ComputeCommandEncoderRef, base: u64) {
        encoder.set_buffer(base, Some(&self.pending_loads_buffer), 0);
        encoder.set_buffer(base + 1, Some(&self.pending_count_buffer), 0);
        // Content pipeline buffers bound separately
    }
}
```

### .gpuapp File Builder (Development Tool)

```rust
/// Build .gpuapp files from bytecode
pub struct GpuAppBuilder {
    name: String,
    bytecode: Vec<BytecodeInst>,
    state_size: u32,
    vertex_budget: u32,
}

impl GpuAppBuilder {
    pub fn new(name: &str) -> Self { ... }

    pub fn add_instruction(&mut self, inst: BytecodeInst) { ... }

    pub fn set_state_size(&mut self, size: u32) { ... }

    pub fn build(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Write header
        let header = GpuAppFileHeader {
            magic: *b"GPUAPP",
            version: 1,
            flags: 0,
            code_offset: 64,  // Right after header
            code_size: self.bytecode.len() as u32,
            state_size: self.state_size,
            vertex_budget: self.vertex_budget,
            thread_count: 256,
            entry_point: 0,
            name: self.name_bytes(),
            _reserved: [0; 8],
        };

        data.extend_from_slice(header.as_bytes());

        // Write bytecode
        for inst in &self.bytecode {
            data.extend_from_slice(inst.as_bytes());
        }

        data
    }

    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.build())
    }
}
```

---

## Test Plan

### Test 1: Header Validation

```rust
#[test]
fn test_gpuapp_header_validation() {
    let device = Device::system_default().unwrap();

    // Valid header
    let valid = create_valid_gpuapp_header();
    assert!(gpu_validate_header(&device, &valid));

    // Invalid magic
    let mut bad_magic = valid.clone();
    bad_magic.magic[0] = 'X';
    assert!(!gpu_validate_header(&device, &bad_magic));

    // Invalid version
    let mut bad_version = valid.clone();
    bad_version.version = 99;
    assert!(!gpu_validate_header(&device, &bad_version));
}
```

### Test 2: Load from Content Pipeline

```rust
#[test]
fn test_load_gpuapp_from_pipeline() {
    let device = Device::system_default().unwrap();

    // Create test .gpuapp file
    let mut builder = GpuAppBuilder::new("test_app");
    builder.add_instruction(BytecodeInst::loadi(4, 100.0));
    builder.add_instruction(BytecodeInst::halt());

    let temp_path = create_temp_gpuapp(&builder);

    // Register with content pipeline
    let mut loader = GpuAppLoader::new(&device);
    let path_idx = loader.register_app_path(temp_path);

    // GPU requests load
    let handle = gpu_request_app_load(&device, &loader, path_idx);

    // Process I/O
    loader.process_io();

    // GPU polls and initializes
    let app_slot = gpu_poll_and_initialize(&device, &loader, handle);

    // Verify app was created
    let app = read_app_descriptor(&device, app_slot);
    assert_eq!(app.app_type, APP_TYPE_BYTECODE);
    assert!(app.flags & ACTIVE_FLAG != 0);
}
```

### Test 3: Run Loaded App

```rust
#[test]
fn test_run_loaded_gpuapp() {
    let device = Device::system_default().unwrap();

    // Create .gpuapp that draws a quad
    let mut builder = GpuAppBuilder::new("quad_app");
    builder.add_instruction(BytecodeInst::setx(4, 100.0));
    builder.add_instruction(BytecodeInst::sety(4, 100.0));
    builder.add_instruction(BytecodeInst::setx(5, 50.0));
    builder.add_instruction(BytecodeInst::sety(5, 50.0));
    builder.add_instruction(BytecodeInst::setx(6, 1.0));  // Red
    builder.add_instruction(BytecodeInst::setw(6, 1.0));  // Alpha
    builder.add_instruction(BytecodeInst::quad(4, 5, 6, 0.5));
    builder.add_instruction(BytecodeInst::halt());
    builder.set_vertex_budget(6);

    // Load and run
    let app_slot = load_and_run_gpuapp(&device, &builder);

    // Run one frame
    run_megakernel_frame(&device);

    // Verify vertices generated
    let app = read_app_descriptor(&device, app_slot);
    assert_eq!(app.vertex_count, 6);
}
```

### Test 4: Batch Load Multiple Apps

```rust
#[test]
fn test_batch_load_apps() {
    let device = Device::system_default().unwrap();

    // Create 10 test apps
    let paths: Vec<_> = (0..10).map(|i| {
        let mut builder = GpuAppBuilder::new(&format!("app_{}", i));
        builder.add_instruction(BytecodeInst::halt());
        create_temp_gpuapp(&builder)
    }).collect();

    // Register all
    let mut loader = GpuAppLoader::new(&device);
    let indices: Vec<_> = paths.iter().map(|p| loader.register_app_path(p.clone())).collect();

    // GPU requests all loads
    let handles = gpu_batch_request_loads(&device, &loader, &indices);

    // Process I/O
    for _ in 0..100 {
        loader.process_io();
        std::thread::sleep(Duration::from_millis(10));
    }

    // All should be loaded
    let loaded_count = gpu_count_ready_handles(&device, &loader, &handles);
    assert_eq!(loaded_count, 10);
}
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| CPU involvement in app init | 0 (after I/O) |
| Header validation | O(1), <10 GPU cycles |
| Bytecode copy | Parallel, all threads |
| Load latency (small app) | <10ms |
| Batch load (10 apps) | <50ms |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/gpu_app_loader.rs` | GpuAppLoader, GpuAppBuilder |
| `src/gpu_os/shaders/gpu_app_loader.metal` | Validation, initialization kernels |
| `tests/test_issue_168_gpuapp_loader.rs` | Tests |

---

## Anti-Patterns Avoided

| Bad Pattern | Why Bad | Good Pattern |
|-------------|---------|--------------|
| CPU parses manifest | CPU in critical path | GPU parses fixed binary header |
| CPU allocates app slot | CPU coordinates | GPU atomic bitmap allocation |
| CPU copies bytecode | CPU moves bytes | GPU parallel copy, or DMA direct |
| CPU tells GPU "ready" | CPU orchestrates | GPU polls status buffer |
| Sequential app init | Slow | Parallel bytecode copy |
