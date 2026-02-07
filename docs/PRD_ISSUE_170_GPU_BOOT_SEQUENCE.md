# PRD: GPU Boot Sequence - Automatic App Discovery and Loading

**Issue**: #170 - GPU Boot Sequence
**Priority**: High (System startup)
**Status**: Design
**Depends On**: #168 (GPU App Loader), #135 (Shared Filesystem Index)

---

## THE GPU IS THE COMPUTER

**The CPU does NOT orchestrate boot. The CPU provides buffers and paths, then gets out of the way.**

### CPU-Centric Thinking (WRONG)
```
CPU: create Metal device
CPU: allocate buffers
CPU: scan ~/apps/ directory
CPU: for each .gpuapp file:
CPU:   load file
CPU:   parse header
CPU:   allocate app slot
CPU:   initialize app
CPU: tell GPU "boot complete"
GPU: "finally, I can start"
```

### GPU-Native Thinking (RIGHT)
```
CPU: create Metal device
CPU: allocate buffers
CPU: register ~/apps/ path
CPU: dispatch boot kernel
CPU: start I/O coprocessor thread
... CPU done with boot, now just services I/O ...

GPU boot kernel:
  1. Initialize system state (atomic)
  2. Launch system apps (Compositor, Dock, MenuBar, Chrome)
  3. Scan filesystem index for .gpuapp files
  4. Batch queue all app loads
  5. Poll for completions, initialize apps as they arrive
  6. Boot complete - normal megakernel loop begins
```

**CPU sets up memory and starts I/O thread. GPU does ALL the work.**

---

## Architecture

### Boot Phases

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0: CPU Setup (One-time, before GPU takes over)           │
│                                                                 │
│   - Create Metal device                                         │
│   - Allocate all buffers (app table, state, vertices, etc.)   │
│   - Register filesystem paths (~/apps/, etc.)                   │
│   - Compile shaders                                             │
│   - Start I/O coprocessor thread                                │
│   - Dispatch GPU boot kernel                                    │
│                                                                 │
│   Time: ~100ms (acceptable, happens once)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: GPU System Init (boot_phase1 kernel)                  │
│                                                                 │
│   - Initialize app table header (atomic)                        │
│   - Initialize all app descriptors to INACTIVE                  │
│   - Initialize unified state buffer                             │
│   - Initialize render state                                     │
│                                                                 │
│   Threads: max_slots (64)                                       │
│   Time: <1ms                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: GPU System Apps (boot_phase2 kernel)                  │
│                                                                 │
│   - Allocate Compositor slot (slot 0)                          │
│   - Allocate Dock slot (slot 1)                                │
│   - Allocate MenuBar slot (slot 2)                             │
│   - Allocate Window Chrome slot (slot 3)                       │
│   - Initialize their state buffers                              │
│                                                                 │
│   Threads: 4 (one per system app)                               │
│   Time: <1ms                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: GPU App Discovery (boot_phase3 kernel)                │
│                                                                 │
│   - Scan filesystem index for .gpuapp entries                   │
│   - For each found: queue I/O load request                      │
│   - Track pending loads in boot state                           │
│                                                                 │
│   Threads: index_entry_count (parallel scan)                    │
│   Time: <5ms for 10K files                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: GPU App Loading (boot_phase4 kernel, loops)           │
│                                                                 │
│   While pending_loads > 0:                                      │
│     - Poll I/O handles for completion                           │
│     - For each completed: initialize app from bytes             │
│     - Decrement pending count                                   │
│                                                                 │
│   Threads: 256 (for parallel copy during init)                  │
│   Time: I/O bound (~10ms per app, parallel)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Boot Complete → Normal Operation                       │
│                                                                 │
│   - Set boot_complete = 1                                       │
│   - Megakernel loop begins                                      │
│   - All apps run every frame                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Boot State (GPU-Resident)

```metal
struct BootState {
    // Phase tracking
    atomic_uint current_phase;
    atomic_uint phase_complete;

    // App discovery
    uint discovered_apps[MAX_BOOT_APPS];  // path_idx for each
    atomic_uint discovered_count;

    // Pending loads
    uint pending_handles[MAX_BOOT_APPS];  // I/O handles
    atomic_uint pending_count;
    atomic_uint loaded_count;

    // Errors
    uint error_count;
    uint last_error;

    // Stats
    uint boot_start_frame;
    uint boot_end_frame;
};

#define MAX_BOOT_APPS 32  // Max apps to auto-load at boot
```

---

## GPU-Side Implementation (Metal)

### Phase 1: System Initialization

```metal
kernel void boot_phase1_system_init(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    device BootState* boot [[buffer(3)]],
    constant uint& max_slots [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Thread 0 initializes header
    if (tid == 0) {
        header->max_slots = max_slots;
        header->active_count = 0;
        header->free_bitmap[0] = 0xFFFFFFFF;  // All free
        header->free_bitmap[1] = 0xFFFFFFFF;

        boot->current_phase = 1;
        boot->discovered_count = 0;
        boot->pending_count = 0;
        boot->loaded_count = 0;
        boot->error_count = 0;
    }

    // All threads initialize app descriptors (parallel)
    if (tid < max_slots) {
        apps[tid].flags = 0;  // INACTIVE
        apps[tid].app_type = 0;
        apps[tid].slot_id = tid;
        apps[tid].state_offset = 0;
        apps[tid].vertex_offset = 0;
        apps[tid].vertex_count = 0;
    }

    // Wait for all
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        atomic_store_explicit(&boot->phase_complete, 1, memory_order_release);
    }
}
```

### Phase 2: System Apps

```metal
kernel void boot_phase2_system_apps(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    device BootState* boot [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Wait for phase 1
    while (atomic_load_explicit(&boot->phase_complete, memory_order_acquire) < 1) {}

    // Each thread initializes one system app
    if (tid == 0) {
        // Compositor (slot 0)
        uint slot = allocate_slot_atomic(header, 0);
        apps[slot].flags = ACTIVE_FLAG | VISIBLE_FLAG;
        apps[slot].app_type = APP_TYPE_COMPOSITOR;
        apps[slot].priority = PRIORITY_REALTIME;
        apps[slot].state_offset = 0;
        apps[slot].state_size = 64;
        // Compositor runs last, renders everything
    }
    else if (tid == 1) {
        // Dock (slot 1)
        uint slot = allocate_slot_atomic(header, 1);
        apps[slot].flags = ACTIVE_FLAG | VISIBLE_FLAG;
        apps[slot].app_type = APP_TYPE_DOCK;
        apps[slot].priority = PRIORITY_REALTIME;
        apps[slot].state_offset = 64;
        apps[slot].state_size = 4096;
        apps[slot].vertex_offset = 0;
        apps[slot].vertex_size = 32 * 48 * sizeof(RenderVertex);
    }
    else if (tid == 2) {
        // MenuBar (slot 2)
        uint slot = allocate_slot_atomic(header, 2);
        apps[slot].flags = ACTIVE_FLAG | VISIBLE_FLAG;
        apps[slot].app_type = APP_TYPE_MENUBAR;
        apps[slot].priority = PRIORITY_REALTIME;
        apps[slot].state_offset = 64 + 4096;
        apps[slot].state_size = 8192;
    }
    else if (tid == 3) {
        // Window Chrome (slot 3)
        uint slot = allocate_slot_atomic(header, 3);
        apps[slot].flags = ACTIVE_FLAG | VISIBLE_FLAG;
        apps[slot].app_type = APP_TYPE_WINDOW_CHROME;
        apps[slot].priority = PRIORITY_REALTIME;
        apps[slot].state_offset = 64 + 4096 + 8192;
        apps[slot].state_size = 256;
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        header->active_count = 4;
        boot->current_phase = 2;
        atomic_store_explicit(&boot->phase_complete, 2, memory_order_release);
    }
}
```

### Phase 3: App Discovery

```metal
kernel void boot_phase3_discover_apps(
    device FilesystemIndex* fs_index [[buffer(0)]],
    device BootState* boot [[buffer(1)]],
    device PipelineState* io_state [[buffer(2)]],
    device IORequest* io_queue [[buffer(3)]],
    device FileHandle* io_handles [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Wait for phase 2
    while (atomic_load_explicit(&boot->phase_complete, memory_order_acquire) < 2) {}

    // Parallel scan of filesystem index
    uint entries_per_thread = (fs_index->entry_count + threads - 1) / threads;
    uint start = tid * entries_per_thread;
    uint end = min(start + entries_per_thread, fs_index->entry_count);

    for (uint i = start; i < end; i++) {
        device FileEntry* entry = &fs_index->entries[i];

        // Check if .gpuapp (by extension hash or suffix check)
        if (entry_is_gpuapp(entry)) {
            // Claim a discovery slot
            uint slot = atomic_fetch_add_explicit(&boot->discovered_count, 1, memory_order_relaxed);
            if (slot < MAX_BOOT_APPS) {
                boot->discovered_apps[slot] = i;  // Store path_idx

                // Queue I/O load request
                uint handle = request_read(io_state, io_queue, io_handles,
                                          i, BOOT_APP_ID, IO_PRIORITY_HIGH);
                if (handle != INVALID_HANDLE) {
                    boot->pending_handles[slot] = handle;
                    atomic_fetch_add_explicit(&boot->pending_count, 1, memory_order_relaxed);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        boot->current_phase = 3;
        atomic_store_explicit(&boot->phase_complete, 3, memory_order_release);
    }
}
```

### Phase 4: App Loading Loop

```metal
kernel void boot_phase4_load_apps(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    device BootState* boot [[buffer(3)]],
    device FileHandle* io_handles [[buffer(4)]],
    device uchar* content_pool [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Wait for phase 3
    while (atomic_load_explicit(&boot->phase_complete, memory_order_acquire) < 3) {}

    // Poll until all pending loads complete
    uint max_iterations = 10000;  // Safety limit

    while (max_iterations-- > 0) {
        uint pending = atomic_load_explicit(&boot->pending_count, memory_order_relaxed);
        if (pending == 0) break;

        // Thread 0 checks handles
        if (tid == 0) {
            uint discovered = boot->discovered_count;
            for (uint i = 0; i < discovered && i < MAX_BOOT_APPS; i++) {
                uint handle = boot->pending_handles[i];
                if (handle == INVALID_HANDLE) continue;

                uint status = check_status(io_handles, handle);

                if (status == STATUS_READY) {
                    // Get loaded file data
                    uint offset = io_handles[handle].content_offset;
                    uint size = io_handles[handle].file_size;
                    device uchar* file_data = content_pool + offset;

                    // Validate header
                    device GpuAppFileHeader* hdr = (device GpuAppFileHeader*)file_data;
                    if (validate_gpuapp_header(hdr)) {
                        // Allocate app slot
                        uint slot = allocate_app_slot_atomic(header);
                        if (slot != INVALID_SLOT) {
                            // Initialize (simplified - thread 0 only for now)
                            init_app_from_gpuapp(apps, unified_state, slot, hdr, file_data);
                            atomic_fetch_add_explicit(&boot->loaded_count, 1, memory_order_relaxed);
                        }
                    } else {
                        boot->error_count++;
                    }

                    // Mark as processed
                    boot->pending_handles[i] = INVALID_HANDLE;
                    atomic_fetch_sub_explicit(&boot->pending_count, 1, memory_order_relaxed);
                }
                else if (status == STATUS_ERROR) {
                    boot->error_count++;
                    boot->pending_handles[i] = INVALID_HANDLE;
                    atomic_fetch_sub_explicit(&boot->pending_count, 1, memory_order_relaxed);
                }
            }
        }

        // Brief spin wait (GPU doesn't have sleep, but can yield)
        for (uint i = 0; i < 1000; i++) { /* spin */ }
    }

    if (tid == 0) {
        boot->current_phase = 4;
        atomic_store_explicit(&boot->phase_complete, 4, memory_order_release);
    }
}
```

### Boot Complete Check

```metal
// Called by megakernel to check if boot is done
inline bool is_boot_complete(device BootState* boot) {
    return atomic_load_explicit(&boot->phase_complete, memory_order_acquire) >= 4;
}
```

---

## Rust-Side Implementation

### GpuOs::boot()

```rust
impl GpuOs {
    /// Boot the GPU OS
    pub fn boot(device: &Device) -> Self {
        // Phase 0: CPU setup
        let buffers = Self::allocate_buffers(device);
        let pipelines = Self::compile_pipelines(device);

        // Register filesystem paths
        let mut content_pipeline = ContentPipeline::new(device, 64 * 1024 * 1024).unwrap();

        // Register apps directory
        if let Some(home) = dirs::home_dir() {
            let apps_dir = home.join("apps");
            if apps_dir.exists() {
                Self::register_directory(&mut content_pipeline, &apps_dir);
            }
        }

        // Build filesystem index on GPU
        let fs_index = GpuFilesystemIndex::build(device, &content_pipeline.paths());

        // Create boot state buffer
        let boot_state = device.new_buffer(
            std::mem::size_of::<BootState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Start I/O coprocessor
        let io_thread = IOCoprocessor::new(Arc::new(content_pipeline)).start();

        // Dispatch boot kernels
        let queue = device.new_command_queue();

        // Phase 1
        Self::dispatch_boot_phase1(&queue, &pipelines, &buffers, &boot_state);

        // Phase 2
        Self::dispatch_boot_phase2(&queue, &pipelines, &buffers, &boot_state);

        // Phase 3
        Self::dispatch_boot_phase3(&queue, &pipelines, &buffers, &boot_state, &fs_index);

        // Phase 4 runs in a loop until complete
        Self::dispatch_boot_phase4_loop(&queue, &pipelines, &buffers, &boot_state);

        // Boot complete - return OS handle
        Self {
            device: device.clone(),
            buffers,
            pipelines,
            boot_state,
            fs_index,
            // ...
        }
    }

    fn dispatch_boot_phase4_loop(
        queue: &CommandQueue,
        pipelines: &Pipelines,
        buffers: &Buffers,
        boot_state: &Buffer,
    ) {
        // Run phase 4 until boot_complete
        loop {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            enc.set_compute_pipeline_state(&pipelines.boot_phase4);
            // Set buffers...
            enc.dispatch_threads(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();

            cmd.commit();
            cmd.wait_until_completed();

            // Check if complete
            let phase = Self::read_boot_phase(boot_state);
            if phase >= 4 {
                break;
            }

            // Brief CPU sleep to not spin too hard
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
```

---

## Test Plan

### Test 1: System Apps Launch at Boot

```rust
#[test]
fn test_boot_system_apps() {
    let device = Device::system_default().unwrap();
    let os = GpuOs::boot(&device);

    // Verify 4 system apps
    assert_eq!(os.app_count(), 4);

    // Verify types
    let apps = os.read_app_descriptors();
    assert_eq!(apps[0].app_type, APP_TYPE_COMPOSITOR);
    assert_eq!(apps[1].app_type, APP_TYPE_DOCK);
    assert_eq!(apps[2].app_type, APP_TYPE_MENUBAR);
    assert_eq!(apps[3].app_type, APP_TYPE_WINDOW_CHROME);
}
```

### Test 2: Auto-Discover Apps in ~/apps/

```rust
#[test]
fn test_boot_discovers_apps() {
    let device = Device::system_default().unwrap();

    // Create test apps
    let apps_dir = tempdir().unwrap();
    create_test_gpuapp(apps_dir.path().join("app1.gpuapp"));
    create_test_gpuapp(apps_dir.path().join("app2.gpuapp"));
    create_test_gpuapp(apps_dir.path().join("app3.gpuapp"));

    // Boot with custom apps dir
    let os = GpuOs::boot_with_apps_dir(&device, apps_dir.path());

    // Should have 4 system + 3 user apps
    assert_eq!(os.app_count(), 7);
}
```

### Test 3: Boot Completes Without Apps Dir

```rust
#[test]
fn test_boot_no_apps_dir() {
    let device = Device::system_default().unwrap();

    // Boot with nonexistent apps dir
    let os = GpuOs::boot_with_apps_dir(&device, Path::new("/nonexistent"));

    // Should still have system apps
    assert_eq!(os.app_count(), 4);
}
```

### Test 4: Boot Performance

```rust
#[test]
fn test_boot_performance() {
    let device = Device::system_default().unwrap();

    // Create 10 test apps
    let apps_dir = tempdir().unwrap();
    for i in 0..10 {
        create_test_gpuapp(apps_dir.path().join(format!("app{}.gpuapp", i)));
    }

    let start = Instant::now();
    let os = GpuOs::boot_with_apps_dir(&device, apps_dir.path());
    let boot_time = start.elapsed();

    // Should boot in <500ms even with 10 apps
    assert!(boot_time < Duration::from_millis(500));

    // All apps loaded
    assert_eq!(os.app_count(), 14);  // 4 system + 10 user
}
```

### Test 5: Boot Stats

```rust
#[test]
fn test_boot_stats() {
    let device = Device::system_default().unwrap();

    let apps_dir = tempdir().unwrap();
    create_test_gpuapp(apps_dir.path().join("app1.gpuapp"));

    let os = GpuOs::boot_with_apps_dir(&device, apps_dir.path());

    let stats = os.boot_stats();
    assert_eq!(stats.discovered_count, 1);
    assert_eq!(stats.loaded_count, 1);
    assert_eq!(stats.error_count, 0);
    assert!(stats.boot_frames > 0);
}
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| System apps launch time | <5ms |
| Per-app load time | <15ms (I/O bound) |
| 10 app boot time | <200ms |
| CPU work after setup | 0 (except I/O thread) |
| Boot without apps | Still succeeds |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/boot.rs` | Boot sequence orchestration |
| `src/gpu_os/shaders/boot.metal` | Boot phase kernels |
| `tests/test_issue_170_boot_sequence.rs` | Tests |

---

## Anti-Patterns Avoided

| Bad Pattern | Why Bad | Good Pattern |
|-------------|---------|--------------|
| CPU scans directory | CPU work | GPU scans index |
| CPU loads each app | Sequential | GPU batches all loads |
| CPU initializes apps | CPU coordinates | GPU initializes from bytes |
| CPU waits for each load | Slow | GPU polls, all loads parallel |
| Boot blocks on I/O | Slow | Async I/O, GPU polls |
