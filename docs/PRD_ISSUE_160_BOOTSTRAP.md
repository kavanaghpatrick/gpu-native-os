# PRD: GPU Bootstrap System (Issue #160)

## Overview

Replace GpuDesktop initialization with GpuAppSystem as the core OS. The CPU's only job is to boot the system, then hand control to GPU.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU creates components | GPU megakernel contains all code |
| CPU orchestrates startup | CPU boots, GPU runs |
| Multiple subsystems | One unified GpuAppSystem |
| Per-frame CPU involvement | Zero CPU in steady state |

## The GPU Insight

The desktop IS the GpuAppSystem. There is no separate "desktop manager" - just apps running in the megakernel:

```
BOOT SEQUENCE:
1. CPU: Create Metal device
2. CPU: Initialize GpuAppSystem
3. CPU: Launch system apps (dock, menubar, compositor)
4. GPU: Megakernel runs everything from here
5. CPU: Just submits command buffers, handles I/O
```

## Design

### Bootstrap Flow

```rust
pub fn boot_gpu_os(device: &Device) -> Result<GpuOs, String> {
    // 1. Create the app system (this IS the OS)
    let mut system = GpuAppSystem::new(device)?;

    // 2. Enable parallel megakernel
    system.set_use_parallel_megakernel(true);

    // 3. Launch system apps (these run at REALTIME priority)
    let compositor = system.launch_by_type(app_type::COMPOSITOR)?;
    system.set_priority(compositor, priority::REALTIME);

    let dock = system.launch_by_type(app_type::DOCK)?;
    system.set_priority(dock, priority::REALTIME);

    let menubar = system.launch_by_type(app_type::MENUBAR)?;
    system.set_priority(menubar, priority::REALTIME);

    // 4. System is ready - GPU takes over
    Ok(GpuOs { system })
}
```

### Main Loop (Minimal CPU)

```rust
impl GpuOs {
    pub fn run_frame(&mut self, drawable: &MetalDrawableRef) {
        // CPU just submits - GPU does everything
        self.system.run_frame();           // Megakernel updates all apps
        self.system.finalize_render();     // Sum vertices
        self.render_to_drawable(drawable); // Single draw call
    }

    pub fn handle_event(&mut self, event: Event) {
        // Queue to GPU - GPU dispatches
        match event {
            Event::MouseMove(x, y) => {
                self.system.queue_input(InputEvent::mouse_move(x, y));
            }
            Event::KeyDown(key) => {
                self.system.queue_input(InputEvent::key_down(key));
            }
            // ...
        }
        self.system.process_input(); // GPU routes to apps
    }
}
```

### System App Types

```rust
pub mod app_type {
    // System apps (priority::REALTIME)
    pub const COMPOSITOR: u32 = 200;   // Combines all window content
    pub const DOCK: u32 = 201;         // App launcher
    pub const MENUBAR: u32 = 202;      // Top menu bar
    pub const WINDOW_CHROME: u32 = 203; // Window decorations

    // User apps (priority::NORMAL)
    pub const TERMINAL: u32 = 5;
    pub const FILESYSTEM: u32 = 4;
    // ...
}
```

### GpuOs Struct

```rust
/// The entire GPU operating system
pub struct GpuOs {
    system: GpuAppSystem,

    // System app slots (for quick access)
    compositor_slot: u32,
    dock_slot: u32,
    menubar_slot: u32,

    // Screen config
    screen_width: f32,
    screen_height: f32,
}

impl GpuOs {
    /// Launch a user app
    pub fn launch_app(&mut self, app_type: u32) -> Option<u32> {
        let slot = self.system.launch_by_type(app_type)?;

        // Create window for it
        let x = 100.0 + (slot as f32 * 30.0);  // Cascade
        let y = 100.0 + (slot as f32 * 30.0);
        self.system.create_window(slot, x, y, 800.0, 600.0);

        // Notify dock
        self.notify_dock_app_launched(slot);

        Some(slot)
    }

    /// Close a user app
    pub fn close_app(&mut self, slot: u32) {
        self.system.close_app(slot);
        self.notify_dock_app_closed(slot);
    }
}
```

## Migration from GpuDesktop

| GpuDesktop Component | GpuOs Replacement |
|---------------------|-------------------|
| `GpuDesktop::new()` | `GpuOs::boot()` |
| `GpuDesktop::render()` | `GpuOs::run_frame()` |
| `compositor.render()` | Compositor app in megakernel |
| `dock.render()` | Dock app in megakernel |
| `window_manager` | GpuAppSystem window methods |
| `apps.launch()` | `system.launch_by_type()` |

## Tests

```rust
#[test]
fn test_boot_gpu_os() {
    let device = Device::system_default().unwrap();
    let os = GpuOs::boot(&device).expect("Boot failed");

    // System apps should be running
    assert!(os.system.get_app(os.compositor_slot).is_some());
    assert!(os.system.get_app(os.dock_slot).is_some());
    assert!(os.system.get_app(os.menubar_slot).is_some());
}

#[test]
fn test_launch_user_app() {
    let device = Device::system_default().unwrap();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL);
    assert!(terminal.is_some());

    let app = os.system.get_app(terminal.unwrap()).unwrap();
    assert_eq!(app.app_type, app_type::TERMINAL);
    assert!(app.is_visible());
}

#[test]
fn test_zero_cpu_frame() {
    let device = Device::system_default().unwrap();
    let mut os = GpuOs::boot(&device).unwrap();

    // Launch some apps
    os.launch_app(app_type::TERMINAL);
    os.launch_app(app_type::FILESYSTEM);

    // Run frames - CPU should just submit
    for _ in 0..100 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }

    // All apps should have run
    let stats = os.system.scheduler_stats();
    assert!(stats.active_count >= 5); // 3 system + 2 user
}

#[test]
fn test_input_routing_gpu() {
    let device = Device::system_default().unwrap();
    let mut os = GpuOs::boot(&device).unwrap();

    let terminal = os.launch_app(app_type::TERMINAL).unwrap();
    os.system.set_focus(terminal);

    // Queue input
    os.handle_event(Event::KeyDown(KEY_A));

    // Terminal should receive it
    let app = os.system.get_app(terminal).unwrap();
    assert!(app.flags & flags::DIRTY != 0);
}
```

## Benchmarks

```rust
#[test]
fn bench_boot_time() {
    let device = Device::system_default().unwrap();

    let start = Instant::now();
    let os = GpuOs::boot(&device).unwrap();
    let boot_time = start.elapsed();

    println!("Boot time: {:?}", boot_time);
    assert!(boot_time < Duration::from_millis(100), "Boot should be <100ms");
}

#[test]
fn bench_frame_time_no_cpu() {
    let device = Device::system_default().unwrap();
    let mut os = GpuOs::boot(&device).unwrap();

    // Launch 10 apps
    for _ in 0..10 {
        os.launch_app(app_type::TERMINAL);
    }

    // Warm up
    for _ in 0..10 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..1000 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }
    let duration = start.elapsed();

    let per_frame = duration.as_nanos() / 1000;
    println!("Per frame: {}ns ({:.0} FPS)", per_frame, 1_000_000_000.0 / per_frame as f64);
}
```

## Success Metrics

1. **Boot time**: < 100ms from init to first frame
2. **Frame time**: < 1ms for 10 apps (parallel megakernel)
3. **CPU usage**: < 5% in steady state
4. **Zero CPU per-app**: No CPU code runs per app per frame
