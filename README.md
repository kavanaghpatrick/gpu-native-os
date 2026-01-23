# GPU-Native OS

A single-threadgroup operating system running entirely on the GPU via Metal compute shaders.

## Overview

This project explores a radical approach to UI architecture: instead of the CPU orchestrating rendering with GPU assistance, **the GPU runs the entire OS** — input handling, hit testing, layout, sorting, and rendering — in a single compute kernel.

All 1024 threads participate in every phase. The CPU's role is minimal: forward input events and submit command buffers.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CPU (minimal)                         │
├──────────────────────────────────────────────────────────────┤
│  macOS events → InputQueue.push() → [shared buffer]          │
│  command_buffer.encode(compute) → encode(render) → present() │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   GPU (1024 threads, 1 threadgroup)          │
├──────────────────────────────────────────────────────────────┤
│  Phase 1: INPUT      - Read events from ring buffer          │
│  Phase 2: HIT TEST   - Test all widgets against cursor       │
│  Phase 3: VISIBILITY - Count visible widgets, generate keys  │
│  Phase 4: SORT       - Bitonic sort by z-order               │
│  Phase 5: UPDATE     - Frame state, draw arguments           │
│                                                              │
│  Vertex Gen → Fragment Shader → Pixels                       │
└──────────────────────────────────────────────────────────────┘
```

## Why?

**Apple Silicon unified memory** eliminates CPU↔GPU copies. Widgets live in shared buffers accessible to both. The GPU reads input, processes state, and renders — all without round-trips.

**Single threadgroup** means all 1024 threads share 32KB of fast threadgroup memory. They can synchronize with barriers and use SIMD operations for parallel reductions.

**Branchless SIMD** keeps all threads executing the same instructions, avoiding warp divergence that kills GPU performance.

## Requirements

- macOS with Apple Silicon (M1/M2/M3) or AMD GPU
- Rust 1.70+
- Xcode Command Line Tools (for Metal compiler)

## Quick Start

```bash
# Build
cargo build --release

# Run the demo
cargo run --example gpu_os_demo --release

# Run tests
cargo test
```

## Project Structure

```
src/
├── lib.rs                    # Library entry
└── gpu_os/
    ├── mod.rs                # Module exports
    ├── kernel.rs             # Compute kernel management
    ├── memory.rs             # GPU buffer allocation
    ├── input.rs              # Input event handling
    ├── layout.rs             # Constraint-based layout
    ├── widget.rs             # Widget state & types
    ├── text.rs               # Text rendering (7-segment + font atlas)
    ├── render.rs             # Hybrid compute+fragment pipeline
    ├── vsync.rs              # Frame timing & sync
    └── shaders/
        └── kernel.metal      # GPU compute kernel

examples/
└── gpu_os_demo.rs            # Visual demo application

tests/
└── test_issue_*.rs           # Integration tests per module

docs/
├── GPU_NATIVE_OS_V2_PRD.md   # Detailed design document
└── ...
```

## Memory Layout

All buffers use `StorageModeShared` for unified CPU/GPU access:

| Buffer | Size | Purpose |
|--------|------|---------|
| Widgets | 24B × 1024 | Compressed widget state |
| Input Queue | ~7KB | Ring buffer (256 events) |
| Vertices | 64B × 6 × 1024 | GPU-generated render data |
| Draw Args | 16B | Indirect draw parameters |
| Frame State | 32B | Cursor, focus, timing |

### Widget Format (24 bytes)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ bounds (8B) │ colors (4B) │ style (2B)  │ parent (2B) │
│ f16×4       │ RGB565×2    │ packed bits │ widget_id   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ child (2B)  │ sibling(2B) │ z_order(2B) │ padding(2B) │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## The Compute Kernel

Every frame, 1024 GPU threads execute 5 phases:

### Phase 1: Input Collection
Thread 0 reads the input queue head/tail. Threads 0-63 copy pending events into threadgroup memory. Thread 0 processes mouse moves to update cursor position.

### Phase 2: Hit Testing
Each thread tests one widget: is the cursor inside bounds? SIMD operations find hits within each 32-thread group. Atomic compare-exchange finds the topmost (highest z-order) hit.

### Phase 3: Visibility Counting
Each thread checks visibility flags, generates sort keys `(z_order << 16) | widget_id`, and atomically counts visible widgets.

### Phase 4: Bitonic Sort
All 1024 threads participate in parallel bitonic sort. O(log²n) passes with barrier synchronization. Result: widgets sorted back-to-front for correct rendering.

### Phase 5: State Update
Thread 0 writes frame state (cursor position, hovered widget, frame number) and draw arguments for the render pass.

## Performance Characteristics

- **Target**: 120 FPS (8.33ms frame budget)
- **Kernel execution**: <1ms for 1024 widgets
- **Memory bandwidth**: ~50KB/frame
- **Thread utilization**: 100% per phase (no divergence)

## Development

Each module maps to a GitHub issue (#11-#18) for tracking:

| Issue | Module | Description |
|-------|--------|-------------|
| #11 | kernel | Unified Worker Model |
| #12 | memory | Memory Architecture |
| #13 | input | Input Pipeline |
| #14 | layout | Layout Engine |
| #15 | widget | Widget System |
| #16 | text | Text Rendering |
| #17 | render | Hybrid Rendering |
| #18 | vsync | VSync Execution |

## License

MIT

## Acknowledgments

Inspired by Casey Muratori's discussions on GPU-driven UI and Apple's TBDR (Tile-Based Deferred Rendering) architecture documentation.
