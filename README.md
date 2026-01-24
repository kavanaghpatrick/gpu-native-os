# GPU-Native OS

A research sandbox proving that GPUs can replace CPUs as the primary compute substrate. All application logic—input handling, state updates, layout, text rendering, and painting—runs entirely on the GPU via Metal compute shaders.

## The Thesis: The GPU Is The Computer

Modern GPUs are general-purpose parallel computers artificially constrained to "graphics acceleration" by legacy software. This project demonstrates:

1. **GPU waves replace CPU threads** — A single wavefront (32-64 threads) replaces traditional CPU threading
2. **Compute shaders are programs** — Metal/CUDA kernels are Turing-complete; they run any algorithm
3. **The CPU is a bottleneck** — Every CPU touchpoint is technical debt we're eliminating
4. **Unified memory changes everything** — Apple Silicon shares memory; the CPU/GPU distinction is artificial

**Goal**: Zero CPU involvement in steady-state operation. CPU boots the system, then hands control to GPU.

---

## Architecture Overview

```
+------------------------------------------------------------------------+
|                      CPU (Rust Host) - MINIMAL                         |
|  +-------------+  +----------------+  +---------------------------+    |
|  | Buffer Setup|  | Command Queue  |  | Async Notification        |    |
|  | (one-time)  |  | Dispatch       |  | (SharedEvent callback)    |    |
+------------------------------------------------------------------------+
                              | Metal API (Non-blocking)
+------------------------------------------------------------------------+
|                        GPU (Metal Compute)                             |
|                                                                        |
|  +------------------------+    +----------------------------------+    |
|  |   Document Pipeline    |    |    Filesystem Search             |    |
|  |   Tokenizer → Parser   |    |    3M parallel threads           |    |
|  |   Style → Layout       |    |    Fuzzy matching                |    |
|  |   Paint → Render       |    |    GPU sorting                   |    |
|  +------------------------+    +----------------------------------+    |
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |  GPU-Resident Data (Zero-Copy via mmap)                          |  |
|  |  - Filesystem Index: mmap → Metal buffer (never copied)          |  |
|  |  - HTML/CSS: raw bytes → GPU tokenizes/parses                    |  |
|  |  - All results: GPU writes → GPU renders                         |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
|  Unified Memory (MTLStorageModeShared) - Zero-copy on Apple Silicon    |
+------------------------------------------------------------------------+
```

---

## Core Systems

### 1. Document Pipeline (GPU-Native Browser Engine)

A complete HTML rendering pipeline running entirely on GPU—no CPU string parsing, no CPU layout calculations.

```
HTML bytes (input)
    ↓ [Tokenizer - 2 GPU passes]
Token[] stream (65K max)
    ↓ [Parser - 3 GPU passes]
Element[] DOM tree (16K nodes)
    ↓ [Style Resolver - 1 kernel per element]
ComputedStyle[] (256 bytes each)
    ↓ [Layout Engine - Issue #89: Level-parallel, 5 kernels]
LayoutBox[] (positions/dimensions)
    ↓ [Paint - 6 GPU kernels]
PaintVertex[]
    ↓ [Fragment Shader - Rasterization]
Frame Buffer
```

#### Tokenizer: HTML → Tokens

**Two-pass parallel tokenization**:

1. **Boundary Detection**: 1024 threads scan HTML bytes in parallel, marking token starts (`<`, `>`, whitespace transitions)
2. **Token Extraction**: Extract token types (TAG_OPEN, TAG_CLOSE, TEXT) with start/end offsets

```rust
#[repr(C)]
pub struct Token {
    pub token_type: u32,  // TAG_OPEN, TAG_CLOSE, TEXT, etc.
    pub start: u32,       // Position in HTML bytes
    pub end: u32,
    pub _padding: u32,
}
```

#### Parser: Tokens → DOM Tree

**Three-pass parsing**:

1. **Allocate**: Count tokens → allocate element slots in parallel
2. **Build Tree**: Stack machine converts tokens to element tree (sets parent, first_child, next_sibling)
3. **Extract Text**: Parallel text copying from token ranges

```rust
#[repr(C)]
pub struct Element {
    pub element_type: u32,      // DIV, SPAN, P, H1, TEXT, etc.
    pub parent: i32,            // Index of parent (-1 if root)
    pub first_child: i32,
    pub next_sibling: i32,
    pub text_start: u32,
    pub text_length: u32,
    pub token_index: u32,
    pub _padding: u32,
}
```

#### Style Resolution: Elements + Selectors → Computed Styles

Each GPU thread processes one element:
1. Iterate all CSS selectors
2. Match element (tagname, class, ID, attributes, pseudo-classes)
3. Calculate specificity
4. Apply properties in specificity order (cascading)
5. Inherit from parent

**ComputedStyle** (256 bytes per element):
- `display`, `position`, `width`, `height`
- `margin[4]`, `padding[4]`, `border_width[4]`
- `flex_direction`, `justify_content`, `align_items`
- `color`, `background_color`, `border_color`
- `font_size`, `line_height`, `text_align`
- `border_radius`, `opacity`, `z_index`
- `overflow_x`, `overflow_y`
- Box shadows, gradients, transforms

#### Layout Engine: Level-Parallel Algorithm (Issue #89)

Traditional recursive descent is CPU-friendly but GPU-hostile. The level-parallel algorithm processes the tree by depth level in parallel.

**Five GPU Kernels**:

1. **Compute Depths**: Each thread finds its element's depth, atomic_max for tree height
2. **Sum Heights**: Parallel prefix sum for intrinsic sizes
3. **Position Siblings**: All children at same level processed in parallel (flex layout)
4. **Finalize Level**: Convert relative → absolute positions
5. **Propagate Widths & Text Height**: Top-down width propagation, text wrapping

```rust
#[repr(C)]
pub struct LayoutBox {
    pub x: f32, pub y: f32,                    // Border box position
    pub width: f32, pub height: f32,
    pub content_x: f32, pub content_y: f32,    // Content box
    pub content_width: f32, pub content_height: f32,
    pub scroll_width: f32, pub scroll_height: f32,
    pub _padding: [f32; 6],                    // GPU alignment
}
```

#### Paint: Layout → Vertices (6 Kernels)

1. **Count Vertices**: Per-element vertex count (background: 4, border: 16, text: 4/char)
2. **Compute Offsets**: Prefix sum for vertex buffer positions
3. **Generate Background**: Quad vertices with colors
4. **Generate Borders**: Four trapezoids (top, right, bottom, left)
5. **Generate Text**: Per-character quads with atlas UVs
6. **Generate Images**: Image quads with atlas UVs

```rust
#[repr(C)]
pub struct PaintVertex {
    pub position: [f32; 2],     // NDC [-1, 1]
    pub tex_coord: [f32; 2],    // UV [0, 1]
    pub color: [f32; 4],        // RGBA
    pub flags: u32,             // FLAG_BACKGROUND|BORDER|TEXT|IMAGE
    _padding: [u32; 3],         // Alignment to 48 bytes
}
```

#### Text Processing (Issue #90: GPU-Native Text Containers)

**Three-stage pipeline**:

1. **Character-to-Glyph Mapping**: ASCII → glyph metrics (width, bearing, atlas position)
2. **Parallel Prefix Sum**: Cumulative widths via Blelloch reduction (O(log N))
3. **Line Breaking & Positioning**: Find break opportunities, assign lines, apply text-align

---

### 2. Filesystem Search (3M+ Parallel Threads)

GPU-accelerated fuzzy search across millions of filesystem paths.

```
Pass 0: tokenize_query_kernel (256 threads)
  → Tokenize raw query bytes into words, atomic slot allocation

Pass 1: fuzzy_search_kernel (11,719 threadgroups × 256 = 3M threads)
  → Each thread scores one path against query words

Pass 2: sort_results_kernel
  → Insertion sort on top 100 results

Pass 3: generate_results_text_kernel
  → Generate TextChar[] for rendering
```

**Key Innovation**: CPU does ONE memcpy of raw query bytes. GPU tokenizes, searches, sorts, and renders.

---

### 3. Zero-Copy Infrastructure

#### mmap Buffer (Issue #82)

```
Traditional:  File → read() → CPU buffer → copy → GPU buffer
Zero-Copy:    File → mmap() → newBufferWithBytesNoCopy() → GPU buffer
                              (same physical memory!)
```

| Metric | Traditional | Zero-Copy |
|--------|-------------|-----------|
| Memory copies | 2 | 0 |
| Memory usage | 2× file size | 1× file size |
| Load time (10MB) | ~15ms | <1ms |

#### GPU-Resident Filesystem Index (Issue #77)

```rust
#[repr(C)]
pub struct GpuPathEntry {
    pub path: [u8; 224],      // Fixed-width (null-padded)
    pub path_len: u16,
    pub flags: u16,           // is_dir, is_hidden
    pub parent_idx: u32,
    pub size: u64,
    pub mtime: u64,
    pub _reserved: [u8; 8],   // Pad to 256 bytes (cache-aligned)
}
```

CPU scans once → mmap index → GPU owns data forever. Zero copies per search.

---

## Metal Shader Architecture

### Two-Language Struct Synchronization

Every GPU struct exists in both Rust and Metal with matching layouts:

**Rust** (`src/gpu_os/document/layout.rs`):
```rust
#[repr(C)]
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    // ... must match Metal exactly
    pub _padding: [f32; 6],  // CRITICAL: GPU alignment
}
```

**Metal** (`src/gpu_os/document/layout.metal`):
```metal
struct LayoutBox {
    float x;
    float y;
    // ... must match Rust exactly
    float _padding[6];
};
```

Use compile-time assertions:
```rust
assert_eq!(std::mem::size_of::<LayoutBox>(), 80);
```

### Key Design Patterns

1. **Parallel Prefix Sum (Blelloch)**: O(log N) cumulative sums for vertex offsets, text widths
2. **Atomic Coordination**: `atomic_fetch_add_explicit` for thread-safe counters
3. **Ring Buffers**: Circular input queue avoids allocation
4. **Indirect Rendering**: GPU computes draw arguments, CPU just dispatches

---

## GpuApp Framework

Standard interface for GPU-native applications:

```rust
pub trait GpuApp {
    fn name(&self) -> &str;
    fn compute_pipeline(&self) -> &ComputePipelineState;
    fn render_pipeline(&self) -> &RenderPipelineState;
    fn vertices_buffer(&self) -> &Buffer;
    fn vertex_count(&self) -> usize;
    fn app_buffers(&self) -> Vec<&Buffer>;
    fn params_buffer(&self) -> &Buffer;
    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32);
    fn handle_input(&mut self, event: &InputEvent);
}
```

### Buffer Slot Convention

| Slot | Buffer | Purpose |
|------|--------|---------|
| 0 | FrameState | OS-provided: cursor, time, frame number |
| 1 | InputQueue | OS-provided: keyboard/mouse events |
| 2 | AppParams | App-specific per-frame parameters |
| 3+ | App buffers | App-specific state |

### Pipeline Modes

```rust
PipelineMode::LowLatency      // Text editors: wait for frame completion
PipelineMode::HighThroughput  // Simulations: allow frame overlap (6.75× speedup)
```

---

## Project Structure

```
src/gpu_os/
├── Foundation
│   ├── mmap_buffer.rs        # #82 - Zero-copy file-to-GPU
│   ├── gpu_index.rs          # #77 - GPU-Resident Filesystem Index
│   ├── parallel_alloc.rs     # #91 - Parallel Prefix Allocator
│   ├── metal_types.rs        # Metal-safe struct definitions
│   └── profiler.rs           # GPU profiling
│
├── Core Framework
│   ├── app.rs                # GpuApp trait, GpuRuntime
│   ├── kernel.rs             # #11 - Unified Worker Model
│   ├── memory.rs             # #12 - Memory Architecture
│   ├── input.rs              # #13 - Input Pipeline (HID → GPU)
│   ├── render.rs             # #17 - Hybrid Rendering
│   └── vsync.rs              # #18 - VSync Execution
│
├── Document Pipeline
│   ├── document/
│   │   ├── tokenizer.rs/.metal   # HTML bytes → Tokens
│   │   ├── parser.rs/.metal      # Tokens → DOM tree
│   │   ├── style.rs/.metal       # CSS selector matching
│   │   ├── layout.rs/.metal      # #89 - Level-parallel layout
│   │   ├── paint.rs/.metal       # Layout → vertices
│   │   ├── text.rs/.metal        # #90 - GPU text containers
│   │   ├── hit_test.rs           # GPU hit testing
│   │   ├── image.rs              # GPU image atlas
│   │   └── navigation.rs         # Link handling
│   └── document_app.rs           # GpuApp implementation
│
├── Applications
│   ├── filesystem.rs         # GPU filesystem search (3M+ paths)
│   ├── content_search.rs     # GPU content search
│   ├── duplicate_finder.rs   # GPU duplicate detection
│   └── text_editor.rs        # GPU text editor
│
└── Demos
    ├── game_of_life.rs       # Cellular automaton
    ├── particles.rs          # 10K+ particle physics
    ├── boids.rs              # 1024-boid flocking
    ├── mandelbrot.rs         # Fractal viewer
    ├── metaballs.rs          # Organic blobs
    └── waves.rs              # Wave simulation
```

---

## Performance

### Benchmarks (Apple M4 Pro)

#### GPU Wins on Architecture, Not Raw Speed

| Scenario | Traditional (3 dispatches) | GPU-Native (1 dispatch) | Speedup |
|----------|---------------------------|------------------------|---------|
| 10 widgets | 313 µs | 99 µs | **3.2×** |
| 200 widgets | 284 µs | 89 µs | **3.2×** |
| 1000 widgets | 348 µs | 110 µs | **3.2×** |

The win: **eliminating CPU-GPU sync points**, not raw compute speed.

#### Frame Pipelining

| Mode | 1000 Frames | FPS |
|------|-------------|-----|
| Serial | 104.6 ms | 9,562 |
| Pipelined | 15.5 ms | 64,584 |
| **Speedup** | | **6.75×** |

#### Filesystem Search

| Metric | Value |
|--------|-------|
| Paths supported | 3,000,000+ |
| Threads per search | 3M (one per path) |
| CPU work per search | 1 memcpy |
| Memory usage | ~24MB fixed |

---

## Current CPU Dependencies (Technical Debt)

| Operation | Current | Target | Status |
|-----------|---------|--------|--------|
| HTML parsing | GPU | GPU | ✅ Implemented |
| CSS matching | GPU | GPU | ✅ Implemented |
| Layout | GPU level-parallel | GPU | ✅ Implemented |
| Text wrapping | GPU | GPU | ✅ Implemented |
| Vertex generation | GPU | GPU | ✅ Implemented |
| Initial HTML load | CPU | GPU-initiated storage | 🔄 In progress |
| Font parsing | CPU | GPU bezier extraction | 📋 Planned |
| Frame submission | CPU | Persistent kernels | 📋 Planned |

---

## Quick Start

```bash
# Build
cargo build --release

# Run demos
cargo run --release --example filesystem_browser   # Main: GPU file search
cargo run --release --example document_viewer      # GPU HTML rendering
cargo run --release --example waves                # Wave simulation
cargo run --release --example boids                # Flocking simulation
cargo run --release --example mandelbrot           # Fractal viewer

# Run tests
cargo test

# Run specific test suites
cargo test --test test_issue_89_layout             # Level-parallel layout
cargo test --test test_issue_60_text_wrapping      # Text processing
cargo test --test test_gpu_native_document         # Document pipeline
```

### Controls (Filesystem Browser)

- **Type**: Fuzzy search
- **Up/Down**: Navigate results
- **Enter**: Open file
- **Escape**: Clear search

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) or AMD GPU
- Rust 1.70+
- Xcode Command Line Tools (Metal compiler)

## License

MIT

## Acknowledgments

Inspired by Casey Muratori's discussions on GPU-driven UI and research on GPU-native computing paradigms.
