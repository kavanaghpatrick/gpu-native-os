// GPU-Native OS Modules
// Each module corresponds to a GitHub issue

pub mod metal_types;    // Metal-safe struct definitions with alignment guarantees
pub mod mmap_buffer;    // #82 - Zero-copy file access via mmap
pub mod gpu_io;         // #112 - GPU-Direct Storage with MTLIOCommandQueue
pub mod batch_io;       // #125 - GPU Batch I/O with MTLIOCommandQueue
pub mod streaming_search; // #132 - Streaming I/O - Overlap File Loading with GPU Search
pub mod gpu_index;      // #77 - GPU-Resident Filesystem Index
pub mod shared_index;   // #135 - Shared GPU-Resident Filesystem Index
pub mod parallel_alloc;   // #91 - GPU-Native Parallel Prefix Allocator
pub mod parallel_compact; // #126 - GPU Parallel Compaction with Prefix Sum
pub mod gpu_cache;        // #127 - GPU-Resident Persistent File Cache
pub mod profiler;       // GPU profiling and persistence boundary measurement
pub mod work_queue;     // GPU work queue for persistent kernel execution
pub mod persistent_search; // #133 - Persistent Search Kernel
pub mod kernel;      // #11 - Unified Worker Model
pub mod memory;      // #12 - Memory Architecture
pub mod input;       // #13 - Input Pipeline
pub mod layout;      // #14 - Layout Engine
pub mod widget;      // #15 - Widget System
pub mod text;        // #16 - Text Rendering (MSDF - partial)
pub mod text_render; // Simple bitmap font text rendering (production)
pub mod render;      // #17 - Hybrid Rendering
pub mod vsync;       // #18 - VSync Execution
// sdf_text archived - see archive/sdf_text/ (was buggy, future: direct bezier rendering)

// Application framework
pub mod app;         // GpuApp trait and GpuRuntime
pub mod event_loop;  // #149 - GPU-Driven Event Dispatch
pub mod gpu_string;  // #79 - GPU String Processing (tokenization, parsing)
pub mod gpu_app_system; // GPU App System - bytecode VM and app management

// Applications (use GpuApp framework)
pub mod game_of_life;      // Game of Life app
pub mod text_editor;       // Text Editor app
pub mod particles;         // Particle System app
pub mod boids;             // Boids Flocking Simulation
pub mod mandelbrot;        // Mandelbrot Fractal Viewer
pub mod metaballs;         // Metaballs Organic Blob Demo
pub mod waves;             // Wave Simulation app
pub mod benchmark_visual;  // GPU vs CPU Benchmark Demo
pub mod filesystem;        // GPU-Native Filesystem (Issue #19-24)
pub mod content_search;    // GPU Content Search (Issue #50)
pub mod content_pipeline;  // #289 - Content Pipeline with Bounds-Checked Handle Access
pub mod duplicate_finder;  // GPU Duplicate Finder (Issue #51)
pub mod shell;             // GPU Shell - PowerShell-style command line (Issue #126)
pub mod document;          // GPU-Native Document Viewer (Issue #25+)
pub mod document_app;      // GPU-Native Document Browser (GpuApp implementation)
pub mod vector;             // GPU Vector Rasterizer (Issue #34 + #35 + #36)

// Desktop environment (Issue #127-#133)
pub mod screenshot;        // #127 - GPU Screenshot & Visual Testing Infrastructure
pub mod desktop;           // #128-#133 - GPU Desktop Environment

// Demo modules (legacy - standalone implementations)
pub mod ball_physics;   // 1024-Ball Physics demo
