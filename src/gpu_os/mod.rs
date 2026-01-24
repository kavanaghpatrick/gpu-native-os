// GPU-Native OS Modules
// Each module corresponds to a GitHub issue

pub mod kernel;      // #11 - Unified Worker Model
pub mod memory;      // #12 - Memory Architecture
pub mod input;       // #13 - Input Pipeline
pub mod layout;      // #14 - Layout Engine
pub mod widget;      // #15 - Widget System
pub mod text;        // #16 - Text Rendering (MSDF - partial)
pub mod text_render; // Simple bitmap font text rendering
pub mod render;      // #17 - Hybrid Rendering
pub mod vsync;       // #18 - VSync Execution

// Application framework
pub mod app;         // GpuApp trait and GpuRuntime

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
pub mod document;          // GPU-Native Document Viewer (Issue #25+)

// Demo modules (legacy - standalone implementations)
pub mod ball_physics;   // 1024-Ball Physics demo
