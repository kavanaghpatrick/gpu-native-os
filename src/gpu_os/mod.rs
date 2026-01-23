// GPU-Native OS Modules
// Each module corresponds to a GitHub issue

pub mod kernel;      // #11 - Unified Worker Model
pub mod memory;      // #12 - Memory Architecture
pub mod input;       // #13 - Input Pipeline
pub mod layout;      // #14 - Layout Engine
pub mod widget;      // #15 - Widget System
pub mod text;        // #16 - Text Rendering
pub mod render;      // #17 - Hybrid Rendering
pub mod vsync;       // #18 - VSync Execution

// Demo modules
pub mod game_of_life;   // Game of Life demo
pub mod ball_physics;   // 1024-Ball Physics demo
