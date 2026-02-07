//! GPU DSL - Write GPU kernels in Rust syntax
//!
//! THE GPU IS THE COMPUTER.
//!
//! This crate provides the `gpu_kernel!` macro for writing GPU programs
//! in a Rust-like syntax that compiles to efficient bytecode.
//!
//! # Example
//!
//! ```ignore
//! use gpu_dsl::gpu_kernel;
//!
//! gpu_kernel! {
//!     fn particle_update() {
//!         let tid = TID;
//!         let pos = STATE[tid * 2];
//!         let vel = STATE[tid * 2 + 1];
//!
//!         // Simple physics
//!         let new_pos = pos + vel;
//!         STATE[tid * 2] = new_pos;
//!
//!         // Emit visual
//!         emit_quad(pos, f4(4.0, 4.0, 0.0, 0.0), f4(1.0, 0.0, 0.0, 1.0), 0.5);
//!     }
//! }
//!
//! fn main() {
//!     let bytecode = particle_update();
//!     // Load bytecode into GPU VM...
//! }
//! ```
//!
//! # Features
//!
//! - **Rust-like syntax**: let bindings, if/else, for/while loops
//! - **Type inference**: Integers vs floats automatically detected
//! - **Compile-time checking**: Errors for unsupported patterns
//! - **Efficient codegen**: Direct bytecode generation via proc macro
//!
//! # Limitations (GPU constraints)
//!
//! - No heap allocation (no Vec, Box, etc.)
//! - No recursion
//! - No closures with captures
//! - No strings (except state memory)
//! - Limited to 16 local variables before spilling

pub use gpu_dsl_macro::gpu_kernel;

// Re-export types from rust-experiment that users might need
pub use rust_experiment::gpu_os::gpu_app_system::{BytecodeAssembler, bytecode_op};
