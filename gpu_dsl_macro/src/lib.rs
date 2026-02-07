//! GPU DSL Proc Macro
//!
//! THE GPU IS THE COMPUTER.
//! Write GPU kernels in Rust-like syntax, compile to efficient bytecode.
//!
//! # Example
//!
//! ```ignore
//! use gpu_dsl::gpu_kernel;
//!
//! gpu_kernel! {
//!     fn my_kernel() {
//!         let x = 10;
//!         let y = 20;
//!         STATE[0] = f4((x + y) as f32, 0.0, 0.0, 1.0);
//!     }
//! }
//!
//! // Use it:
//! let bytecode = my_kernel();
//! ```

mod ast;
mod codegen;
mod parser;
mod regalloc;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Generate GPU bytecode from a Rust-like kernel function.
///
/// THE GPU IS THE COMPUTER - this macro compiles to efficient bytecode
/// that runs directly on the GPU via our bytecode VM.
///
/// # Syntax
///
/// ```ignore
/// gpu_kernel! {
///     fn kernel_name() {
///         // Variable declarations
///         let x = 10;
///         let y = 20.0;
///
///         // Arithmetic
///         let sum = x + 5;
///         let prod = y * 2.0;
///
///         // Control flow
///         if sum > 10 {
///             // ...
///         }
///
///         for i in 0..10 {
///             // ...
///         }
///
///         while condition {
///             // ...
///         }
///
///         // Memory access
///         let val = STATE[0];      // Read from state
///         STATE[1] = val;          // Write to state
///
///         // Atomics
///         ATOMIC[0] = 42;          // Atomic store
///         let count = ATOMIC[0];   // Atomic load
///         atomic_add(0, 1);        // Atomic add
///
///         // Graphics
///         emit_quad(pos, size, color, depth);
///
///         // Built-in variables
///         let tid = TID;           // Thread ID
///         let tg_size = TGSIZE;    // Threadgroup size
///     }
/// }
/// ```
///
/// # Types
///
/// Types are inferred from usage:
/// - Integer literals (`10`, `-5`) are `i32`
/// - Float literals (`10.0`, `-5.5`) are `f32`
/// - Use casts for explicit conversion: `x as f32`, `y as i32`
///
/// # Built-in Functions
///
/// - `f4(x, y, z, w)` - Construct float4
/// - `f2(x, y)` - Construct float2
/// - `emit_quad(pos, size, color, depth)` - Emit a quad
/// - `atomic_add(addr, val)` - Atomic add, returns old value
/// - `atomic_inc(addr)` - Atomic increment, returns old value
/// - `sin(x)`, `cos(x)`, `sqrt(x)`, `abs(x)` - Math functions
/// - `min(a, b)`, `max(a, b)` - Min/max
#[proc_macro]
pub fn gpu_kernel(input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as ast::GpuFunction);

    let mut codegen = codegen::CodeGenerator::new();
    let output = codegen.generate(&func);

    output.into()
}
