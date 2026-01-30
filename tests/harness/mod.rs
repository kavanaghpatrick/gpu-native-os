//! Test Harness for Comprehensive Bytecode Testing
//!
//! This module provides the infrastructure for differential testing
//! between GPU bytecode execution and CPU reference (wasmtime).

pub mod wasm_builder;
pub mod executor;
pub mod cpu_reference;
pub mod differential;

pub use wasm_builder::{WasmBuilder, ValType, wasm_ops};
pub use executor::{BytecodeExecutor, ExecutionError, BytecodeStats};
pub use cpu_reference::{CpuReference, CpuError};
pub use differential::{DifferentialTester, DiffError};

/// Standard edge case values for i32 testing
///
/// These values are chosen to catch common bugs:
/// - Boundary values (0, -1, MAX, MIN)
/// - Denormal danger zone (<8M when stored as float bits)
/// - Overflow boundaries
pub const I32_EDGE_CASES: &[i32] = &[
    0,
    1,
    -1,
    2,
    -2,
    7,              // Small positive
    -7,             // Small negative
    64,             // Common denormal issue
    127,            // Max i8
    -128,           // Min i8
    255,            // Max u8
    256,            // Just above u8
    1000,           // Common value
    32767,          // Max i16
    -32768,         // Min i16
    65535,          // Max u16
    65536,          // Just above u16
    8_388_607,      // Max safe integer in f32 mantissa (2^23 - 1)
    8_388_608,      // DENORMAL BOUNDARY - 2^23
    16_777_215,     // Max precise f32 integer (2^24 - 1)
    16_777_216,     // 2^24
    i32::MAX,       // 2147483647
    i32::MIN,       // -2147483648
    i32::MAX - 1,
    i32::MIN + 1,
];

/// Standard edge case values for i64 testing
///
/// Focus on 32-bit boundary crossing which is where our
/// split-register representation can fail.
pub const I64_EDGE_CASES: &[i64] = &[
    0,
    1,
    -1,
    i32::MAX as i64,
    i32::MIN as i64,
    i32::MAX as i64 + 1,    // Just beyond i32
    i32::MIN as i64 - 1,
    u32::MAX as i64,        // Max u32
    u32::MAX as i64 + 1,    // Just beyond u32
    0x0000_0001_0000_0000_i64,  // Bit 32 set (low word = 0)
    0x0000_0000_FFFF_FFFF_i64,  // Low word all 1s
    0xFFFF_FFFF_0000_0000_u64 as i64,  // High word all 1s
    0x8000_0000_0000_0000_u64 as i64,  // Sign bit only
    0x7FFF_FFFF_FFFF_FFFF_i64,  // Max i64
    i64::MAX,
    i64::MIN,
    i64::MAX - 1,
    i64::MIN + 1,
];

/// Standard edge case values for f32 testing
pub const F32_EDGE_CASES: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    0.1,            // Common inexact representation
    0.3,            // 0.1 + 0.1 + 0.1 != 0.3 in float
    f32::MIN_POSITIVE,     // Smallest positive normal
    f32::EPSILON,          // Smallest difference from 1.0
    f32::MAX,
    f32::MIN,              // Most negative
    // Note: We skip infinity and NaN for most tests as behavior may differ
];

/// Standard edge case values for f64 testing
/// THE GPU IS THE COMPUTER - Metal doesn't support native f64, so we use double-single:
/// double-single uses f32 range with extended precision (~47 bits mantissa)
/// Values must be within f32 range (~±3.4e38) to avoid overflow
/// Standard edge case values for f64 testing
/// THE GPU IS THE COMPUTER - Metal doesn't support native f64, so we use double-single:
/// double-single uses f32 range with extended precision (~47 bits mantissa)
/// Values must be within f32 range (~±3.4e38) to avoid overflow
/// Use f32-based constants since double-single can't represent f64-only values
/// Note: Avoid extreme values that could overflow in division (a/b where a is large, b is tiny)
pub const F64_EDGE_CASES: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    0.1,
    f32::EPSILON as f64,       // Use f32 epsilon (~1.2e-7)
    1e15,              // Large but safe: 1e15/1e-15 = 1e30 (within f32 range ~3.4e38)
    -1e15,
    1e-15,             // Small but safe for division
    -1e-15,
];

/// Shift amounts to test for 32-bit operations
/// Includes boundary cases and out-of-range values
pub const SHIFT_AMOUNTS_32: &[i32] = &[
    0, 1, 7, 8, 15, 16, 23, 24, 31,
    32,     // Undefined behavior in WASM (masked to 0)
    33,     // Also masked
    63, 64, // Way out of range
];

/// Shift amounts to test for 64-bit operations
/// Focus on 32-bit boundary
pub const SHIFT_AMOUNTS_64: &[i64] = &[
    0, 1, 7, 8, 15, 16, 23, 24, 31,
    32,     // Critical boundary
    33, 47, 48, 63,
    64,     // Undefined (masked to 0)
    65,
];

/// Values in the "denormal danger zone"
/// When these are stored as float bits, they become denormal floats
/// which may be flushed to zero on some GPUs
pub const DENORMAL_DANGER_VALUES: &[i32] = &[
    1, 2, 3, 4, 5, 6, 7, 8,
    15, 16, 17,
    31, 32, 33,
    63, 64, 65,
    100, 104, 127, 128, 255, 256,
    1000, 4096, 8191, 8192,
    1_000_000, 2_000_000, 4_000_000,
    8_388_607,      // Max before 2^23
    8_388_608,      // Exactly 2^23 - should be safe
];

/// Helper to build WASM for i32 binary operation test
pub fn build_i32_binop(a: i32, b: i32, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::i32_const(a));
    body.extend(wasm_ops::i32_const(b));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::i32_func(&body)
}

/// Helper to build WASM for i32 unary operation test
pub fn build_i32_unop(val: i32, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::i32_const(val));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::i32_func(&body)
}

/// Helper to build WASM for i64 binary operation test
pub fn build_i64_binop(a: i64, b: i64, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::i64_const(a));
    body.extend(wasm_ops::i64_const(b));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::i64_func(&body)
}

/// Helper to build WASM for i64 unary operation test
pub fn build_i64_unop(val: i64, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::i64_const(val));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::i64_func(&body)
}

/// Helper to build WASM for f32 binary operation test
pub fn build_f32_binop(a: f32, b: f32, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::f32_const(a));
    body.extend(wasm_ops::f32_const(b));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::f32_func(&body)
}

/// Helper to build WASM for f32 unary operation test
pub fn build_f32_unop(val: f32, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::f32_const(val));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::f32_func(&body)
}

/// Helper to build WASM for f64 binary operation test
pub fn build_f64_binop(a: f64, b: f64, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::f64_const(a));
    body.extend(wasm_ops::f64_const(b));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::f64_func(&body)
}

/// Helper to build WASM for f64 unary operation test
pub fn build_f64_unop(val: f64, op: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(wasm_ops::f64_const(val));
    body.push(op);
    body.push(wasm_ops::END);
    WasmBuilder::f64_func(&body)
}
