//! Tests for transcendental opcodes (Issue #284)
//!
//! These opcodes enable rand/nalgebra compatibility by providing:
//! - LN (natural log) - needed for Normal, Exponential distributions
//! - EXP (e^x) - needed for Poisson, statistical functions
//! - POW, LOG2, EXP2 - general scientific computing
//! - TAN, ASIN, ACOS, ATAN, ATAN2 - trigonometry

mod harness;

use harness::BytecodeExecutor;
use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;
use std::f32::consts::{E, PI};

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon || (a.is_nan() && b.is_nan())
}

// Helper: create bytecode that computes a unary op and stores result to state[3]
fn run_unary_op(exec: &mut BytecodeExecutor, op_fn: fn(&mut BytecodeAssembler, u8, u8) -> usize, input: f32) -> f32 {
    let mut asm = BytecodeAssembler::new();
    asm.loadi(0, input);           // r0 = input
    op_fn(&mut asm, 1, 0);         // r1 = op(r0)
    asm.loadi(2, 0.0);             // r2 = 0 (base address for state)
    asm.st(2, 1, 3.0);             // state[3] = r1 (store result)
    asm.halt();
    let bytecode = asm.build(0);
    exec.run_bytecode_f32_direct(&bytecode).unwrap()
}

// Helper: create bytecode that computes a binary op and stores result to state[3]
fn run_binary_op(exec: &mut BytecodeExecutor, op_fn: fn(&mut BytecodeAssembler, u8, u8, u8) -> usize, a: f32, b: f32) -> f32 {
    let mut asm = BytecodeAssembler::new();
    asm.loadi(0, a);               // r0 = a
    asm.loadi(1, b);               // r1 = b
    op_fn(&mut asm, 2, 0, 1);      // r2 = op(r0, r1)
    asm.loadi(3, 0.0);             // r3 = 0 (base address)
    asm.st(3, 2, 3.0);             // state[3] = r2
    asm.halt();
    let bytecode = asm.build(0);
    exec.run_bytecode_f32_direct(&bytecode).unwrap()
}

// ============ LN (natural log) ============

#[test]
fn test_ln_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // ln(1) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, 1.0);
    assert!(approx_eq(result, 0.0, 1e-6), "ln(1) should be 0, got {}", result);

    // ln(e) = 1
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, E);
    assert!(approx_eq(result, 1.0, 1e-5), "ln(e) should be 1, got {}", result);

    // ln(e^2) = 2
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, E * E);
    assert!(approx_eq(result, 2.0, 1e-5), "ln(e^2) should be 2, got {}", result);
}

#[test]
fn test_ln_edge_cases() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // ln(0) = -inf
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, 0.0);
    assert!(result.is_infinite() && result < 0.0, "ln(0) should be -inf, got {}", result);

    // ln(-1) = NaN
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, -1.0);
    assert!(result.is_nan(), "ln(-1) should be NaN, got {}", result);
}

// ============ EXP (e^x) ============

#[test]
fn test_exp_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // exp(0) = 1
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp, 0.0);
    assert!(approx_eq(result, 1.0, 1e-6), "exp(0) should be 1, got {}", result);

    // exp(1) = e
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp, 1.0);
    assert!(approx_eq(result, E, 1e-5), "exp(1) should be e, got {}", result);

    // exp(2) = e^2
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp, 2.0);
    assert!(approx_eq(result, E * E, 1e-4), "exp(2) should be e^2, got {}", result);
}

#[test]
fn test_exp_ln_inverse() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // exp(ln(x)) = x for various x
    for &x in &[0.5f32, 1.0, 2.0, 10.0, 100.0] {
        let ln_x = x.ln();
        let result = run_unary_op(&mut exec, BytecodeAssembler::exp, ln_x);
        assert!(approx_eq(result, x, 1e-4), "exp(ln({})) should be {}, got {}", x, x, result);
    }
}

// ============ LOG2 ============

#[test]
fn test_log2_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // log2(1) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::log2, 1.0);
    assert!(approx_eq(result, 0.0, 1e-6), "log2(1) should be 0, got {}", result);

    // log2(2) = 1
    let result = run_unary_op(&mut exec, BytecodeAssembler::log2, 2.0);
    assert!(approx_eq(result, 1.0, 1e-6), "log2(2) should be 1, got {}", result);

    // log2(8) = 3
    let result = run_unary_op(&mut exec, BytecodeAssembler::log2, 8.0);
    assert!(approx_eq(result, 3.0, 1e-5), "log2(8) should be 3, got {}", result);
}

// ============ EXP2 ============

#[test]
fn test_exp2_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // exp2(0) = 1
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp2, 0.0);
    assert!(approx_eq(result, 1.0, 1e-6), "exp2(0) should be 1, got {}", result);

    // exp2(3) = 8
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp2, 3.0);
    assert!(approx_eq(result, 8.0, 1e-5), "exp2(3) should be 8, got {}", result);

    // exp2(10) = 1024
    let result = run_unary_op(&mut exec, BytecodeAssembler::exp2, 10.0);
    assert!(approx_eq(result, 1024.0, 1e-3), "exp2(10) should be 1024, got {}", result);
}

// ============ POW ============

#[test]
fn test_pow_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // 2^3 = 8
    let result = run_binary_op(&mut exec, BytecodeAssembler::pow, 2.0, 3.0);
    assert!(approx_eq(result, 8.0, 1e-5), "2^3 should be 8, got {}", result);

    // 3^2 = 9
    let result = run_binary_op(&mut exec, BytecodeAssembler::pow, 3.0, 2.0);
    assert!(approx_eq(result, 9.0, 1e-5), "3^2 should be 9, got {}", result);

    // x^0 = 1
    let result = run_binary_op(&mut exec, BytecodeAssembler::pow, 42.0, 0.0);
    assert!(approx_eq(result, 1.0, 1e-6), "x^0 should be 1, got {}", result);

    // x^1 = x
    let result = run_binary_op(&mut exec, BytecodeAssembler::pow, 7.5, 1.0);
    assert!(approx_eq(result, 7.5, 1e-5), "x^1 should be x, got {}", result);
}

// ============ TAN ============

#[test]
fn test_tan_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // tan(0) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::tan, 0.0);
    assert!(approx_eq(result, 0.0, 1e-6), "tan(0) should be 0, got {}", result);

    // tan(pi/4) = 1
    let result = run_unary_op(&mut exec, BytecodeAssembler::tan, PI / 4.0);
    assert!(approx_eq(result, 1.0, 1e-5), "tan(pi/4) should be 1, got {}", result);
}

// ============ ASIN/ACOS ============

#[test]
fn test_asin_acos_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // asin(0) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::asin, 0.0);
    assert!(approx_eq(result, 0.0, 1e-6), "asin(0) should be 0, got {}", result);

    // asin(1) = pi/2
    let result = run_unary_op(&mut exec, BytecodeAssembler::asin, 1.0);
    assert!(approx_eq(result, PI / 2.0, 1e-5), "asin(1) should be pi/2, got {}", result);

    // acos(1) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::acos, 1.0);
    assert!(approx_eq(result, 0.0, 1e-6), "acos(1) should be 0, got {}", result);

    // acos(0) = pi/2
    let result = run_unary_op(&mut exec, BytecodeAssembler::acos, 0.0);
    assert!(approx_eq(result, PI / 2.0, 1e-5), "acos(0) should be pi/2, got {}", result);
}

// ============ ATAN ============

#[test]
fn test_atan_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // atan(0) = 0
    let result = run_unary_op(&mut exec, BytecodeAssembler::atan, 0.0);
    assert!(approx_eq(result, 0.0, 1e-6), "atan(0) should be 0, got {}", result);

    // atan(1) = pi/4
    let result = run_unary_op(&mut exec, BytecodeAssembler::atan, 1.0);
    assert!(approx_eq(result, PI / 4.0, 1e-5), "atan(1) should be pi/4, got {}", result);
}

// ============ ATAN2 ============

#[test]
fn test_atan2_basic() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // atan2(0, 1) = 0 (point on positive x-axis)
    let result = run_binary_op(&mut exec, BytecodeAssembler::atan2, 0.0, 1.0);
    assert!(approx_eq(result, 0.0, 1e-6), "atan2(0, 1) should be 0, got {}", result);

    // atan2(1, 0) = pi/2 (point on positive y-axis)
    let result = run_binary_op(&mut exec, BytecodeAssembler::atan2, 1.0, 0.0);
    assert!(approx_eq(result, PI / 2.0, 1e-5), "atan2(1, 0) should be pi/2, got {}", result);

    // atan2(1, 1) = pi/4 (45 degrees)
    let result = run_binary_op(&mut exec, BytecodeAssembler::atan2, 1.0, 1.0);
    assert!(approx_eq(result, PI / 4.0, 1e-5), "atan2(1, 1) should be pi/4, got {}", result);
}

// ============ Box-Muller Components Test ============

#[test]
fn test_box_muller_components() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");

    // Box-Muller transform for Normal distribution:
    // z = sqrt(-2 * ln(u)) * cos(2 * pi * v)
    //
    // Test verifies ln works for uniform u in (0,1)

    let u = 0.5f32;
    let expected_ln = u.ln();
    let result = run_unary_op(&mut exec, BytecodeAssembler::ln, u);
    assert!(approx_eq(result, expected_ln, 1e-5),
        "ln({}) should be {}, got {}", u, expected_ln, result);
}
