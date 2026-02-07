//! Smoke tests for the test harness
//!
//! These tests verify the harness itself works before we use it
//! for comprehensive opcode testing.

mod harness;

use harness::{
    WasmBuilder, wasm_ops, BytecodeExecutor, CpuReference, DifferentialTester,
    I32_EDGE_CASES, build_i32_binop,
};

// ============ WasmBuilder Tests ============

#[test]
fn test_wasm_builder_i32_const() {
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(42)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    // Verify it's valid WASM (magic number)
    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]);
    // Verify version
    assert_eq!(&wasm[4..8], &[0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn test_wasm_builder_negative_const() {
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(-1)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]);
}

#[test]
fn test_wasm_builder_i64_const() {
    let wasm = WasmBuilder::i64_func(&[
        wasm_ops::i64_const(0x123456789ABCDEF0_i64)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]);
}

#[test]
fn test_wasm_builder_f32_const() {
    let wasm = WasmBuilder::f32_func(&[
        wasm_ops::f32_const(3.14159)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]);
}

// ============ CPU Reference Tests ============

#[test]
fn test_cpu_reference_i32_const() {
    let cpu = CpuReference::new();
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(42)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    let result = cpu.run_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_cpu_reference_i32_add() {
    let cpu = CpuReference::new();
    let wasm = build_i32_binop(10, 32, wasm_ops::I32_ADD);

    let result = cpu.run_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_cpu_reference_i32_sub() {
    let cpu = CpuReference::new();
    let wasm = build_i32_binop(50, 8, wasm_ops::I32_SUB);

    let result = cpu.run_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_cpu_reference_i32_mul() {
    let cpu = CpuReference::new();
    let wasm = build_i32_binop(6, 7, wasm_ops::I32_MUL);

    let result = cpu.run_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_cpu_reference_negative() {
    let cpu = CpuReference::new();
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(-1)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    let result = cpu.run_i32(&wasm).unwrap();
    assert_eq!(result, -1);
}

#[test]
fn test_cpu_reference_i64() {
    let cpu = CpuReference::new();
    let wasm = WasmBuilder::i64_func(&[
        wasm_ops::i64_const(0x123456789ABCDEF0_i64)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    let result = cpu.run_i64(&wasm).unwrap();
    assert_eq!(result, 0x123456789ABCDEF0_i64);
}

// ============ GPU Executor Tests ============

#[test]
fn test_gpu_executor_i32_const() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(42)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    let result = exec.run_wasm_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_gpu_executor_i32_add() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");
    let wasm = build_i32_binop(10, 32, wasm_ops::I32_ADD);

    let result = exec.run_wasm_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn test_gpu_executor_negative() {
    let mut exec = BytecodeExecutor::new().expect("No Metal device");
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(-1)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    let result = exec.run_wasm_i32(&wasm).unwrap();
    assert_eq!(result, -1);
}

// ============ Differential Tests ============

#[test]
fn test_differential_i32_const() {
    let mut tester = DifferentialTester::new().expect("No Metal device");
    let wasm = WasmBuilder::i32_func(&[
        wasm_ops::i32_const(42)[..].to_vec(),
        vec![wasm_ops::END],
    ].concat());

    tester.assert_same_i32(&wasm, "i32.const 42");
}

#[test]
fn test_differential_i32_add() {
    let mut tester = DifferentialTester::new().expect("No Metal device");
    let wasm = build_i32_binop(10, 32, wasm_ops::I32_ADD);

    tester.assert_same_i32(&wasm, "i32.add(10, 32)");
}

#[test]
fn test_differential_i32_sub() {
    let mut tester = DifferentialTester::new().expect("No Metal device");
    let wasm = build_i32_binop(50, 8, wasm_ops::I32_SUB);

    tester.assert_same_i32(&wasm, "i32.sub(50, 8)");
}

#[test]
fn test_differential_i32_mul() {
    let mut tester = DifferentialTester::new().expect("No Metal device");
    let wasm = build_i32_binop(6, 7, wasm_ops::I32_MUL);

    tester.assert_same_i32(&wasm, "i32.mul(6, 7)");
}

#[test]
fn test_differential_edge_cases_sample() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Test a sample of edge cases
    for &val in &[0, 1, -1, 42, i32::MAX, i32::MIN] {
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_i32(&wasm, &format!("i32.const {}", val));
    }
}

#[test]
fn test_differential_denormal_constants() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Test values in the denormal danger zone
    for &val in &[1, 2, 64, 100, 127, 255, 1000, 8_388_607] {
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_i32(&wasm, &format!("denormal constant {}", val));
    }
}

// ============ Edge Case Constants Tests ============

#[test]
fn test_edge_cases_defined() {
    // Verify edge case arrays are non-empty
    assert!(!I32_EDGE_CASES.is_empty());
    assert!(I32_EDGE_CASES.contains(&0));
    assert!(I32_EDGE_CASES.contains(&i32::MAX));
    assert!(I32_EDGE_CASES.contains(&i32::MIN));
}

#[test]
fn test_build_helpers() {
    // Test the helper functions compile and produce valid WASM
    let wasm1 = build_i32_binop(1, 2, wasm_ops::I32_ADD);
    assert!(!wasm1.is_empty());

    let wasm2 = harness::build_i32_unop(42, wasm_ops::I32_CLZ);
    assert!(!wasm2.is_empty());
}
