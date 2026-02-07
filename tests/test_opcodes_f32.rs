//! Comprehensive f32 opcode tests
//!
//! Tests all f32 operations with edge cases.
//! f32 is critical because our GPU uses native float32 hardware.
//!
//! NOTE: Most tests are ignored because f32 opcodes return raw bits
//! instead of properly handling float values. See issue #197.

mod harness;

use harness::{
    DifferentialTester, wasm_ops, WasmBuilder,
    build_f32_binop, build_f32_unop, F32_EDGE_CASES,
};

/// Default epsilon for f32 comparisons - allows for minor floating-point differences
const F32_EPSILON: f32 = 1e-6;

// ============ Constants ============

#[test]
fn test_f32_const_basic() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F32_EDGE_CASES {
        let wasm = WasmBuilder::f32_func(&[
            wasm_ops::f32_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.const({})", val));
    }
}

#[test]
fn test_f32_const_special() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let special_values = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0_f32,
        -0.0_f32,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];

    for &val in &special_values {
        let wasm = WasmBuilder::f32_func(&[
            wasm_ops::f32_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.const({:?})", val));
    }
}

// ============ Arithmetic Operations ============

#[test]
fn test_f32_add_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_ADD);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.add({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_sub_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_SUB);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.sub({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_mul_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_MUL);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.mul({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_div_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let non_zero: Vec<f32> = F32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0.0)
        .collect();

    for &a in F32_EDGE_CASES {
        for &b in &non_zero {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_DIV);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.div({}, {})", a, b));
        }
    }
}

// ============ Unary Operations ============

#[test]
fn test_f32_abs_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F32_EDGE_CASES {
        let wasm = build_f32_unop(val, wasm_ops::F32_ABS);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.abs({})", val));
    }
}

#[test]
fn test_f32_neg_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F32_EDGE_CASES {
        let wasm = build_f32_unop(val, wasm_ops::F32_NEG);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.neg({})", val));
    }
}

#[test]
fn test_f32_ceil_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f32, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f32_unop(val, wasm_ops::F32_CEIL);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.ceil({})", val));
    }
}

#[test]
fn test_f32_floor_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f32, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f32_unop(val, wasm_ops::F32_FLOOR);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.floor({})", val));
    }
}

#[test]
fn test_f32_trunc_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f32, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f32_unop(val, wasm_ops::F32_TRUNC);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.trunc({})", val));
    }
}

#[test]
fn test_f32_nearest_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f32, 0.4, 0.5, 0.6, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 3.5,
        -0.0, -0.4, -0.5, -0.6, -1.0, -1.4, -1.5, -1.6, -2.0, -2.5, -3.5,
    ];

    for &val in &test_values {
        let wasm = build_f32_unop(val, wasm_ops::F32_NEAREST);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.nearest({})", val));
    }
}

#[test]
fn test_f32_sqrt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values: Vec<f32> = F32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x >= 0.0)
        .collect();

    for &val in &test_values {
        let wasm = build_f32_unop(val, wasm_ops::F32_SQRT);
        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.sqrt({})", val));
    }
}

// ============ Min/Max Operations ============

#[test]
fn test_f32_min_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_MIN);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.min({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_max_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_MAX);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.max({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_copysign_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [1.0_f32, -1.0, 0.0, -0.0, 100.5, -100.5];

    for &a in &test_values {
        for &b in &test_values {
            let wasm = build_f32_binop(a, b, wasm_ops::F32_COPYSIGN);
            tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.copysign({}, {})", a, b));
        }
    }
}

// ============ Comparison Operations ============

#[test]
fn test_f32_eq_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_EQ);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.eq({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_ne_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_NE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.ne({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_lt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_LT);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.lt({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_gt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_GT);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.gt({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_le_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_LE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.le({}, {})", a, b));
        }
    }
}

#[test]
fn test_f32_ge_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F32_EDGE_CASES {
        for &b in F32_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f32_const(a));
            body.extend(wasm_ops::f32_const(b));
            body.push(wasm_ops::F32_GE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f32.ge({}, {})", a, b));
        }
    }
}

// ============ Conversion Operations ============

#[test]
fn test_i32_trunc_f32_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let safe_values = [
        0.0_f32, 1.0, -1.0, 0.5, -0.5, 0.9, -0.9,
        100.0, -100.0, 1000.5, -1000.5,
        2147483520.0, -2147483520.0,
    ];

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::f32_const(val));
        body.push(wasm_ops::I32_TRUNC_F32_S);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.trunc_f32_s({})", val));
    }
}

#[test]
fn test_i32_trunc_f32_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let safe_values = [
        0.0_f32, 1.0, 0.5, 0.9,
        100.0, 1000.5,
        4294967040.0,
    ];

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::f32_const(val));
        body.push(wasm_ops::I32_TRUNC_F32_U);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.trunc_f32_u({})", val));
    }
}

#[test]
fn test_f32_convert_i32_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::F32_CONVERT_I32_S);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f32_func(&body);

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.convert_i32_s({})", val));
    }
}

#[test]
fn test_f32_convert_i32_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::F32_CONVERT_I32_U);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f32_func(&body);

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.convert_i32_u({})", val));
    }
}

#[test]
fn test_f32_demote_f64_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f64, 1.0, -1.0, 0.5, -0.5,
        100.0, -100.0, 1000.125, -1000.125,
        f32::MAX as f64, f32::MIN as f64,
    ];

    for &val in &test_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::f64_const(val));
        body.push(wasm_ops::F32_DEMOTE_F64);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f32_func(&body);

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.demote_f64({})", val));
    }
}

// ============ Reinterpret Operations ============

#[test]
fn test_i32_reinterpret_f32_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::f32_const(val));
        body.push(wasm_ops::I32_REINTERPRET_F32);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.reinterpret_f32({:?})", val));
    }
}

#[test]
fn test_f32_reinterpret_i32_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let safe_values: Vec<i32> = harness::I32_EDGE_CASES.iter()
        .copied()
        .filter(|&v| {
            let bits = v as u32;
            let exp = (bits >> 23) & 0xFF;
            let frac = bits & 0x7FFFFF;
            !(exp == 0xFF && frac != 0)
        })
        .collect();

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::F32_REINTERPRET_I32);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f32_func(&body);

        tester.assert_same_f32(&wasm, F32_EPSILON, &format!("f32.reinterpret_i32({:#x})", val));
    }
}
