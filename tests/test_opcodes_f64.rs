//! Comprehensive f64 opcode tests
//!
//! Tests all f64 operations with edge cases.
//!
//! NOTE: All tests are ignored because:
//! 1. f64 relies on i64 result reading which is broken (issue #195)
//! 2. f32/f64 opcodes return raw bits instead of proper floats (issue #197)

mod harness;

use harness::{
    DifferentialTester, wasm_ops, WasmBuilder,
    build_f64_binop, build_f64_unop, F64_EDGE_CASES,
};

/// Default epsilon for f64 comparisons
/// THE GPU IS THE COMPUTER - Metal doesn't support native f64, so we use double-single:
/// double-single uses f32 internally, giving ~47 bits mantissa (vs 52 for native f64)
/// f32 precision is ~1e-7, so we use that as our epsilon
const F64_EPSILON: f64 = 1e-6;  // f32 precision limit

// ============ Constants ============

#[test]
fn test_f64_const_basic() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F64_EDGE_CASES {
        let wasm = WasmBuilder::f64_func(&[
            wasm_ops::f64_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.const({})", val));
    }
}

#[test]
fn test_f64_const_special() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let special_values = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0_f64,
        -0.0_f64,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ];

    for &val in &special_values {
        let wasm = WasmBuilder::f64_func(&[
            wasm_ops::f64_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.const({:?})", val));
    }
}

// ============ Arithmetic Operations ============

#[test]
fn test_f64_add_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_ADD);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.add({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_sub_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_SUB);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.sub({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_mul_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_MUL);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.mul({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_div_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let non_zero: Vec<f64> = F64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0.0)
        .collect();

    for &a in F64_EDGE_CASES {
        for &b in &non_zero {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_DIV);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.div({}, {})", a, b));
        }
    }
}

// ============ Unary Operations ============

#[test]
fn test_f64_abs_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F64_EDGE_CASES {
        let wasm = build_f64_unop(val, wasm_ops::F64_ABS);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.abs({})", val));
    }
}

#[test]
fn test_f64_neg_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F64_EDGE_CASES {
        let wasm = build_f64_unop(val, wasm_ops::F64_NEG);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.neg({})", val));
    }
}

#[test]
fn test_f64_ceil_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f64, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f64_unop(val, wasm_ops::F64_CEIL);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.ceil({})", val));
    }
}

#[test]
fn test_f64_floor_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f64, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f64_unop(val, wasm_ops::F64_FLOOR);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.floor({})", val));
    }
}

#[test]
fn test_f64_trunc_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f64, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
        -0.0, -0.1, -0.5, -0.9, -1.0, -1.1, -1.5, -1.9,
        100.4, -100.4, 1000.999, -1000.999,
    ];

    for &val in &test_values {
        let wasm = build_f64_unop(val, wasm_ops::F64_TRUNC);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.trunc({})", val));
    }
}

#[test]
fn test_f64_nearest_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [
        0.0_f64, 0.4, 0.5, 0.6, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 3.5,
        -0.0, -0.4, -0.5, -0.6, -1.0, -1.4, -1.5, -1.6, -2.0, -2.5, -3.5,
    ];

    for &val in &test_values {
        let wasm = build_f64_unop(val, wasm_ops::F64_NEAREST);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.nearest({})", val));
    }
}

#[test]
fn test_f64_sqrt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values: Vec<f64> = F64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x >= 0.0)
        .collect();

    for &val in &test_values {
        let wasm = build_f64_unop(val, wasm_ops::F64_SQRT);
        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.sqrt({})", val));
    }
}

// ============ Min/Max Operations ============

#[test]
fn test_f64_min_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_MIN);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.min({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_max_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_MAX);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.max({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_copysign_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let test_values = [1.0_f64, -1.0, 0.0, -0.0, 100.5, -100.5];

    for &a in &test_values {
        for &b in &test_values {
            let wasm = build_f64_binop(a, b, wasm_ops::F64_COPYSIGN);
            tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.copysign({}, {})", a, b));
        }
    }
}

// ============ Comparison Operations ============
// These return i32, so they might work if f64.const works

#[test]
fn test_f64_eq_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_EQ);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.eq({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_ne_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_NE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.ne({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_lt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_LT);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.lt({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_gt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_GT);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.gt({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_le_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_LE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.le({}, {})", a, b));
        }
    }
}

#[test]
fn test_f64_ge_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in F64_EDGE_CASES {
        for &b in F64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::f64_const(a));
            body.extend(wasm_ops::f64_const(b));
            body.push(wasm_ops::F64_GE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("f64.ge({}, {})", a, b));
        }
    }
}

// ============ Conversion Operations ============

#[test]
fn test_i32_trunc_f64_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let safe_values = [
        0.0_f64, 1.0, -1.0, 0.5, -0.5, 0.9, -0.9,
        100.0, -100.0, 1000.5, -1000.5,
        2147483520.0, -2147483520.0,
    ];

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::f64_const(val));
        body.push(wasm_ops::I32_TRUNC_F64_S);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.trunc_f64_s({})", val));
    }
}

#[test]
fn test_i32_trunc_f64_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let safe_values = [
        0.0_f64, 1.0, 0.5, 0.9,
        100.0, 1000.5,
        4294967040.0,
    ];

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::f64_const(val));
        body.push(wasm_ops::I32_TRUNC_F64_U);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.trunc_f64_u({})", val));
    }
}

#[test]
fn test_f64_convert_i32_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::F64_CONVERT_I32_S);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f64_func(&body);

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.convert_i32_s({})", val));
    }
}

#[test]
fn test_f64_convert_i32_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::F64_CONVERT_I32_U);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f64_func(&body);

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.convert_i32_u({})", val));
    }
}

#[test]
fn test_f64_promote_f32_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::F32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::f32_const(val));
        body.push(wasm_ops::F64_PROMOTE_F32);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f64_func(&body);

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.promote_f32({})", val));
    }
}

// ============ Reinterpret Operations ============
//
// KNOWN LIMITATION: Bit-exact reinterpret operations cannot work with double-single emulation.
//
// ## Why This Is Mathematically Unfixable
//
// IEEE 754 f64 format:
//   - 1 sign bit + 11 exponent bits + 52 mantissa bits = 64 bits total
//   - Example: 0.1 = 0x3FB999999999999A (52 mantissa bits fully utilized)
//
// Double-single format (our GPU emulation):
//   - Two f32 values: hi + lo where |lo| < ulp(hi)
//   - Each f32 has only 23 mantissa bits
//   - Combined precision: ~47 bits (not 52!)
//   - Example: 0.1 becomes 0x3FB99999A0000000 (low 5 bits lost)
//
// The core issue:
//   f64.const(0.1) -> loads IEEE 754 bits -> converts to double-single -> loses 5 bits
//   i64.reinterpret_f64 -> reconstructs f64 bits from double-single -> different bits!
//
// This is NOT a bug - it's a fundamental limitation of emulating 64-bit floats
// using 32-bit hardware. The only fix would be native f64 support in Metal,
// which Apple does not provide.
//
// Arithmetic operations (add, mul, etc.) work fine with double-single because
// we only need VALUE equality (within epsilon). Bit-exact operations require
// REPRESENTATION equality, which is impossible with fewer mantissa bits.

#[test]
#[ignore = "Double-single emulation cannot preserve full IEEE 754 f64 bit patterns (47 vs 52 mantissa bits)"]
fn test_i64_reinterpret_f64_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in F64_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::f64_const(val));
        body.push(wasm_ops::I64_REINTERPRET_F64);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i64_func(&body);

        tester.assert_same_i64(&wasm, &format!("i64.reinterpret_f64({:?})", val));
    }
}

#[test]
fn test_f64_reinterpret_i64_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Use safe i64 values that don't produce NaN
    let safe_values: Vec<i64> = harness::I64_EDGE_CASES.iter()
        .copied()
        .filter(|&v| {
            let bits = v as u64;
            let exp = (bits >> 52) & 0x7FF;
            let frac = bits & 0xFFFFFFFFFFFFF;
            !(exp == 0x7FF && frac != 0)
        })
        .collect();

    for &val in &safe_values {
        let mut body = Vec::new();
        body.extend(wasm_ops::i64_const(val));
        body.push(wasm_ops::F64_REINTERPRET_I64);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::f64_func(&body);

        tester.assert_same_f64(&wasm, F64_EPSILON, &format!("f64.reinterpret_i64({:#x})", val));
    }
}
