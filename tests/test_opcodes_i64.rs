//! Comprehensive i64 opcode tests
//!
//! Tests all i64 operations with edge cases.
//! i64 is critical because our GPU represents it as a split (lo, hi) pair.

mod harness;

use harness::{
    DifferentialTester, wasm_ops, WasmBuilder,
    build_i64_binop, build_i64_unop, I64_EDGE_CASES, SHIFT_AMOUNTS_64,
};

// ============ Constants ============

#[test]
fn test_i64_const_basic() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Test basic constants
    for &val in &[0_i64, 1, -1, 42, 1000, i64::MAX, i64::MIN] {
        let wasm = WasmBuilder::i64_func(&[
            wasm_ops::i64_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_i64(&wasm, &format!("i64.const({})", val));
    }
}

#[test]
fn test_i64_const_32bit_boundary() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Critical test: values that cross the 32-bit boundary
    let boundary_values = [
        0x00000000_00000000_i64,  // Zero
        0x00000000_FFFFFFFF_i64,  // Max u32
        0x00000001_00000000_i64,  // Just above u32
        0x7FFFFFFF_FFFFFFFF_i64,  // Max i64
        0xFFFFFFFF_00000000_u64 as i64,  // High word all 1s
        0x80000000_00000000_u64 as i64,  // Just sign bit
        0xFFFFFFFF_FFFFFFFF_u64 as i64,  // -1
    ];

    for &val in &boundary_values {
        let wasm = WasmBuilder::i64_func(&[
            wasm_ops::i64_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_i64(&wasm, &format!("i64.const({:#018x})", val as u64));
    }
}

// ============ Arithmetic Operations ============

#[test]
fn test_i64_add_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_ADD);
            tester.assert_same_i64(&wasm, &format!("i64.add({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_sub_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_SUB);
            tester.assert_same_i64(&wasm, &format!("i64.sub({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_mul_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_MUL);
            tester.assert_same_i64(&wasm, &format!("i64.mul({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_div_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i64> = I64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I64_EDGE_CASES {
        for &b in &divisors {
            // Skip MIN / -1 (overflow)
            if a == i64::MIN && b == -1 {
                continue;
            }

            let wasm = build_i64_binop(a, b, wasm_ops::I64_DIV_S);
            tester.assert_same_i64(&wasm, &format!("i64.div_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_div_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i64> = I64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I64_EDGE_CASES {
        for &b in &divisors {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_DIV_U);
            tester.assert_same_i64(&wasm, &format!("i64.div_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_rem_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i64> = I64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I64_EDGE_CASES {
        for &b in &divisors {
            // Skip MIN % -1 (overflow)
            if a == i64::MIN && b == -1 {
                continue;
            }

            let wasm = build_i64_binop(a, b, wasm_ops::I64_REM_S);
            tester.assert_same_i64(&wasm, &format!("i64.rem_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_rem_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i64> = I64_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I64_EDGE_CASES {
        for &b in &divisors {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_REM_U);
            tester.assert_same_i64(&wasm, &format!("i64.rem_u({}, {})", a, b));
        }
    }
}

// ============ Bitwise Operations ============

#[test]
fn test_i64_and_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_AND);
            tester.assert_same_i64(&wasm, &format!("i64.and({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i64_or_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_OR);
            tester.assert_same_i64(&wasm, &format!("i64.or({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i64_xor_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = build_i64_binop(a, b, wasm_ops::I64_XOR);
            tester.assert_same_i64(&wasm, &format!("i64.xor({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i64_shl_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_64 {
            let wasm = build_i64_binop(val, shift, wasm_ops::I64_SHL);
            tester.assert_same_i64(&wasm, &format!("i64.shl({:#x}, {})", val as u64, shift));
        }
    }
}

#[test]
fn test_i64_shr_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_64 {
            let wasm = build_i64_binop(val, shift, wasm_ops::I64_SHR_S);
            tester.assert_same_i64(&wasm, &format!("i64.shr_s({:#x}, {})", val as u64, shift));
        }
    }
}

#[test]
fn test_i64_shr_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_64 {
            let wasm = build_i64_binop(val, shift, wasm_ops::I64_SHR_U);
            tester.assert_same_i64(&wasm, &format!("i64.shr_u({:#x}, {})", val as u64, shift));
        }
    }
}

#[test]
fn test_i64_rotl_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        for &rotate in SHIFT_AMOUNTS_64 {
            let wasm = build_i64_binop(val, rotate, wasm_ops::I64_ROTL);
            tester.assert_same_i64(&wasm, &format!("i64.rotl({:#x}, {})", val as u64, rotate));
        }
    }
}

#[test]
fn test_i64_rotr_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        for &rotate in SHIFT_AMOUNTS_64 {
            let wasm = build_i64_binop(val, rotate, wasm_ops::I64_ROTR);
            tester.assert_same_i64(&wasm, &format!("i64.rotr({:#x}, {})", val as u64, rotate));
        }
    }
}

// ============ Unary Operations ============

#[test]
fn test_i64_clz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        let wasm = build_i64_unop(val, wasm_ops::I64_CLZ);
        tester.assert_same_i64(&wasm, &format!("i64.clz({:#x})", val as u64));
    }
}

#[test]
fn test_i64_ctz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        let wasm = build_i64_unop(val, wasm_ops::I64_CTZ);
        tester.assert_same_i64(&wasm, &format!("i64.ctz({:#x})", val as u64));
    }
}

#[test]
fn test_i64_popcnt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        let wasm = build_i64_unop(val, wasm_ops::I64_POPCNT);
        tester.assert_same_i64(&wasm, &format!("i64.popcnt({:#x})", val as u64));
    }
}

// ============ Comparison Operations ============

#[test]
fn test_i64_eqz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        // i64.eqz returns i32, so we need special handling
        let mut body = Vec::new();
        body.extend(wasm_ops::i64_const(val));
        body.push(wasm_ops::I64_EQZ);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);  // Returns i32!

        tester.assert_same_i32(&wasm, &format!("i64.eqz({})", val));
    }
}

#[test]
fn test_i64_eq_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            // i64 comparisons return i32
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_EQ);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.eq({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_ne_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_NE);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.ne({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_lt_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_LT_S);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.lt_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_lt_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_LT_U);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.lt_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_gt_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_GT_S);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.gt_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_gt_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_GT_U);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.gt_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_le_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_LE_S);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.le_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_le_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_LE_U);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.le_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_ge_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_GE_S);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.ge_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i64_ge_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let mut body = Vec::new();
            body.extend(wasm_ops::i64_const(a));
            body.extend(wasm_ops::i64_const(b));
            body.push(wasm_ops::I64_GE_U);
            body.push(wasm_ops::END);
            let wasm = WasmBuilder::i32_func(&body);

            tester.assert_same_i32(&wasm, &format!("i64.ge_u({}, {})", a, b));
        }
    }
}

// ============ Conversion Operations ============

#[test]
fn test_i32_wrap_i64_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I64_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i64_const(val));
        body.push(wasm_ops::I32_WRAP_I64);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i32_func(&body);

        tester.assert_same_i32(&wasm, &format!("i32.wrap_i64({:#x})", val as u64));
    }
}

#[test]
fn test_i64_extend_i32_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::I64_EXTEND_I32_S);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i64_func(&body);

        tester.assert_same_i64(&wasm, &format!("i64.extend_i32_s({})", val));
    }
}

#[test]
fn test_i64_extend_i32_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in harness::I32_EDGE_CASES {
        let mut body = Vec::new();
        body.extend(wasm_ops::i32_const(val));
        body.push(wasm_ops::I64_EXTEND_I32_U);
        body.push(wasm_ops::END);
        let wasm = WasmBuilder::i64_func(&body);

        tester.assert_same_i64(&wasm, &format!("i64.extend_i32_u({})", val));
    }
}

// ============ 32-bit Boundary Crossing Tests ============

#[test]
fn test_i64_add_carry_across_32bit() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Tests that specifically cause carry from low word to high word
    let test_cases = [
        (0xFFFFFFFF_i64, 1_i64),           // 0xFFFFFFFF + 1 = 0x100000000
        (0x7FFFFFFF_i64, 0x80000001_i64),  // Carry into bit 31
        (0xFFFFFFFF_FFFFFFFF_u64 as i64, 1),  // -1 + 1 = 0
    ];

    for (a, b) in test_cases {
        let wasm = build_i64_binop(a, b, wasm_ops::I64_ADD);
        tester.assert_same_i64(&wasm, &format!("i64.add({:#x}, {:#x})", a as u64, b as u64));
    }
}

#[test]
fn test_i64_mul_overflow_32bit() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Multiplication that overflows 32 bits
    let test_cases = [
        (0x10000_i64, 0x10000_i64),     // 65536 * 65536 = 4294967296
        (0xFFFFFFFF_i64, 2_i64),         // MAX_U32 * 2
        (0x80000000_i64, 2_i64),         // 2^31 * 2
    ];

    for (a, b) in test_cases {
        let wasm = build_i64_binop(a, b, wasm_ops::I64_MUL);
        tester.assert_same_i64(&wasm, &format!("i64.mul({:#x}, {:#x})", a as u64, b as u64));
    }
}

#[test]
fn test_i64_shift_across_32bit() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Shifts that move bits across the 32-bit boundary
    let test_cases = [
        (1_i64, 32_i64),               // 1 << 32 = 0x100000000
        (1_i64, 63_i64),               // 1 << 63 = sign bit
        (0xFFFFFFFF_i64, 32_i64),       // Move low word to high
        (0x8000000000000000_u64 as i64, 1_i64),  // Shift out sign bit
    ];

    for (val, shift) in test_cases {
        let wasm = build_i64_binop(val, shift, wasm_ops::I64_SHL);
        tester.assert_same_i64(&wasm, &format!("i64.shl({:#x}, {})", val as u64, shift));
    }
}
