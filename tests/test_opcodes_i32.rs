//! Comprehensive i32 opcode tests
//!
//! Tests all i32 operations with edge cases to catch bugs
//! through differential testing (GPU vs CPU).

mod harness;

use harness::{
    DifferentialTester, wasm_ops, build_i32_binop, build_i32_unop,
    I32_EDGE_CASES, SHIFT_AMOUNTS_32, DENORMAL_DANGER_VALUES,
};

// ============ Arithmetic Operations ============

#[test]
fn test_i32_add_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_ADD);
            tester.assert_same_i32(&wasm, &format!("i32.add({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_sub_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_SUB);
            tester.assert_same_i32(&wasm, &format!("i32.sub({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_mul_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_MUL);
            tester.assert_same_i32(&wasm, &format!("i32.mul({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_div_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Skip division by zero (trap in WASM)
    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            // Skip MIN / -1 (overflow trap in WASM)
            if a == i32::MIN && b == -1 {
                continue;
            }

            let wasm = build_i32_binop(a, b, wasm_ops::I32_DIV_S);
            tester.assert_same_i32(&wasm, &format!("i32.div_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_div_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_DIV_U);
            tester.assert_same_i32(&wasm, &format!("i32.div_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_rem_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            // Skip MIN % -1 (overflow trap in WASM)
            if a == i32::MIN && b == -1 {
                continue;
            }

            let wasm = build_i32_binop(a, b, wasm_ops::I32_REM_S);
            tester.assert_same_i32(&wasm, &format!("i32.rem_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_rem_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_REM_U);
            tester.assert_same_i32(&wasm, &format!("i32.rem_u({}, {})", a, b));
        }
    }
}

// ============ Bitwise Operations ============

#[test]
fn test_i32_and_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_AND);
            tester.assert_same_i32(&wasm, &format!("i32.and({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i32_or_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_OR);
            tester.assert_same_i32(&wasm, &format!("i32.or({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i32_xor_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_XOR);
            tester.assert_same_i32(&wasm, &format!("i32.xor({:#x}, {:#x})", a, b));
        }
    }
}

#[test]
fn test_i32_shl_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_32 {
            let wasm = build_i32_binop(val, shift, wasm_ops::I32_SHL);
            tester.assert_same_i32(&wasm, &format!("i32.shl({:#x}, {})", val, shift));
        }
    }
}

#[test]
fn test_i32_shr_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_32 {
            let wasm = build_i32_binop(val, shift, wasm_ops::I32_SHR_S);
            tester.assert_same_i32(&wasm, &format!("i32.shr_s({:#x}, {})", val, shift));
        }
    }
}

#[test]
fn test_i32_shr_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        for &shift in SHIFT_AMOUNTS_32 {
            let wasm = build_i32_binop(val, shift, wasm_ops::I32_SHR_U);
            tester.assert_same_i32(&wasm, &format!("i32.shr_u({:#x}, {})", val, shift));
        }
    }
}

#[test]
fn test_i32_rotl_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        for &rotate in SHIFT_AMOUNTS_32 {
            let wasm = build_i32_binop(val, rotate, wasm_ops::I32_ROTL);
            tester.assert_same_i32(&wasm, &format!("i32.rotl({:#x}, {})", val, rotate));
        }
    }
}

#[test]
fn test_i32_rotr_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        for &rotate in SHIFT_AMOUNTS_32 {
            let wasm = build_i32_binop(val, rotate, wasm_ops::I32_ROTR);
            tester.assert_same_i32(&wasm, &format!("i32.rotr({:#x}, {})", val, rotate));
        }
    }
}

// ============ Unary Operations ============

#[test]
fn test_i32_clz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        let wasm = build_i32_unop(val, wasm_ops::I32_CLZ);
        tester.assert_same_i32(&wasm, &format!("i32.clz({:#x})", val));
    }
}

#[test]
fn test_i32_ctz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        let wasm = build_i32_unop(val, wasm_ops::I32_CTZ);
        tester.assert_same_i32(&wasm, &format!("i32.ctz({:#x})", val));
    }
}

#[test]
fn test_i32_popcnt_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        let wasm = build_i32_unop(val, wasm_ops::I32_POPCNT);
        tester.assert_same_i32(&wasm, &format!("i32.popcnt({:#x})", val));
    }
}

// ============ Comparison Operations ============

#[test]
fn test_i32_eqz_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in I32_EDGE_CASES {
        let wasm = build_i32_unop(val, wasm_ops::I32_EQZ);
        tester.assert_same_i32(&wasm, &format!("i32.eqz({})", val));
    }
}

#[test]
fn test_i32_eq_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_EQ);
            tester.assert_same_i32(&wasm, &format!("i32.eq({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_ne_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_NE);
            tester.assert_same_i32(&wasm, &format!("i32.ne({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_lt_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_LT_S);
            tester.assert_same_i32(&wasm, &format!("i32.lt_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_lt_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_LT_U);
            tester.assert_same_i32(&wasm, &format!("i32.lt_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_gt_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_GT_S);
            tester.assert_same_i32(&wasm, &format!("i32.gt_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_gt_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_GT_U);
            tester.assert_same_i32(&wasm, &format!("i32.gt_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_le_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_LE_S);
            tester.assert_same_i32(&wasm, &format!("i32.le_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_le_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_LE_U);
            tester.assert_same_i32(&wasm, &format!("i32.le_u({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_ge_s_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_GE_S);
            tester.assert_same_i32(&wasm, &format!("i32.ge_s({}, {})", a, b));
        }
    }
}

#[test]
fn test_i32_ge_u_exhaustive() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = build_i32_binop(a, b, wasm_ops::I32_GE_U);
            tester.assert_same_i32(&wasm, &format!("i32.ge_u({}, {})", a, b));
        }
    }
}

// ============ Denormal Edge Cases ============

#[test]
fn test_denormal_constants() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    for &val in DENORMAL_DANGER_VALUES {
        let wasm = harness::WasmBuilder::i32_func(&[
            wasm_ops::i32_const(val)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        tester.assert_same_i32(&wasm, &format!("denormal_const({})", val));
    }
}

#[test]
fn test_denormal_arithmetic() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // Test arithmetic on denormal values
    for &val in DENORMAL_DANGER_VALUES {
        // Add 1
        let wasm = build_i32_binop(val, 1, wasm_ops::I32_ADD);
        tester.assert_same_i32(&wasm, &format!("denormal_add({}, 1)", val));

        // Subtract 1
        let wasm = build_i32_binop(val, 1, wasm_ops::I32_SUB);
        tester.assert_same_i32(&wasm, &format!("denormal_sub({}, 1)", val));

        // Multiply by 2
        let wasm = build_i32_binop(val, 2, wasm_ops::I32_MUL);
        tester.assert_same_i32(&wasm, &format!("denormal_mul({}, 2)", val));
    }
}

// ============ Overflow/Underflow Edge Cases ============

#[test]
fn test_i32_overflow_cases() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    // MAX + 1 wraps to MIN
    let wasm = build_i32_binop(i32::MAX, 1, wasm_ops::I32_ADD);
    tester.assert_same_i32(&wasm, "i32.add(MAX, 1)");

    // MIN - 1 wraps to MAX
    let wasm = build_i32_binop(i32::MIN, 1, wasm_ops::I32_SUB);
    tester.assert_same_i32(&wasm, "i32.sub(MIN, 1)");

    // MAX * 2 wraps
    let wasm = build_i32_binop(i32::MAX, 2, wasm_ops::I32_MUL);
    tester.assert_same_i32(&wasm, "i32.mul(MAX, 2)");

    // MIN * 2 wraps
    let wasm = build_i32_binop(i32::MIN, 2, wasm_ops::I32_MUL);
    tester.assert_same_i32(&wasm, "i32.mul(MIN, 2)");

    // -1 * -1 = 1
    let wasm = build_i32_binop(-1, -1, wasm_ops::I32_MUL);
    tester.assert_same_i32(&wasm, "i32.mul(-1, -1)");
}

// ============ Special Bit Patterns ============

#[test]
fn test_special_bit_patterns() {
    let mut tester = DifferentialTester::new().expect("No Metal device");

    let patterns = [
        0x00000000_u32 as i32,  // All zeros
        0xFFFFFFFF_u32 as i32,  // All ones (-1)
        0x55555555_u32 as i32,  // Alternating 01
        0xAAAAAAAA_u32 as i32,  // Alternating 10
        0x0F0F0F0F_u32 as i32,  // Nibble pattern
        0xF0F0F0F0_u32 as i32,  // Inverse nibble
        0x00FF00FF_u32 as i32,  // Byte pattern
        0xFF00FF00_u32 as i32,  // Inverse byte
        0x80000000_u32 as i32,  // Just sign bit
        0x7FFFFFFF_u32 as i32,  // All but sign bit
    ];

    for &a in &patterns {
        for &b in &patterns {
            // AND
            let wasm = build_i32_binop(a, b, wasm_ops::I32_AND);
            tester.assert_same_i32(&wasm, &format!("i32.and({:#010x}, {:#010x})", a as u32, b as u32));

            // OR
            let wasm = build_i32_binop(a, b, wasm_ops::I32_OR);
            tester.assert_same_i32(&wasm, &format!("i32.or({:#010x}, {:#010x})", a as u32, b as u32));

            // XOR
            let wasm = build_i32_binop(a, b, wasm_ops::I32_XOR);
            tester.assert_same_i32(&wasm, &format!("i32.xor({:#010x}, {:#010x})", a as u32, b as u32));
        }
    }
}
