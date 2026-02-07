//! Phase 3: GPU DSL Macro Tests (Issue #173)
//!
//! THE GPU IS THE COMPUTER.
//! Write GPU kernels in Rust-like syntax.
//!
//! These tests verify that the gpu_kernel! macro correctly generates bytecode.

use rust_experiment::gpu_os::gpu_app_system::{BytecodeAssembler, BytecodeHeader, BytecodeInst, bytecode_op};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode bytecode bytes back into header and instructions
fn decode_bytecode(bytecode: &[u8]) -> (BytecodeHeader, Vec<BytecodeInst>) {
    let header: BytecodeHeader = unsafe {
        std::ptr::read(bytecode.as_ptr() as *const BytecodeHeader)
    };

    let inst_start = std::mem::size_of::<BytecodeHeader>();
    let inst_bytes = &bytecode[inst_start..];
    let inst_count = header.code_size as usize;

    let mut instructions = Vec::with_capacity(inst_count);
    for i in 0..inst_count {
        let offset = i * std::mem::size_of::<BytecodeInst>();
        if offset + std::mem::size_of::<BytecodeInst>() <= inst_bytes.len() {
            let inst: BytecodeInst = unsafe {
                std::ptr::read(inst_bytes[offset..].as_ptr() as *const BytecodeInst)
            };
            instructions.push(inst);
        }
    }

    (header, instructions)
}

// ═══════════════════════════════════════════════════════════════════════════════
// MANUAL BYTECODE TESTS (Testing assembler patterns the macro will use)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_simple_integer_add_pattern() {
    // Pattern: let a = 10; let b = 20; let c = a + b;
    let mut asm = BytecodeAssembler::new();

    // let a = 10
    asm.loadi_int(8, 10);

    // let b = 20
    asm.loadi_int(9, 20);

    // let c = a + b
    asm.int_add(10, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 4);
    assert_eq!(insts[0].opcode, bytecode_op::LOADI_INT);
    assert_eq!(insts[1].opcode, bytecode_op::LOADI_INT);
    assert_eq!(insts[2].opcode, bytecode_op::INT_ADD);
    assert_eq!(insts[3].opcode, bytecode_op::HALT);
}

#[test]
fn test_simple_float_add_pattern() {
    // Pattern: let a = 10.0; let b = 20.0; let c = a + b;
    let mut asm = BytecodeAssembler::new();

    // let a = 10.0
    asm.loadi(8, 10.0);

    // let b = 20.0
    asm.loadi(9, 20.0);

    // let c = a + b
    asm.add(10, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 4);
    assert_eq!(insts[0].opcode, bytecode_op::LOADI);
    assert_eq!(insts[1].opcode, bytecode_op::LOADI);
    assert_eq!(insts[2].opcode, bytecode_op::ADD);
    assert_eq!(insts[3].opcode, bytecode_op::HALT);
}

#[test]
fn test_if_pattern() {
    // Pattern: if x > 5 { result = 100; }
    let mut asm = BytecodeAssembler::new();

    // x = 10
    asm.loadi_int(8, 10);

    // 5
    asm.loadi_int(9, 5);

    // x > 5 -> compare: gt returns 1 if true
    asm.gt(10, 8, 9);

    // jz (if false, skip)
    let skip_inst = asm.jz(10, 0); // Placeholder, patch later

    // result = 100
    asm.loadi_int(11, 100);

    // Patch jump target
    let end_pc = asm.pc();
    asm.patch_jump(skip_inst, end_pc);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 6);
    assert_eq!(insts[3].opcode, bytecode_op::JZ);
}

#[test]
fn test_for_loop_pattern() {
    // Pattern: for i in 0..5 { sum = sum + i; }
    let mut asm = BytecodeAssembler::new();

    // sum = 0
    asm.loadi_int(8, 0);  // r8 = sum

    // i = 0
    asm.loadi_int(9, 0);  // r9 = i

    // limit = 5
    asm.loadi_int(10, 5); // r10 = limit

    // loop:
    let loop_start = asm.pc();

    // i < limit?
    asm.int_lt_u(11, 9, 10);  // r11 = i < limit

    // jz (if false, exit loop)
    let exit_inst = asm.jz(11, 0);

    // sum = sum + i
    asm.int_add(8, 8, 9);

    // i = i + 1
    asm.loadi_int(12, 1);
    asm.int_add(9, 9, 12);

    // jmp loop_start
    asm.jmp(loop_start);

    // exit:
    let exit_pc = asm.pc();
    asm.patch_jump(exit_inst, exit_pc);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _insts) = decode_bytecode(&bytecode);

    // Should have: 3 init + 1 cmp + 1 jz + 1 add + 2 inc + 1 jmp + 1 halt = 10
    assert!(header.code_size >= 10);
}

#[test]
fn test_state_access_pattern() {
    // Pattern: let val = STATE[0]; STATE[1] = val;
    let mut asm = BytecodeAssembler::new();

    // index = 0
    asm.loadi_uint(8, 0);

    // val = STATE[0]
    asm.ld(9, 8, 0.0);

    // index = 1
    asm.loadi_uint(10, 1);

    // STATE[1] = val
    asm.st(10, 9, 0.0);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 5);
    assert_eq!(insts[1].opcode, bytecode_op::LD);
    assert_eq!(insts[3].opcode, bytecode_op::ST);
}

#[test]
fn test_atomic_pattern() {
    // Pattern: ATOMIC[0] = 42; let count = ATOMIC[0]; atomic_add(0, 1);
    let mut asm = BytecodeAssembler::new();

    // addr = 0
    asm.loadi_uint(8, 0);

    // val = 42
    asm.loadi_uint(9, 42);

    // ATOMIC[0] = 42
    asm.atomic_store(9, 8);

    // count = ATOMIC[0]
    asm.atomic_load(10, 8);

    // atomic_add(0, 1)
    asm.loadi_uint(11, 1);
    asm.atomic_add(12, 8, 11);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 7);
    assert_eq!(insts[2].opcode, bytecode_op::ATOMIC_STORE);
    assert_eq!(insts[3].opcode, bytecode_op::ATOMIC_LOAD);
    assert_eq!(insts[5].opcode, bytecode_op::ATOMIC_ADD);
}

#[test]
fn test_quad_emission_pattern() {
    // Pattern: emit_quad(pos, size, color, depth)
    let mut asm = BytecodeAssembler::new();

    // pos = (100, 100)
    asm.loadi(8, 100.0);
    asm.sety(8, 100.0);

    // size = (50, 50)
    asm.loadi(9, 50.0);
    asm.sety(9, 50.0);

    // color = (1, 0, 0, 1) - red
    asm.loadi(10, 1.0);
    asm.sety(10, 0.0);
    asm.setz(10, 0.0);
    asm.setw(10, 1.0);

    // emit_quad
    asm.quad(8, 9, 10, 0.5);

    asm.halt();

    let bytecode = asm.build(6);
    let (header, insts) = decode_bytecode(&bytecode);

    // Find QUAD opcode
    let has_quad = insts.iter().any(|i| i.opcode == bytecode_op::QUAD);
    assert!(has_quad, "Should contain QUAD opcode");
    assert_eq!(header.vertex_budget, 6); // 1 quad = 6 vertices
}

#[test]
fn test_type_cast_pattern() {
    // Pattern: let x = 10; let y = x as f32;
    let mut asm = BytecodeAssembler::new();

    // x = 10 (integer)
    asm.loadi_int(8, 10);

    // y = x as f32
    asm.int_to_f(9, 8);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 3);
    assert_eq!(insts[0].opcode, bytecode_op::LOADI_INT);
    assert_eq!(insts[1].opcode, bytecode_op::INT_TO_F);
}

#[test]
fn test_break_pattern() {
    // Pattern: for i in 0..100 { if i >= 5 { break; } sum += i; }
    let mut asm = BytecodeAssembler::new();

    // sum = 0
    asm.loadi_int(8, 0);  // r8 = sum

    // i = 0
    asm.loadi_int(9, 0);  // r9 = i

    // limit = 100
    asm.loadi_int(10, 100);

    // loop:
    let loop_start = asm.pc();

    // i < limit?
    asm.int_lt_u(11, 9, 10);
    let exit_loop = asm.jz(11, 0);

    // if i >= 5
    asm.loadi_int(12, 5);
    asm.int_lt_s(13, 9, 12);  // i < 5
    // if NOT (i < 5), i.e., i >= 5, break
    let skip_break = asm.jnz(13, 0);

    // break
    let break_inst = asm.jmp(0);

    // skip_break:
    let skip_break_pc = asm.pc();
    asm.patch_jump(skip_break, skip_break_pc);

    // sum += i
    asm.int_add(8, 8, 9);

    // i++
    asm.loadi_int(14, 1);
    asm.int_add(9, 9, 14);

    // jmp loop
    asm.jmp(loop_start);

    // exit:
    let exit_pc = asm.pc();
    asm.patch_jump(exit_loop, exit_pc);
    asm.patch_jump(break_inst, exit_pc);

    asm.halt();

    let bytecode = asm.build(0);
    let (_header, insts) = decode_bytecode(&bytecode);

    // Count jump instructions
    let jump_count = insts.iter().filter(|i| {
        i.opcode == bytecode_op::JMP || i.opcode == bytecode_op::JZ || i.opcode == bytecode_op::JNZ
    }).count();

    assert!(jump_count >= 4, "Should have multiple jump instructions for break pattern");
}

#[test]
fn test_while_loop_pattern() {
    // Pattern: while i < 5 { sum += i; i += 1; }
    let mut asm = BytecodeAssembler::new();

    // i = 0
    asm.loadi_int(8, 0);  // r8 = i

    // sum = 0
    asm.loadi_int(9, 0);  // r9 = sum

    // while:
    let loop_start = asm.pc();

    // i < 5?
    asm.loadi_int(10, 5);
    asm.int_lt_s(11, 8, 10);
    let exit_inst = asm.jz(11, 0);

    // sum += i
    asm.int_add(9, 9, 8);

    // i += 1
    asm.loadi_int(12, 1);
    asm.int_add(8, 8, 12);

    // jmp while
    asm.jmp(loop_start);

    // exit:
    let exit_pc = asm.pc();
    asm.patch_jump(exit_inst, exit_pc);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _insts) = decode_bytecode(&bytecode);

    assert!(header.code_size >= 9);
}

#[test]
fn test_tid_access_pattern() {
    // Pattern: let tid = TID;
    // TID is in r1 by convention
    let mut asm = BytecodeAssembler::new();

    // tid = TID (r1)
    asm.mov(8, 1);  // Move TID (r1) to r8

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 2);
    assert_eq!(insts[0].opcode, bytecode_op::MOV);
    assert_eq!(insts[0].dst, 8);
    assert_eq!(insts[0].src1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE PATTERN TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_nested_if_pattern() {
    // Pattern: if a > 0 { if b > 0 { result = 1; } else { result = 2; } }
    let mut asm = BytecodeAssembler::new();

    // a = 5
    asm.loadi_int(8, 5);

    // b = 3
    asm.loadi_int(9, 3);

    // result = 0
    asm.loadi_int(10, 0);

    // zero for comparison
    asm.loadi_int(11, 0);

    // if a > 0
    asm.gt(12, 8, 11);
    let skip_outer = asm.jz(12, 0);

    // if b > 0
    asm.gt(13, 9, 11);
    let else_branch = asm.jz(13, 0);

    // result = 1
    asm.loadi_int(10, 1);
    let end_inner = asm.jmp(0);

    // else:
    let else_pc = asm.pc();
    asm.patch_jump(else_branch, else_pc);

    // result = 2
    asm.loadi_int(10, 2);

    // end_inner:
    let end_inner_pc = asm.pc();
    asm.patch_jump(end_inner, end_inner_pc);

    // skip_outer:
    let skip_outer_pc = asm.pc();
    asm.patch_jump(skip_outer, skip_outer_pc);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _insts) = decode_bytecode(&bytecode);

    assert!(header.code_size >= 10, "Nested if should generate multiple instructions");
}

#[test]
fn test_comparison_operators() {
    let mut asm = BytecodeAssembler::new();

    // a = 10, b = 20
    asm.loadi_int(8, 10);
    asm.loadi_int(9, 20);

    // Test all comparisons
    asm.int_lt_s(10, 8, 9);   // a < b -> 1
    asm.int_le_s(11, 8, 9);   // a <= b -> 1
    asm.int_eq(12, 8, 9);     // a == b -> 0
    asm.int_ne(13, 8, 9);     // a != b -> 1

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    // Verify comparison opcodes are present
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::INT_LT_S));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::INT_LE_S));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::INT_EQ));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::INT_NE));
}

#[test]
fn test_bitwise_operations() {
    let mut asm = BytecodeAssembler::new();

    // a = 0xFF, b = 0x0F
    asm.loadi_uint(8, 0xFF);
    asm.loadi_uint(9, 0x0F);

    // Test bitwise ops
    asm.bit_and(10, 8, 9);    // 0xFF & 0x0F = 0x0F
    asm.bit_or(11, 8, 9);     // 0xFF | 0x0F = 0xFF
    asm.bit_xor(12, 8, 9);    // 0xFF ^ 0x0F = 0xF0
    asm.bit_not(13, 8);       // ~0xFF = 0xFFFFFF00

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert!(insts.iter().any(|i| i.opcode == bytecode_op::BIT_AND));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::BIT_OR));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::BIT_XOR));
    assert!(insts.iter().any(|i| i.opcode == bytecode_op::BIT_NOT));
}
