//! Phase 1: Integer Operations Tests (Issue #171)
//!
//! THE GPU IS THE COMPUTER.
//! Integers are just bits - reinterpret don't convert.
//!
//! These tests verify:
//! 1. BytecodeAssembler generates correct instruction encoding
//! 2. Integer opcodes are correctly defined
//! 3. Bit patterns are preserved through float storage

use rust_experiment::gpu_os::gpu_app_system::{
    BytecodeAssembler, BytecodeHeader, BytecodeInst, bytecode_op,
};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode bytecode buffer into header + instructions for inspection
fn decode_bytecode(bytecode: &[u8]) -> (BytecodeHeader, Vec<BytecodeInst>) {
    let header_size = std::mem::size_of::<BytecodeHeader>();
    let inst_size = std::mem::size_of::<BytecodeInst>();

    // Read header
    let header: BytecodeHeader = unsafe {
        std::ptr::read_unaligned(bytecode.as_ptr() as *const BytecodeHeader)
    };

    // Read instructions
    let mut instructions = Vec::new();
    let inst_data = &bytecode[header_size..];
    for i in 0..header.code_size as usize {
        let offset = i * inst_size;
        let inst: BytecodeInst = unsafe {
            std::ptr::read_unaligned(inst_data[offset..].as_ptr() as *const BytecodeInst)
        };
        instructions.push(inst);
    }

    (header, instructions)
}

/// Get the u32 bits from a float immediate
fn imm_as_u32(imm: f32) -> u32 {
    imm.to_bits()
}

/// Get the i32 value from a float immediate (bit reinterpret)
fn imm_as_i32(imm: f32) -> i32 {
    imm.to_bits() as i32
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE CONSTANT TESTS - Verify opcodes match PRD specification
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_integer_arithmetic_opcodes() {
    // Integer arithmetic should be in 0xC0-0xC7 range
    assert_eq!(bytecode_op::INT_ADD, 0xC0);
    assert_eq!(bytecode_op::INT_SUB, 0xC1);
    assert_eq!(bytecode_op::INT_MUL, 0xC2);
    assert_eq!(bytecode_op::INT_DIV_S, 0xC3);
    assert_eq!(bytecode_op::INT_DIV_U, 0xC4);
    assert_eq!(bytecode_op::INT_REM_S, 0xC5);
    assert_eq!(bytecode_op::INT_REM_U, 0xC6);
    assert_eq!(bytecode_op::INT_NEG, 0xC7);
}

#[test]
fn test_bitwise_opcodes() {
    // Bitwise should be in 0xCA-0xCF range
    assert_eq!(bytecode_op::BIT_AND, 0xCA);
    assert_eq!(bytecode_op::BIT_OR, 0xCB);
    assert_eq!(bytecode_op::BIT_XOR, 0xCC);
    assert_eq!(bytecode_op::BIT_NOT, 0xCD);
    assert_eq!(bytecode_op::SHL, 0xCE);
    assert_eq!(bytecode_op::SHR_U, 0xCF);
}

#[test]
fn test_shift_opcodes() {
    // Shifts should be in 0xD0-0xD3 range
    assert_eq!(bytecode_op::SHR_S, 0xD0);
    assert_eq!(bytecode_op::ROTL, 0xD1);
    assert_eq!(bytecode_op::ROTR, 0xD2);
    assert_eq!(bytecode_op::CLZ, 0xD3);
}

#[test]
fn test_comparison_opcodes() {
    // Comparison should be in 0xD4-0xD9 range
    assert_eq!(bytecode_op::INT_EQ, 0xD4);
    assert_eq!(bytecode_op::INT_NE, 0xD5);
    assert_eq!(bytecode_op::INT_LT_S, 0xD6);
    assert_eq!(bytecode_op::INT_LT_U, 0xD7);
    assert_eq!(bytecode_op::INT_LE_S, 0xD8);
    assert_eq!(bytecode_op::INT_LE_U, 0xD9);
}

#[test]
fn test_conversion_opcodes() {
    // Conversion should be in 0xDA-0xDD range
    assert_eq!(bytecode_op::INT_TO_F, 0xDA);
    assert_eq!(bytecode_op::UINT_TO_F, 0xDB);
    assert_eq!(bytecode_op::F_TO_INT, 0xDC);
    assert_eq!(bytecode_op::F_TO_UINT, 0xDD);
}

#[test]
fn test_loadi_opcodes() {
    // Load immediate should be in 0xDE-0xDF range
    assert_eq!(bytecode_op::LOADI_INT, 0xDE);
    assert_eq!(bytecode_op::LOADI_UINT, 0xDF);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASSEMBLER ENCODING TESTS - Verify instruction bytes are correct
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_int_add_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.int_add(10, 8, 9);  // r10 = r8 + r9
    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 2);
    assert_eq!(insts[0].opcode, bytecode_op::INT_ADD);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
    assert_eq!(insts[0].src2, 9);
}

#[test]
fn test_int_neg_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.int_neg(10, 8);  // r10 = -r8
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::INT_NEG);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
    assert_eq!(insts[0].src2, 0);  // Unused for unary ops
}

#[test]
fn test_loadi_int_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, -42);  // r8 = -42
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::LOADI_INT);
    assert_eq!(insts[0].dst, 8);

    // The immediate should contain the bit pattern of -42
    let bits = imm_as_i32(insts[0].imm);
    assert_eq!(bits, -42, "loadi_int should preserve signed int bits");
}

#[test]
fn test_loadi_uint_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0xDEADBEEF);  // r8 = 0xDEADBEEF
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::LOADI_UINT);
    assert_eq!(insts[0].dst, 8);

    // The immediate should contain the bit pattern of 0xDEADBEEF
    let bits = imm_as_u32(insts[0].imm);
    assert_eq!(bits, 0xDEADBEEF, "loadi_uint should preserve unsigned int bits");
}

#[test]
fn test_shl_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.shl(10, 8, 9);  // r10 = r8 << r9
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::SHL);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
    assert_eq!(insts[0].src2, 9);
}

#[test]
fn test_clz_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.clz(10, 8);  // r10 = clz(r8)
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::CLZ);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIT PATTERN PRESERVATION TESTS - Critical for GPU correctness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_signed_int_bit_preservation() {
    // Verify that signed integers round-trip through float storage
    let test_values: &[i32] = &[
        0, 1, -1,
        42, -42,
        i32::MAX, i32::MIN,
        0x7FFFFFFF, -0x80000000i32,
    ];

    for &val in test_values {
        let mut asm = BytecodeAssembler::new();
        asm.loadi_int(8, val);
        asm.halt();

        let bytecode = asm.build(0);
        let (_, insts) = decode_bytecode(&bytecode);

        let recovered = imm_as_i32(insts[0].imm);
        assert_eq!(recovered, val, "Signed int {} should round-trip", val);
    }
}

#[test]
fn test_unsigned_int_bit_preservation() {
    // Verify that unsigned integers round-trip through float storage
    let test_values: &[u32] = &[
        0, 1,
        42,
        0xDEADBEEF,
        0xFFFFFFFF,
        0x80000000,
        0x12345678,
    ];

    for &val in test_values {
        let mut asm = BytecodeAssembler::new();
        asm.loadi_uint(8, val);
        asm.halt();

        let bytecode = asm.build(0);
        let (_, insts) = decode_bytecode(&bytecode);

        let recovered = imm_as_u32(insts[0].imm);
        assert_eq!(recovered, val, "Unsigned int 0x{:08X} should round-trip", val);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROGRAM CONSTRUCTION TESTS - Verify assembler produces valid programs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sum_loop_bytecode() {
    // Generate bytecode for sum 1 to 10
    let mut asm = BytecodeAssembler::new();

    // r8 = counter (1), r9 = sum (0), r10 = limit (11), r11 = 1
    asm.loadi_int(8, 1);
    asm.loadi_int(9, 0);
    asm.loadi_int(10, 11);
    asm.loadi_int(11, 1);

    let loop_start = asm.pc();

    // sum += counter
    asm.int_add(9, 9, 8);

    // counter++
    asm.int_add(8, 8, 11);

    // if counter < limit, continue
    asm.int_lt_s(12, 8, 10);
    asm.jnz(12, loop_start);

    // Store result to state[0]
    asm.loadi_uint(20, 0);
    asm.st(20, 9, 0.0);
    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    // Verify structure
    assert_eq!(header.code_size, 11, "Sum loop should have 11 instructions");

    // Verify loop structure
    assert_eq!(insts[4].opcode, bytecode_op::INT_ADD, "Should have int_add for sum");
    assert_eq!(insts[5].opcode, bytecode_op::INT_ADD, "Should have int_add for increment");
    assert_eq!(insts[6].opcode, bytecode_op::INT_LT_S, "Should have signed comparison");
    assert_eq!(insts[7].opcode, bytecode_op::JNZ, "Should have conditional jump");
}

#[test]
fn test_bitfield_extraction_bytecode() {
    // Generate bytecode for extracting bits 8-15 from a value
    let mut asm = BytecodeAssembler::new();

    // r8 = value, r9 = shift, r10 = mask
    asm.loadi_uint(8, 0xDEADBEEF);
    asm.loadi_uint(9, 8);
    asm.loadi_uint(10, 0xFF);

    // Shift right then mask
    asm.shr_u(11, 8, 9);
    asm.bit_and(12, 11, 10);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 6);
    assert_eq!(insts[3].opcode, bytecode_op::SHR_U, "Should use unsigned shift");
    assert_eq!(insts[4].opcode, bytecode_op::BIT_AND, "Should use AND for masking");
}

#[test]
fn test_conversion_sequence_bytecode() {
    // Generate bytecode for int -> float -> int round trip
    let mut asm = BytecodeAssembler::new();

    asm.loadi_int(8, -42);     // r8 = -42 (as int bits)
    asm.int_to_f(9, 8);        // r9 = -42.0 (as float)
    asm.f_to_int(10, 9);       // r10 = -42 (as int bits again)
    asm.halt();

    let bytecode = asm.build(0);
    let (header, insts) = decode_bytecode(&bytecode);

    assert_eq!(header.code_size, 4);
    assert_eq!(insts[0].opcode, bytecode_op::LOADI_INT);
    assert_eq!(insts[1].opcode, bytecode_op::INT_TO_F);
    assert_eq!(insts[2].opcode, bytecode_op::F_TO_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALL OPCODES COVERAGE TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_integer_ops_emit() {
    // Verify all integer operations can be emitted without panicking
    let mut asm = BytecodeAssembler::new();

    // Arithmetic
    asm.int_add(10, 8, 9);
    asm.int_sub(10, 8, 9);
    asm.int_mul(10, 8, 9);
    asm.int_div_s(10, 8, 9);
    asm.int_div_u(10, 8, 9);
    asm.int_rem_s(10, 8, 9);
    asm.int_rem_u(10, 8, 9);
    asm.int_neg(10, 8);

    // Bitwise
    asm.bit_and(10, 8, 9);
    asm.bit_or(10, 8, 9);
    asm.bit_xor(10, 8, 9);
    asm.bit_not(10, 8);
    asm.shl(10, 8, 9);
    asm.shr_u(10, 8, 9);
    asm.shr_s(10, 8, 9);
    asm.rotl(10, 8, 9);
    asm.rotr(10, 8, 9);
    asm.clz(10, 8);

    // Comparison
    asm.int_eq(10, 8, 9);
    asm.int_ne(10, 8, 9);
    asm.int_lt_s(10, 8, 9);
    asm.int_lt_u(10, 8, 9);
    asm.int_le_s(10, 8, 9);
    asm.int_le_u(10, 8, 9);

    // Conversion
    asm.int_to_f(10, 8);
    asm.uint_to_f(10, 8);
    asm.f_to_int(10, 8);
    asm.f_to_uint(10, 8);

    // Load immediate
    asm.loadi_int(10, -123);
    asm.loadi_uint(10, 456);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _) = decode_bytecode(&bytecode);

    // 30 integer ops + 1 halt
    assert_eq!(header.code_size, 31, "Should have 31 instructions (30 int ops + halt)");
}
