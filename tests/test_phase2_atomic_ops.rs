//! Phase 2: Atomic Operations Tests (Issue #172)
//!
//! THE GPU IS THE COMPUTER.
//! Atomics are NOT for locks. Atomics are for LOCK-FREE COORDINATION.
//! GPU NEVER WAITS. GPU NEVER BLOCKS.
//!
//! These tests verify:
//! - Atomic load/store
//! - Atomic read-modify-write (add, sub, max, min, and, or, xor)
//! - Atomic compare-and-swap
//! - Atomic increment/decrement
//! - Queue slot claiming pattern

use rust_experiment::gpu_os::gpu_app_system::{bytecode_op, BytecodeAssembler, BytecodeInst, BytecodeHeader};

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
        let inst: BytecodeInst = unsafe {
            std::ptr::read(inst_bytes[offset..].as_ptr() as *const BytecodeInst)
        };
        instructions.push(inst);
    }

    (header, instructions)
}

/// Extract immediate as u32 bits
fn imm_as_u32(imm: f32) -> u32 {
    imm.to_bits()
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE CONSTANT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_opcode_values() {
    // Verify all atomic opcodes are in 0xE0-0xEE range
    assert_eq!(bytecode_op::ATOMIC_LOAD, 0xE0);
    assert_eq!(bytecode_op::ATOMIC_STORE, 0xE1);
    assert_eq!(bytecode_op::ATOMIC_ADD, 0xE2);
    assert_eq!(bytecode_op::ATOMIC_SUB, 0xE3);
    assert_eq!(bytecode_op::ATOMIC_MAX_U, 0xE4);
    assert_eq!(bytecode_op::ATOMIC_MIN_U, 0xE5);
    assert_eq!(bytecode_op::ATOMIC_MAX_S, 0xE6);
    assert_eq!(bytecode_op::ATOMIC_MIN_S, 0xE7);
    assert_eq!(bytecode_op::ATOMIC_AND, 0xE8);
    assert_eq!(bytecode_op::ATOMIC_OR, 0xE9);
    assert_eq!(bytecode_op::ATOMIC_XOR, 0xEA);
    assert_eq!(bytecode_op::ATOMIC_CAS, 0xEB);
    assert_eq!(bytecode_op::ATOMIC_INC, 0xEC);
    assert_eq!(bytecode_op::ATOMIC_DEC, 0xED);
    assert_eq!(bytecode_op::MEM_FENCE, 0xEE);
}

#[test]
fn test_atomic_opcodes_dont_overlap_with_integers() {
    // Ensure atomic range (0xE0-0xEE) doesn't overlap with integer range (0xC0-0xDF)
    assert!(bytecode_op::ATOMIC_LOAD > bytecode_op::LOADI_UINT);
    assert_eq!(bytecode_op::LOADI_UINT, 0xDF);
    assert_eq!(bytecode_op::ATOMIC_LOAD, 0xE0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASSEMBLER ENCODING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_load_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_load(10, 8);  // dst=r10, addr_reg=r8
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_LOAD);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
}

#[test]
fn test_atomic_store_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_store(9, 8);  // val_reg=r9, addr_reg=r8
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_STORE);
    assert_eq!(insts[0].src1, 9);  // val_reg
    assert_eq!(insts[0].src2, 8);  // addr_reg
}

#[test]
fn test_atomic_add_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_add(10, 8, 9);  // dst=r10, addr_reg=r8, val_reg=r9
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_ADD);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);   // addr_reg
    assert_eq!(insts[0].src2, 9);   // val_reg
}

#[test]
fn test_atomic_cas_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_cas(10, 8, 9, 0xDEADBEEF);  // dst=r10, addr=r8, expected=r9, desired=0xDEADBEEF
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_CAS);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);    // addr_reg
    assert_eq!(insts[0].src2, 9);    // expected_reg
    assert_eq!(imm_as_u32(insts[0].imm), 0xDEADBEEF, "CAS desired value should be preserved in imm");
}

#[test]
fn test_atomic_inc_dec_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_inc(10, 8);
    asm.atomic_dec(11, 8);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_INC);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);

    assert_eq!(insts[1].opcode, bytecode_op::ATOMIC_DEC);
    assert_eq!(insts[1].dst, 11);
    assert_eq!(insts[1].src1, 8);
}

#[test]
fn test_mem_fence_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.mem_fence();
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::MEM_FENCE);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BITWISE ATOMIC ENCODING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_bitwise_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_and(10, 8, 9);
    asm.atomic_or(11, 8, 9);
    asm.atomic_xor(12, 8, 9);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_AND);
    assert_eq!(insts[1].opcode, bytecode_op::ATOMIC_OR);
    assert_eq!(insts[2].opcode, bytecode_op::ATOMIC_XOR);
}

#[test]
fn test_atomic_max_min_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_max_u(10, 8, 9);
    asm.atomic_min_u(11, 8, 9);
    asm.atomic_max_s(12, 8, 9);
    asm.atomic_min_s(13, 8, 9);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_MAX_U);
    assert_eq!(insts[1].opcode, bytecode_op::ATOMIC_MIN_U);
    assert_eq!(insts[2].opcode, bytecode_op::ATOMIC_MAX_S);
    assert_eq!(insts[3].opcode, bytecode_op::ATOMIC_MIN_S);
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUEUE PATTERN TESTS (Assembler-level)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_queue_slot_claim_pattern() {
    // This pattern is how GPU threads claim I/O queue slots
    // THE GPU NEVER WAITS - fire and forget

    let mut asm = BytecodeAssembler::new();

    // Load queue tail address into r8
    asm.loadi_uint(8, 0);  // Address 0 = queue tail

    // Atomic increment to claim a slot
    asm.atomic_inc(10, 8);  // r10 = our claimed slot index

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    // Verify the pattern
    assert_eq!(insts[0].opcode, bytecode_op::LOADI_UINT);
    assert_eq!(insts[1].opcode, bytecode_op::ATOMIC_INC);
}

#[test]
fn test_completion_poll_pattern() {
    // Non-blocking check if I/O is complete
    // THE GPU NEVER BLOCKS - poll and continue

    let mut asm = BytecodeAssembler::new();

    // Load completion flag address for slot 5
    let completion_base: u32 = 16;  // Base address for completion flags
    let slot: u32 = 5;
    asm.loadi_uint(8, completion_base + slot);

    // Atomic load the completion flag
    asm.atomic_load(10, 8);  // r10 = completion status

    // Check if complete (non-blocking)
    let skip_addr = 10; // placeholder
    asm.jz(10, skip_addr);  // If not complete, skip to other work

    // ... do something with completed data ...

    // Label: skip - continue with other work
    asm.nop();  // Placeholder

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::LOADI_UINT);
    assert_eq!(insts[1].opcode, bytecode_op::ATOMIC_LOAD);
    assert_eq!(insts[2].opcode, bytecode_op::JZ);
}

#[test]
fn test_cas_for_complex_state_transition() {
    // CAS pattern for lock-free state machine transitions
    // Expected: old_state -> new_state only if still old_state

    let mut asm = BytecodeAssembler::new();

    // Address of state variable
    asm.loadi_uint(8, 100);  // r8 = state address

    // Expected old value
    asm.loadi_uint(9, 0);  // r9 = expected (STATE_IDLE = 0)

    // Try to transition: if state == IDLE, set to RUNNING (1)
    asm.atomic_cas(10, 8, 9, 1);  // r10 = success (1) or failure (0)

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::LOADI_UINT);
    assert_eq!(insts[1].opcode, bytecode_op::LOADI_UINT);
    assert_eq!(insts[2].opcode, bytecode_op::ATOMIC_CAS);
    assert_eq!(imm_as_u32(insts[2].imm), 1);  // desired = 1
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC SUB TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_sub_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.atomic_sub(10, 8, 9);  // dst=r10, addr_reg=r8, val_reg=r9
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::ATOMIC_SUB);
    assert_eq!(insts[0].dst, 10);
    assert_eq!(insts[0].src1, 8);
    assert_eq!(insts[0].src2, 9);
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE INSTRUCTION COUNT TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_atomic_ops_emit_one_instruction() {
    // Each atomic operation should emit exactly one instruction

    let ops: Vec<(&str, Box<dyn Fn(&mut BytecodeAssembler)>)> = vec![
        ("atomic_load", Box::new(|a: &mut BytecodeAssembler| { a.atomic_load(10, 8); })),
        ("atomic_store", Box::new(|a: &mut BytecodeAssembler| { a.atomic_store(9, 8); })),
        ("atomic_add", Box::new(|a: &mut BytecodeAssembler| { a.atomic_add(10, 8, 9); })),
        ("atomic_sub", Box::new(|a: &mut BytecodeAssembler| { a.atomic_sub(10, 8, 9); })),
        ("atomic_max_u", Box::new(|a: &mut BytecodeAssembler| { a.atomic_max_u(10, 8, 9); })),
        ("atomic_min_u", Box::new(|a: &mut BytecodeAssembler| { a.atomic_min_u(10, 8, 9); })),
        ("atomic_max_s", Box::new(|a: &mut BytecodeAssembler| { a.atomic_max_s(10, 8, 9); })),
        ("atomic_min_s", Box::new(|a: &mut BytecodeAssembler| { a.atomic_min_s(10, 8, 9); })),
        ("atomic_and", Box::new(|a: &mut BytecodeAssembler| { a.atomic_and(10, 8, 9); })),
        ("atomic_or", Box::new(|a: &mut BytecodeAssembler| { a.atomic_or(10, 8, 9); })),
        ("atomic_xor", Box::new(|a: &mut BytecodeAssembler| { a.atomic_xor(10, 8, 9); })),
        ("atomic_cas", Box::new(|a: &mut BytecodeAssembler| { a.atomic_cas(10, 8, 9, 42); })),
        ("atomic_inc", Box::new(|a: &mut BytecodeAssembler| { a.atomic_inc(10, 8); })),
        ("atomic_dec", Box::new(|a: &mut BytecodeAssembler| { a.atomic_dec(10, 8); })),
        ("mem_fence", Box::new(|a: &mut BytecodeAssembler| { a.mem_fence(); })),
    ];

    for (name, emit_fn) in ops {
        let mut asm = BytecodeAssembler::new();
        emit_fn(&mut asm);
        assert_eq!(asm.pc(), 1, "{} should emit exactly 1 instruction", name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE RANGE VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_opcodes_contiguous() {
    // Verify atomic opcodes form a contiguous range (except for gaps)
    let opcodes = [
        bytecode_op::ATOMIC_LOAD,
        bytecode_op::ATOMIC_STORE,
        bytecode_op::ATOMIC_ADD,
        bytecode_op::ATOMIC_SUB,
        bytecode_op::ATOMIC_MAX_U,
        bytecode_op::ATOMIC_MIN_U,
        bytecode_op::ATOMIC_MAX_S,
        bytecode_op::ATOMIC_MIN_S,
        bytecode_op::ATOMIC_AND,
        bytecode_op::ATOMIC_OR,
        bytecode_op::ATOMIC_XOR,
        bytecode_op::ATOMIC_CAS,
        bytecode_op::ATOMIC_INC,
        bytecode_op::ATOMIC_DEC,
        bytecode_op::MEM_FENCE,
    ];

    // All should be in 0xE0-0xEE range
    for op in opcodes {
        assert!(op >= 0xE0 && op <= 0xEE, "Opcode 0x{:02X} outside atomic range", op);
    }
}

#[test]
fn test_atomic_opcode_count() {
    // We have 15 atomic operations total
    assert_eq!(bytecode_op::MEM_FENCE - bytecode_op::ATOMIC_LOAD + 1, 15);
}
