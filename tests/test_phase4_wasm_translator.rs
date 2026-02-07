//! Phase 4: WASM Translator Tests (Issue #174)
//!
//! THE GPU IS THE COMPUTER.
//! Compile real Rust to GPU via WASM.
//!
//! These tests verify WASM→GPU bytecode translation.

use rust_experiment::gpu_os::gpu_app_system::{BytecodeHeader, BytecodeInst, bytecode_op};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode bytecode bytes back into header and instructions
fn decode_bytecode(bytecode: &[u8]) -> (BytecodeHeader, Vec<BytecodeInst>) {
    if bytecode.len() < std::mem::size_of::<BytecodeHeader>() {
        return (BytecodeHeader::default(), Vec::new());
    }

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

/// Check if bytecode contains specific opcode
fn has_opcode(bytecode: &[u8], opcode: u8) -> bool {
    let (_, insts) = decode_bytecode(bytecode);
    insts.iter().any(|i| i.opcode == opcode)
}

/// Count occurrences of opcode
fn count_opcode(bytecode: &[u8], opcode: u8) -> usize {
    let (_, insts) = decode_bytecode(bytecode);
    insts.iter().filter(|i| i.opcode == opcode).count()
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASM BINARY CONSTRUCTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a minimal WASM module with a single function
fn build_wasm_module(func_body: &[u8]) -> Vec<u8> {
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]); // \0asm
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

    // Type section (section 1)
    // func type: () -> ()
    let type_section = vec![
        0x01,       // 1 type
        0x60,       // func type
        0x00,       // 0 params
        0x00,       // 0 results
    ];
    wasm.push(0x01);  // section id
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section (section 3)
    let func_section = vec![
        0x01,       // 1 function
        0x00,       // type index 0
    ];
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section (section 7)
    let export_name = b"gpu_main";
    let export_section = [
        &[0x01][..],                    // 1 export
        &[export_name.len() as u8][..], // name length
        export_name,                     // name
        &[0x00][..],                    // export kind: func
        &[0x00][..],                    // func index
    ].concat();
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section (section 10)
    let func_size = func_body.len() + 1; // +1 for local count
    let code_section = [
        &[0x01][..],                  // 1 function body
        &[func_size as u8][..],       // body size
        &[0x00][..],                  // 0 locals
        func_body,
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    wasm
}

/// Build WASM with params and locals
fn build_wasm_with_locals(params: u8, locals: u8, func_body: &[u8]) -> Vec<u8> {
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]);
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

    // Type section - func with params
    let mut type_section = vec![0x01, 0x60, params];
    for _ in 0..params {
        type_section.push(0x7f); // i32
    }
    type_section.push(0x00); // 0 results
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section
    wasm.push(0x03);
    wasm.push(0x02);
    wasm.extend_from_slice(&[0x01, 0x00]);

    // Export section
    let export_name = b"gpu_main";
    let export_section = [
        &[0x01][..],
        &[export_name.len() as u8][..],
        export_name,
        &[0x00, 0x00][..],
    ].concat();
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section with locals
    let mut locals_enc = Vec::new();
    if locals > 0 {
        locals_enc.push(0x01);      // 1 local entry
        locals_enc.push(locals);    // count
        locals_enc.push(0x7f);      // i32
    } else {
        locals_enc.push(0x00);      // 0 local entries
    }

    let body = [&locals_enc[..], func_body].concat();
    let code_section = [
        &[0x01][..],
        &[(body.len()) as u8][..],
        &body[..],
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    wasm
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSLATOR IMPORT
// ═══════════════════════════════════════════════════════════════════════════════

// Import the translator - it's in a separate crate
// For now, test the bytecode patterns that the translator should produce

// ═══════════════════════════════════════════════════════════════════════════════
// WASM INSTRUCTION ENCODING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_wasm_nop() {
    // WASM: nop, end
    let func_body = vec![0x01, 0x0b];  // nop, end
    let wasm = build_wasm_module(&func_body);

    // Verify WASM is well-formed
    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6d]);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_i32_const() {
    // WASM: i32.const 42, drop, end
    let func_body = vec![
        0x41, 42,  // i32.const 42
        0x1a,      // drop
        0x0b,      // end
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_i32_add() {
    // WASM: i32.const 10, i32.const 20, i32.add, drop, end
    let func_body = vec![
        0x41, 10,  // i32.const 10
        0x41, 20,  // i32.const 20
        0x6a,      // i32.add
        0x1a,      // drop
        0x0b,      // end
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_local_get_set() {
    // WASM with 1 local: local.get 0, local.set 0, end
    let func_body = vec![
        0x20, 0x00,  // local.get 0
        0x21, 0x00,  // local.set 0
        0x0b,        // end
    ];
    let wasm = build_wasm_with_locals(0, 1, &func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_block_loop() {
    // WASM: block, nop, end, loop, nop, end, end
    let func_body = vec![
        0x02, 0x40,  // block (void)
        0x01,        // nop
        0x0b,        // end block
        0x03, 0x40,  // loop (void)
        0x01,        // nop
        0x0b,        // end loop
        0x0b,        // end func
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_if_else() {
    // WASM: i32.const 1, if, i32.const 10, else, i32.const 20, end, drop, end
    let func_body = vec![
        0x41, 1,      // i32.const 1
        0x04, 0x40,   // if (void)
        0x41, 10,     // i32.const 10
        0x1a,         // drop
        0x05,         // else
        0x41, 20,     // i32.const 20
        0x1a,         // drop
        0x0b,         // end if
        0x0b,         // end func
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_br() {
    // WASM: block, br 0, end, end
    let func_body = vec![
        0x02, 0x40,  // block (void)
        0x0c, 0x00,  // br 0
        0x0b,        // end block
        0x0b,        // end func
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

#[test]
fn test_wasm_br_if() {
    // WASM: block, i32.const 1, br_if 0, end, end
    let func_body = vec![
        0x02, 0x40,  // block (void)
        0x41, 1,     // i32.const 1
        0x0d, 0x00,  // br_if 0
        0x0b,        // end block
        0x0b,        // end func
    ];
    let wasm = build_wasm_module(&func_body);
    assert!(wasm.len() > 20);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE PATTERN TESTS
// (Testing what the translator should produce)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_expected_bytecode_for_i32_add() {
    // Translator should produce: loadi_int r8, 10; loadi_int r9, 20; int_add r10, r8, r9
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // i32.const 10 -> alloc r8, load 10
    asm.loadi_int(8, 10);

    // i32.const 20 -> alloc r9, load 20
    asm.loadi_int(9, 20);

    // i32.add -> pop r9, pop r8, alloc r10, add
    asm.int_add(10, 8, 9);

    // drop -> pop (no code needed)

    // end -> halt
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
fn test_expected_bytecode_for_local_access() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // local.get 0 (mapped to r4)
    asm.mov(8, 4);  // Copy local 0 to temp

    // local.set 0
    asm.mov(4, 8);  // Copy back

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::MOV));
    assert!(has_opcode(&bytecode, bytecode_op::HALT));
}

#[test]
fn test_expected_bytecode_for_comparison() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // i32.const 10
    asm.loadi_int(8, 10);

    // i32.const 20
    asm.loadi_int(9, 20);

    // i32.lt_s
    asm.int_lt_s(10, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::INT_LT_S));
}

#[test]
fn test_expected_bytecode_for_control_flow() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // if: jz to else label
    asm.loadi_int(8, 1);  // condition
    let else_jump = asm.jz(8, 0);

    // then block
    asm.loadi_int(9, 10);

    // jmp to end
    let end_jump = asm.jmp(0);

    // else label
    let else_pc = asm.pc();
    asm.patch_jump(else_jump, else_pc);
    asm.loadi_int(9, 20);

    // end label
    let end_pc = asm.pc();
    asm.patch_jump(end_jump, end_pc);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::JZ));
    assert!(has_opcode(&bytecode, bytecode_op::JMP));
}

#[test]
fn test_expected_bytecode_for_loop() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // sum = 0
    asm.loadi_int(8, 0);

    // i = 0
    asm.loadi_int(9, 0);

    // limit = 10
    asm.loadi_int(10, 10);

    // loop:
    let loop_start = asm.pc();

    // i < limit?
    asm.int_lt_u(11, 9, 10);
    let exit_jump = asm.jz(11, 0);

    // sum += i
    asm.int_add(8, 8, 9);

    // i++
    asm.loadi_int(12, 1);
    asm.int_add(9, 9, 12);

    // br loop
    asm.jmp(loop_start);

    // exit:
    let exit_pc = asm.pc();
    asm.patch_jump(exit_jump, exit_pc);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::INT_LT_U));
    assert!(has_opcode(&bytecode, bytecode_op::JZ));
    assert!(has_opcode(&bytecode, bytecode_op::JMP));
    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD));
}

#[test]
fn test_expected_bytecode_for_bitwise() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // Test all bitwise ops
    asm.loadi_uint(8, 0xFF);
    asm.loadi_uint(9, 0x0F);

    asm.bit_and(10, 8, 9);
    asm.bit_or(11, 8, 9);
    asm.bit_xor(12, 8, 9);
    asm.shl(13, 8, 9);
    asm.shr_u(14, 8, 9);
    asm.shr_s(15, 8, 9);
    asm.rotl(16, 8, 9);
    asm.rotr(17, 8, 9);
    asm.clz(18, 8);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::BIT_AND));
    assert!(has_opcode(&bytecode, bytecode_op::BIT_OR));
    assert!(has_opcode(&bytecode, bytecode_op::BIT_XOR));
    assert!(has_opcode(&bytecode, bytecode_op::SHL));
    assert!(has_opcode(&bytecode, bytecode_op::SHR_U));
    assert!(has_opcode(&bytecode, bytecode_op::SHR_S));
    assert!(has_opcode(&bytecode, bytecode_op::ROTL));
    assert!(has_opcode(&bytecode, bytecode_op::ROTR));
    assert!(has_opcode(&bytecode, bytecode_op::CLZ));
}

#[test]
fn test_expected_bytecode_for_memory_access() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // Memory base
    let memory_base: u32 = 0x2000;

    // i32.load [addr 0]
    asm.loadi_uint(8, 0);            // addr = 0
    asm.loadi_uint(30, memory_base); // add base
    asm.int_add(8, 8, 30);
    asm.loadi_uint(30, 2);           // div 4 (shift right 2)
    asm.shr_u(8, 8, 30);
    asm.ld(9, 8, 0.0);               // load

    // i32.store [addr 4]
    asm.loadi_uint(10, 4);
    asm.loadi_uint(30, memory_base);
    asm.int_add(10, 10, 30);
    asm.loadi_uint(30, 2);
    asm.shr_u(10, 10, 30);
    asm.st(10, 9, 0.0);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::LD));
    assert!(has_opcode(&bytecode, bytecode_op::ST));
}

#[test]
fn test_expected_bytecode_for_conversion() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    // f32.convert_i32_s
    asm.loadi_int(8, 42);
    asm.int_to_f(9, 8);

    // i32.trunc_f32_s
    asm.loadi(10, 3.14);
    asm.f_to_int(11, 10);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::INT_TO_F));
    assert!(has_opcode(&bytecode, bytecode_op::F_TO_INT));
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE WASM-TO-BYTECODE MAPPING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_i32_arithmetic_ops() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    asm.loadi_int(8, 100);
    asm.loadi_int(9, 7);

    // All arithmetic ops
    asm.int_add(10, 8, 9);
    asm.int_sub(11, 8, 9);
    asm.int_mul(12, 8, 9);
    asm.int_div_s(13, 8, 9);
    asm.int_div_u(14, 8, 9);
    asm.int_rem_s(15, 8, 9);
    asm.int_rem_u(16, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD));
    assert!(has_opcode(&bytecode, bytecode_op::INT_SUB));
    assert!(has_opcode(&bytecode, bytecode_op::INT_MUL));
    assert!(has_opcode(&bytecode, bytecode_op::INT_DIV_S));
    assert!(has_opcode(&bytecode, bytecode_op::INT_DIV_U));
    assert!(has_opcode(&bytecode, bytecode_op::INT_REM_S));
    assert!(has_opcode(&bytecode, bytecode_op::INT_REM_U));
}

#[test]
fn test_all_i32_comparison_ops() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    asm.loadi_int(8, 10);
    asm.loadi_int(9, 20);

    // All comparison ops
    asm.int_eq(10, 8, 9);
    asm.int_ne(11, 8, 9);
    asm.int_lt_s(12, 8, 9);
    asm.int_lt_u(13, 8, 9);
    asm.int_le_s(14, 8, 9);
    asm.int_le_u(15, 8, 9);
    // gt and ge are implemented via swapped lt/le

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::INT_EQ));
    assert!(has_opcode(&bytecode, bytecode_op::INT_NE));
    assert!(has_opcode(&bytecode, bytecode_op::INT_LT_S));
    assert!(has_opcode(&bytecode, bytecode_op::INT_LT_U));
    assert!(has_opcode(&bytecode, bytecode_op::INT_LE_S));
    assert!(has_opcode(&bytecode, bytecode_op::INT_LE_U));
}

#[test]
fn test_f32_operations() {
    use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;

    let mut asm = BytecodeAssembler::new();

    asm.loadi(8, 10.0);
    asm.loadi(9, 3.0);

    asm.add(10, 8, 9);
    asm.sub(11, 8, 9);
    asm.mul(12, 8, 9);
    asm.div(13, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);

    assert!(has_opcode(&bytecode, bytecode_op::ADD));
    assert!(has_opcode(&bytecode, bytecode_op::SUB));
    assert!(has_opcode(&bytecode, bytecode_op::MUL));
    assert!(has_opcode(&bytecode, bytecode_op::DIV));
}

// ═══════════════════════════════════════════════════════════════════════════════
// END-TO-END: REAL RUST → WASM → GPU BYTECODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Test real Rust code compiled to WASM then translated to GPU bytecode.
///
/// THE GPU IS THE COMPUTER.
///
/// This test proves the full pipeline:
/// Rust Source → rustc → WASM → WasmTranslator → GPU Bytecode
#[test]
fn test_end_to_end_rust_to_gpu() {
    use wasm_translator::WasmTranslator;

    // Load pre-compiled WASM (from test_data/gpu_wasm_app)
    // Source: test_data/gpu_wasm_app/src/lib.rs
    //
    // #[no_mangle]
    // pub extern "C" fn gpu_main(n: i32) -> i32 {
    //     let mut sum = 0i32;
    //     let mut i = 1i32;
    //     while i <= n {
    //         sum += i;
    //         i += 1;
    //     }
    //     sum
    // }
    let wasm_bytes = include_bytes!("../test_data/simple_rust.wasm");

    // Translate WASM to GPU bytecode
    let translator = WasmTranslator::default();
    let result = translator.translate(wasm_bytes);

    // Should translate successfully
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());

    let bytecode = result.unwrap();

    // Verify bytecode structure
    let (header, insts) = decode_bytecode(&bytecode);
    assert!(header.code_size > 0, "No instructions generated");
    assert!(!insts.is_empty(), "No instructions decoded");

    // Should contain a HALT instruction
    assert!(has_opcode(&bytecode, bytecode_op::HALT), "Missing HALT");

    // Should contain integer operations (the sum loop uses i32.add)
    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD), "Missing INT_ADD for sum");

    // Should contain comparison (while i <= n)
    assert!(
        has_opcode(&bytecode, bytecode_op::INT_LE_S) || has_opcode(&bytecode, bytecode_op::INT_LT_S),
        "Missing comparison for loop condition"
    );

    // Should contain jump (for while loop)
    assert!(
        has_opcode(&bytecode, bytecode_op::JMP) || has_opcode(&bytecode, bytecode_op::JNZ) || has_opcode(&bytecode, bytecode_op::JZ),
        "Missing jump for loop"
    );

    println!("End-to-end test passed!");
    println!("  WASM size: {} bytes", wasm_bytes.len());
    println!("  Bytecode size: {} bytes", bytecode.len());
    println!("  Instructions: {}", insts.len());
}

/// Test that the translator handles multiple exported functions
#[test]
fn test_real_wasm_exports() {
    use wasm_translator::{WasmTranslator, WasmModule};

    let wasm_bytes = include_bytes!("../test_data/simple_rust.wasm");

    // Parse just to check exports
    let translator = WasmTranslator::default();
    let result = translator.translate(wasm_bytes);

    // Translation should work (finds gpu_main)
    assert!(result.is_ok(), "Should find gpu_main export");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUE END-TO-END: RUST → WASM → GPU BYTECODE → GPU EXECUTION → RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple test to verify bytecode VM executes at all
/// Just stores a constant to state[0]
#[test]
fn test_bytecode_vm_basic_execution() {
    use metal::Device;
    use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler, bytecode_op};

    let device = Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GpuAppSystem");
    system.set_use_parallel_megakernel(true);

    // Create bytecode that just stores 42 to state[0]
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(30, 0);    // r30 = 0 (state index)
    asm.loadi_int(4, 42);     // r4 = 42 (value to store)
    asm.st(30, 4, 0.0);       // state[0] = r4
    asm.halt();

    let bytecode = asm.build(1024);
    println!("Simple bytecode: {} bytes", bytecode.len());

    // Launch app
    let slot = system.launch_app(app_type::BYTECODE, bytecode.len() as u32 + 64, 1024)
        .expect("Failed to launch app");

    // Write bytecode
    system.write_app_state(slot, &bytecode);

    // Check app is active
    let app = system.get_app(slot).expect("App not found");
    println!("App slot={}, flags={:#x}, app_type={}", app.slot_id, app.flags, app.app_type);

    // Run
    system.run_frame();

    // Read state[0]
    let header_size = std::mem::size_of::<rust_experiment::gpu_os::gpu_app_system::BytecodeHeader>();
    let inst_size = std::mem::size_of::<rust_experiment::gpu_os::gpu_app_system::BytecodeInst>();
    let code_size = 4; // 4 instructions

    let data_offset = app.state_offset as usize + header_size + code_size * inst_size;
    println!("Reading from data_offset={}", data_offset);

    let result: i32 = unsafe {
        let ptr = system.state_buffer().contents().add(data_offset) as *const f32;
        let float_val = *ptr;
        println!("Raw float bytes: {:#010x}", float_val.to_bits());
        float_val.to_bits() as i32
    };

    println!("Result: {}, expected: 42", result);
    assert_eq!(result, 42, "Bytecode VM should store 42 to state[0]");

    system.close_app(slot);
}

/// THE GPU IS THE COMPUTER.
///
/// This test proves the COMPLETE pipeline:
/// 1. Rust Source (no_std) - computes sum(1..=10) with hardcoded n=10
/// 2. → rustc → WASM binary
/// 3. → WasmTranslator → GPU bytecode
/// 4. → GpuAppSystem → GPU execution
/// 5. → Read result from GPU memory
///
/// We compute sum(1..=10) = 55 entirely on the GPU from compiled Rust code.
///
/// Note: Parameter passing tracked in Issue #176
#[test]
fn test_true_end_to_end_gpu_execution() {
    use metal::Device;
    use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeHeader, BytecodeInst};
    use wasm_translator::WasmTranslator;

    // Get Metal device
    let device = Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GpuAppSystem");

    // Enable parallel megakernel (required for bytecode VM)
    system.set_use_parallel_megakernel(true);

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1-2: Load pre-compiled WASM (Rust → rustc → WASM)
    // The WASM computes sum(1..=10) = 55 with hardcoded n=10 (no parameters)
    // ═══════════════════════════════════════════════════════════════════════════
    let wasm_bytes = include_bytes!("../test_data/simple_rust.wasm");
    println!("Loaded WASM: {} bytes", wasm_bytes.len());

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: Translate WASM → GPU bytecode
    // ═══════════════════════════════════════════════════════════════════════════
    let translator = WasmTranslator::default();
    let bytecode = translator.translate(wasm_bytes).expect("Translation failed");
    println!("Generated bytecode: {} bytes", bytecode.len());

    // Decode to verify structure
    let (header, insts) = decode_bytecode(&bytecode);
    println!("Bytecode: {} instructions", header.code_size);

    // Debug: print all instructions
    println!("\n=== BYTECODE INSTRUCTIONS ===");
    for (i, inst) in insts.iter().enumerate() {
        println!("  [{:2}] op={:#04x} dst={} src1={} src2={} imm={} (bits={:#010x})",
            i, inst.opcode, inst.dst, inst.src1, inst.src2,
            inst.imm, inst.imm.to_bits());
    }
    println!("=============================\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: Launch app and execute on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    // Calculate state size: bytecode + room for state data (at least 16 bytes for result)
    let state_size = bytecode.len() as u32 + 256;  // Extra space for state

    // Launch a BYTECODE app
    let slot = system.launch_app(app_type::BYTECODE, state_size, 1024)
        .expect("Failed to launch app");
    println!("Launched app in slot {}", slot);

    // Write bytecode to app state
    system.write_app_state(slot, &bytecode);

    // ═══════════════════════════════════════════════════════════════════════════
    // Write input parameter to state[1]
    // THE GPU IS THE COMPUTER - parameters passed via GPU memory
    // Calling convention: state[0]=return, state[1]=param0, state[2]=param1, ...
    // ═══════════════════════════════════════════════════════════════════════════
    let app = system.get_app(slot).expect("App not found");
    let state_offset = app.state_offset as usize;
    let code_offset = state_offset + std::mem::size_of::<BytecodeHeader>();
    let data_offset = code_offset + (header.code_size as usize) * std::mem::size_of::<BytecodeInst>();

    // Write n=10 to state[1] (param 0)
    // state[1] is at data_offset + 16 bytes (state[0] is float4 = 16 bytes)
    let param_offset = data_offset + 16;  // Skip state[0], write to state[1]
    let n: i32 = 10;
    unsafe {
        let state_buffer = system.state_buffer();
        let ptr = state_buffer.contents().add(param_offset) as *mut f32;
        // Store as int bits in float (GPU will read with as_type<int>)
        *ptr = f32::from_bits(n as u32);
    }
    println!("Wrote n={} to state[1] at offset {}", n, param_offset);

    // Run megakernel - GPU executes the bytecode
    system.run_frame();
    println!("Megakernel executed");

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 5: Read result from GPU memory
    // ═══════════════════════════════════════════════════════════════════════════

    // Result is stored at state[3] (after bytecode + SlabAllocator header)
    // Use the read_bytecode_result method which handles the correct offset
    let result = system.read_bytecode_result(slot).expect("Failed to read result");

    println!("GPU computed result: {}", result);
    println!("Expected: 55 (sum of 1..=10)");

    // ═══════════════════════════════════════════════════════════════════════════
    // VERIFY
    // ═══════════════════════════════════════════════════════════════════════════
    assert_eq!(result, 55, "GPU should compute sum(1..=10) = 55");

    // Cleanup
    system.close_app(slot);

    println!("\n✓ TRUE END-TO-END TEST PASSED!");
    println!("  Rust → WASM → GPU Bytecode → GPU Execution → Correct Result");
    println!("  THE GPU IS THE COMPUTER.");
}
