//! Phase 5: Function Call Support Tests (Issue #178)
//!
//! THE GPU IS THE COMPUTER.
//! Test function inlining and GPU intrinsics.
//!
//! These tests verify:
//! - Helper function inlining
//! - Recursion detection
//! - GPU intrinsics (thread_id, sin, cos, sqrt)

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

/// LEB128 encode a u32
fn leb128_u32(val: u32) -> Vec<u8> {
    let mut result = Vec::new();
    let mut n = val;
    loop {
        let byte = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            result.push(byte);
            break;
        } else {
            result.push(byte | 0x80);
        }
    }
    result
}

/// Build a minimal WASM module with a single function
fn build_wasm_module(func_body: &[u8]) -> Vec<u8> {
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]); // \0asm
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

    // Type section (section 1)
    // func type: () -> i32
    let type_section = vec![
        0x01,       // 1 type
        0x60,       // func type
        0x00,       // 0 params
        0x01, 0x7f, // 1 result: i32
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
    let export_name = b"main";
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

/// Build WASM module with multiple functions (for inlining tests)
fn build_wasm_with_helper(main_body: &[u8], helper_body: &[u8]) -> Vec<u8> {
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]); // \0asm
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

    // Type section - two types:
    // type 0: () -> i32 (main)
    // type 1: (i32) -> i32 (helper)
    let type_section = vec![
        0x02,       // 2 types
        0x60, 0x00, 0x01, 0x7f,  // () -> i32
        0x60, 0x01, 0x7f, 0x01, 0x7f,  // (i32) -> i32
    ];
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section - 2 functions
    let func_section = vec![
        0x02,       // 2 functions
        0x00,       // main: type 0
        0x01,       // helper: type 1
    ];
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section
    let export_name = b"main";
    let export_section = [
        &[0x01][..],
        &[export_name.len() as u8][..],
        export_name,
        &[0x00, 0x00][..],  // func index 0
    ].concat();
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section - 2 function bodies
    let main_size = main_body.len() + 1;
    let helper_size = helper_body.len() + 1;
    let code_section = [
        &[0x02][..],  // 2 function bodies
        // main
        &[main_size as u8][..],
        &[0x00][..],  // 0 locals
        main_body,
        // helper
        &[helper_size as u8][..],
        &[0x00][..],  // 0 locals
        helper_body,
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    wasm
}

/// Build WASM module with imports (for GPU intrinsics tests)
fn build_wasm_with_import(
    import_module: &str,
    import_name: &str,
    import_type: &[u8],  // type signature bytes
    func_type: &[u8],    // func type for main
    func_body: &[u8],
) -> Vec<u8> {
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]);
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

    // Type section
    let mut type_section = vec![0x02];  // 2 types
    type_section.extend_from_slice(import_type);  // import type
    type_section.extend_from_slice(func_type);    // main type
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Import section
    let import_section = [
        &[0x01][..],  // 1 import
        &[import_module.len() as u8][..],
        import_module.as_bytes(),
        &[import_name.len() as u8][..],
        import_name.as_bytes(),
        &[0x00, 0x00][..],  // func, type 0
    ].concat();
    wasm.push(0x02);  // import section
    wasm.push(import_section.len() as u8);
    wasm.extend_from_slice(&import_section);

    // Function section - 1 function (main)
    let func_section = vec![0x01, 0x01];  // 1 func, type 1
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section
    let export_name = b"main";
    let export_section = [
        &[0x01][..],
        &[export_name.len() as u8][..],
        export_name,
        &[0x00, 0x01][..],  // func index 1 (after import)
    ].concat();
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section
    let func_size = func_body.len() + 1;
    let code_section = [
        &[0x01][..],
        &[func_size as u8][..],
        &[0x00][..],
        func_body,
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    wasm
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: SIMPLE HELPER FUNCTION INLINING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_simple_helper_inline() {
    use wasm_translator::WasmTranslator;

    // Helper function: add_one(x: i32) -> i32 = x + 1
    let helper_body = vec![
        0x20, 0x00,  // local.get 0
        0x41, 0x01,  // i32.const 1
        0x6a,        // i32.add
        0x0b,        // end
    ];

    // Main function: call helper(42)
    let main_body = vec![
        0x41, 42,    // i32.const 42
        0x10, 0x01,  // call 1 (helper function)
        0x0b,        // end
    ];

    let wasm = build_wasm_with_helper(&main_body, &helper_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "Translation should succeed: {:?}", result.err());

    let bytecode = result.unwrap();
    let (header, insts) = decode_bytecode(&bytecode);

    // Should have generated code (inlined helper)
    assert!(header.code_size > 0, "Should generate instructions");

    // Should contain INT_ADD from the inlined helper
    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD),
        "Should inline helper function with INT_ADD");

    // Should end with HALT
    assert!(has_opcode(&bytecode, bytecode_op::HALT),
        "Should end with HALT");

    println!("Helper inline test passed!");
    println!("  Generated {} instructions", insts.len());
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: MULTIPLE HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multiple_helpers() {
    use wasm_translator::WasmTranslator;

    // This test constructs a WASM module with 3 functions:
    // - func 0 (main): calls func 1 and func 2, returns sum
    // - func 1 (double): x * 2
    // - func 2 (triple): x * 3

    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]);
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

    // Type section: 2 types
    // type 0: () -> i32
    // type 1: (i32) -> i32
    let type_section = vec![
        0x02,       // 2 types
        0x60, 0x00, 0x01, 0x7f,  // () -> i32
        0x60, 0x01, 0x7f, 0x01, 0x7f,  // (i32) -> i32
    ];
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section: 3 functions
    let func_section = vec![
        0x03,       // 3 functions
        0x00,       // main: type 0
        0x01,       // double: type 1
        0x01,       // triple: type 1
    ];
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section
    let export_section = vec![
        0x01,       // 1 export
        0x04, b'm', b'a', b'i', b'n',  // "main"
        0x00, 0x00, // func index 0
    ];
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section: 3 function bodies
    // main: call double(5), call triple(5), add results
    let main_body = vec![
        0x41, 5,     // i32.const 5
        0x10, 0x01,  // call double (func 1)
        0x41, 5,     // i32.const 5
        0x10, 0x02,  // call triple (func 2)
        0x6a,        // i32.add
        0x0b,        // end
    ];

    // double: x * 2
    let double_body = vec![
        0x20, 0x00,  // local.get 0
        0x41, 2,     // i32.const 2
        0x6c,        // i32.mul
        0x0b,        // end
    ];

    // triple: x * 3
    let triple_body = vec![
        0x20, 0x00,  // local.get 0
        0x41, 3,     // i32.const 3
        0x6c,        // i32.mul
        0x0b,        // end
    ];

    let main_size = main_body.len() + 1;
    let double_size = double_body.len() + 1;
    let triple_size = triple_body.len() + 1;

    let code_section = [
        &[0x03][..],  // 3 function bodies
        &[main_size as u8][..], &[0x00][..], &main_body[..],
        &[double_size as u8][..], &[0x00][..], &double_body[..],
        &[triple_size as u8][..], &[0x00][..], &triple_body[..],
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "Translation should succeed: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should have multiple INT_MUL (from double and triple)
    let mul_count = count_opcode(&bytecode, bytecode_op::INT_MUL);
    assert!(mul_count >= 2, "Should inline both helper functions (got {} MUL)", mul_count);

    // Should have INT_ADD (from summing results)
    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD),
        "Should have INT_ADD from summing results");

    println!("Multiple helpers test passed!");
    println!("  Found {} INT_MUL instructions", mul_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: RECURSION DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recursion_detection() {
    use wasm_translator::WasmTranslator;

    // Create a recursive function: calls itself
    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]);
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

    // Type section
    let type_section = vec![
        0x01,       // 1 type
        0x60, 0x01, 0x7f, 0x01, 0x7f,  // (i32) -> i32
    ];
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section
    let func_section = vec![0x01, 0x00];  // 1 func, type 0
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section
    let export_section = vec![
        0x01,       // 1 export
        0x04, b'm', b'a', b'i', b'n',
        0x00, 0x00, // func index 0
    ];
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section: recursive function
    // factorial(n) = n == 0 ? 1 : n * factorial(n-1)
    // But simplified: just call self unconditionally (will detect recursion)
    let func_body = vec![
        0x20, 0x00,  // local.get 0
        0x10, 0x00,  // call 0 (self - RECURSION!)
        0x0b,        // end
    ];

    let func_size = func_body.len() + 1;
    let code_section = [
        &[0x01][..],
        &[func_size as u8][..],
        &[0x00][..],
        &func_body[..],
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    // Should fail with recursion error
    assert!(result.is_err(), "Recursion should be detected and rejected");

    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    assert!(err_msg.contains("recursion") || err_msg.contains("Recursion"),
        "Error should mention recursion: {}", err_msg);

    println!("Recursion detection test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - thread_id()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_thread_id() {
    use wasm_translator::WasmTranslator;

    // Import thread_id from env, call it
    // thread_id: () -> i32
    let import_type = &[0x60, 0x00, 0x01, 0x7f];  // () -> i32
    let func_type = &[0x60, 0x00, 0x01, 0x7f];    // () -> i32

    let func_body = vec![
        0x10, 0x00,  // call 0 (thread_id import)
        0x0b,        // end
    ];

    let wasm = build_wasm_with_import("env", "thread_id", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "Thread ID intrinsic should work: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should contain MOV to get thread ID from r1
    assert!(has_opcode(&bytecode, bytecode_op::MOV),
        "Should use MOV to get thread ID");

    println!("thread_id() intrinsic test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - sin()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_sin() {
    use wasm_translator::WasmTranslator;

    // Import sin from env
    // sin: (f32) -> f32
    let import_type = &[0x60, 0x01, 0x7d, 0x01, 0x7d];  // (f32) -> f32
    let func_type = &[0x60, 0x00, 0x01, 0x7d];          // () -> f32

    let func_body = vec![
        0x43, 0x00, 0x00, 0x00, 0x40,  // f32.const 2.0
        0x10, 0x00,                     // call sin (import 0)
        0x0b,                           // end
    ];

    let wasm = build_wasm_with_import("env", "sin", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "sin() intrinsic should work: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should contain SIN opcode
    assert!(has_opcode(&bytecode, bytecode_op::SIN),
        "Should emit SIN opcode for sin() intrinsic");

    println!("sin() intrinsic test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - cos()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_cos() {
    use wasm_translator::WasmTranslator;

    // Import cos from env
    let import_type = &[0x60, 0x01, 0x7d, 0x01, 0x7d];  // (f32) -> f32
    let func_type = &[0x60, 0x00, 0x01, 0x7d];          // () -> f32

    let func_body = vec![
        0x43, 0x00, 0x00, 0x00, 0x40,  // f32.const 2.0
        0x10, 0x00,                     // call cos (import 0)
        0x0b,                           // end
    ];

    let wasm = build_wasm_with_import("env", "cos", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "cos() intrinsic should work: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should contain COS opcode
    assert!(has_opcode(&bytecode, bytecode_op::COS),
        "Should emit COS opcode for cos() intrinsic");

    println!("cos() intrinsic test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - sqrt()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_sqrt() {
    use wasm_translator::WasmTranslator;

    // Import sqrt from env
    let import_type = &[0x60, 0x01, 0x7d, 0x01, 0x7d];  // (f32) -> f32
    let func_type = &[0x60, 0x00, 0x01, 0x7d];          // () -> f32

    let func_body = vec![
        0x43, 0x00, 0x00, 0x80, 0x40,  // f32.const 4.0
        0x10, 0x00,                     // call sqrt (import 0)
        0x0b,                           // end
    ];

    let wasm = build_wasm_with_import("env", "sqrt", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "sqrt() intrinsic should work: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should contain SQRT opcode
    assert!(has_opcode(&bytecode, bytecode_op::SQRT),
        "Should emit SQRT opcode for sqrt() intrinsic");

    println!("sqrt() intrinsic test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: NESTED FUNCTION CALLS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_nested_calls() {
    use wasm_translator::WasmTranslator;

    // Test: main calls foo, foo calls bar
    // foo(x) = bar(x) + 1
    // bar(x) = x * 2

    let mut wasm = Vec::new();

    // Magic + version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]);
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

    // Type section
    let type_section = vec![
        0x02,       // 2 types
        0x60, 0x00, 0x01, 0x7f,          // type 0: () -> i32
        0x60, 0x01, 0x7f, 0x01, 0x7f,    // type 1: (i32) -> i32
    ];
    wasm.push(0x01);
    wasm.push(type_section.len() as u8);
    wasm.extend_from_slice(&type_section);

    // Function section: 3 functions
    let func_section = vec![
        0x03,       // 3 functions
        0x00,       // main: type 0
        0x01,       // foo: type 1
        0x01,       // bar: type 1
    ];
    wasm.push(0x03);
    wasm.push(func_section.len() as u8);
    wasm.extend_from_slice(&func_section);

    // Export section
    let export_section = vec![
        0x01,
        0x04, b'm', b'a', b'i', b'n',
        0x00, 0x00,
    ];
    wasm.push(0x07);
    wasm.push(export_section.len() as u8);
    wasm.extend_from_slice(&export_section);

    // Code section
    // main: foo(10)
    let main_body = vec![
        0x41, 10,    // i32.const 10
        0x10, 0x01,  // call foo (func 1)
        0x0b,        // end
    ];

    // foo: bar(x) + 1
    let foo_body = vec![
        0x20, 0x00,  // local.get 0
        0x10, 0x02,  // call bar (func 2)
        0x41, 1,     // i32.const 1
        0x6a,        // i32.add
        0x0b,        // end
    ];

    // bar: x * 2
    let bar_body = vec![
        0x20, 0x00,  // local.get 0
        0x41, 2,     // i32.const 2
        0x6c,        // i32.mul
        0x0b,        // end
    ];

    let main_size = main_body.len() + 1;
    let foo_size = foo_body.len() + 1;
    let bar_size = bar_body.len() + 1;

    let code_section = [
        &[0x03][..],
        &[main_size as u8][..], &[0x00][..], &main_body[..],
        &[foo_size as u8][..], &[0x00][..], &foo_body[..],
        &[bar_size as u8][..], &[0x00][..], &bar_body[..],
    ].concat();
    wasm.push(0x0a);
    wasm.push(code_section.len() as u8);
    wasm.extend_from_slice(&code_section);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "Nested calls should work: {:?}", result.err());

    let bytecode = result.unwrap();

    // Should have INT_MUL (from bar)
    assert!(has_opcode(&bytecode, bytecode_op::INT_MUL),
        "Should inline bar with INT_MUL");

    // Should have INT_ADD (from foo)
    assert!(has_opcode(&bytecode, bytecode_op::INT_ADD),
        "Should inline foo with INT_ADD");

    println!("Nested calls test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: UNSUPPORTED IMPORT HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_unsupported_import() {
    use wasm_translator::WasmTranslator;

    // Import an unknown function
    let import_type = &[0x60, 0x00, 0x01, 0x7f];  // () -> i32
    let func_type = &[0x60, 0x00, 0x01, 0x7f];   // () -> i32

    let func_body = vec![
        0x10, 0x00,  // call unknown_func (import 0)
        0x0b,        // end
    ];

    let wasm = build_wasm_with_import("env", "unknown_func", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    // Should fail with unsupported import
    assert!(result.is_err(), "Unknown import should fail");

    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    assert!(err_msg.contains("unsupported") || err_msg.contains("Unsupported"),
        "Error should mention unsupported: {}", err_msg);

    println!("Unsupported import test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - threadgroup_size()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_threadgroup_size() {
    use wasm_translator::WasmTranslator;

    let import_type = &[0x60, 0x00, 0x01, 0x7f];  // () -> i32
    let func_type = &[0x60, 0x00, 0x01, 0x7f];    // () -> i32

    let func_body = vec![
        0x10, 0x00,  // call threadgroup_size (import 0)
        0x0b,        // end
    ];

    let wasm = build_wasm_with_import("env", "threadgroup_size", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "threadgroup_size() should work: {:?}", result.err());

    let bytecode = result.unwrap();
    assert!(has_opcode(&bytecode, bytecode_op::MOV),
        "Should use MOV to get threadgroup_size");

    println!("threadgroup_size() intrinsic test passed!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: GPU INTRINSIC - frame()
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_intrinsic_frame() {
    use wasm_translator::WasmTranslator;

    let import_type = &[0x60, 0x00, 0x01, 0x7f];  // () -> i32
    let func_type = &[0x60, 0x00, 0x01, 0x7f];    // () -> i32

    let func_body = vec![
        0x10, 0x00,  // call frame (import 0)
        0x0b,        // end
    ];

    let wasm = build_wasm_with_import("env", "frame", import_type, func_type, &func_body);

    let translator = WasmTranslator::default();
    let result = translator.translate(&wasm);

    assert!(result.is_ok(), "frame() should work: {:?}", result.err());

    let bytecode = result.unwrap();
    assert!(has_opcode(&bytecode, bytecode_op::MOV),
        "Should use MOV to get frame number");

    println!("frame() intrinsic test passed!");
}
