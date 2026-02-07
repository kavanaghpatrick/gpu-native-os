//! Debug minimal test for Snake-like return value issue

use metal::Device;
use wasm_translator::{WasmTranslator, TranslatorConfig};
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeHeader, BytecodeInst, bytecode_op};
use std::fs;

fn decode_and_show(bytecode: &[u8], limit: usize) {
    let header: BytecodeHeader = unsafe {
        std::ptr::read(bytecode.as_ptr() as *const BytecodeHeader)
    };
    println!("Header: code_size={}, entry_point={}", header.code_size, header.entry_point);

    let inst_start = std::mem::size_of::<BytecodeHeader>();
    let show_count = (header.code_size as usize).min(limit);
    for i in 0..show_count {
        let offset = inst_start + i * std::mem::size_of::<BytecodeInst>();
        let inst: BytecodeInst = unsafe {
            std::ptr::read(bytecode[offset..].as_ptr() as *const BytecodeInst)
        };
        let op_name = match inst.opcode {
            bytecode_op::NOP => "NOP",
            bytecode_op::HALT => "HALT",
            bytecode_op::MOV => "MOV",
            bytecode_op::LOADI => "LOADI",
            bytecode_op::LOADI_INT => "LOADI_INT",
            bytecode_op::LOADI_UINT => "LOADI_UINT",
            bytecode_op::ST => "ST",
            bytecode_op::LD => "LD",
            bytecode_op::JMP => "JMP",
            bytecode_op::JZ => "JZ",
            bytecode_op::JNZ => "JNZ",
            bytecode_op::INT_ADD => "INT_ADD",
            bytecode_op::INT_SUB => "INT_SUB",
            bytecode_op::INT_MUL => "INT_MUL",
            bytecode_op::INT_LT_S => "INT_LT_S",
            bytecode_op::INT_EQ => "INT_EQ",
            bytecode_op::QUAD => "QUAD",
            _ => "OTHER",
        };
        println!("  [{:03}] op={:02x} {:10} d={:2} s1={:2} s2={:2} imm={:.1} (0x{:08X})",
                 i, inst.opcode, op_name, inst.dst, inst.src1, inst.src2, inst.imm, inst.imm.to_bits());
    }
    if header.code_size as usize > limit {
        println!("  ... ({} more instructions)", header.code_size as usize - limit);
    }
    // Show last 10 instructions
    if header.code_size as usize > limit + 10 {
        println!("  === Last 10 instructions ===");
        for i in (header.code_size as usize - 10)..header.code_size as usize {
            let offset = inst_start + i * std::mem::size_of::<BytecodeInst>();
            let inst: BytecodeInst = unsafe {
                std::ptr::read(bytecode[offset..].as_ptr() as *const BytecodeInst)
            };
            let op_name = match inst.opcode {
                bytecode_op::NOP => "NOP",
                bytecode_op::HALT => "HALT",
                bytecode_op::MOV => "MOV",
                bytecode_op::LOADI => "LOADI",
                bytecode_op::LOADI_INT => "LOADI_INT",
                bytecode_op::LOADI_UINT => "LOADI_UINT",
                bytecode_op::ST => "ST",
                bytecode_op::LD => "LD",
                bytecode_op::JMP => "JMP",
                bytecode_op::JZ => "JZ",
                bytecode_op::JNZ => "JNZ",
                bytecode_op::INT_ADD => "INT_ADD",
                bytecode_op::INT_MUL => "INT_MUL",
                bytecode_op::INT_LT_S => "INT_LT_S",
                bytecode_op::QUAD => "QUAD",
                _ => "OTHER",
            };
            println!("  [{:03}] op={:02x} {:10} d={:2} s1={:2} s2={:2} imm={:.1} (0x{:08X})",
                     i, inst.opcode, op_name, inst.dst, inst.src1, inst.src2, inst.imm, inst.imm.to_bits());
        }
    }
}

#[test]
fn test_minimal_return_400() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/minimal_test/target/wasm32-unknown-unknown/release/minimal_test.wasm")
        .expect("Read minimal test WASM");

    println!("\n=== MINIMAL TEST DEBUG ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 20);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 400");

    assert_eq!(result, 400, "Minimal test should return 400");
}

#[test]
fn test_nested_loop() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/nested_loop_test/target/wasm32-unknown-unknown/release/nested_loop_test.wasm")
        .expect("Read nested loop test WASM");

    println!("\n=== NESTED LOOP TEST ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 100);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 100 (10*10 nested loop)");

    assert_eq!(result, 20, "Nested loop should return 20 (5*4)");
}

#[test]
fn test_snake_debug() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("test_programs/apps/snake/target/wasm32-unknown-unknown/release/snake.wasm")
        .expect("Read Snake WASM");

    println!("\n=== SNAKE DEBUG ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 50);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 400 (20*20 cells)");
}

#[test]
fn test_mandelbrot_debug() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("test_programs/apps/mandelbrot_interactive/target/wasm32-unknown-unknown/release/mandelbrot_interactive.wasm")
        .expect("Read Mandelbrot WASM");

    println!("\n=== MANDELBROT DEBUG ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 100);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 4096 (64*64 pixels)");
}

#[test]
fn test_f32_simple() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/f32_test/target/wasm32-unknown-unknown/release/f32_test.wasm")
        .expect("Read f32 test WASM");

    println!("\n=== F32 SIMPLE TEST ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 50);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 26 (20 quads + 6.0 as int)");

    assert_eq!(result, 26, "f32 test should return 26");
}

#[test]
fn test_loop_counter() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/loop_counter_test/target/wasm32-unknown-unknown/release/loop_counter_test.wasm")
        .expect("Read loop counter test WASM");

    println!("\n=== LOOP COUNTER TEST ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 100);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 20 (4 * 5)");

    assert_eq!(result, 20, "Loop counter test should return 20");
}

#[test]
fn test_unoptimized_loops() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/state_test/target/wasm32-unknown-unknown/release/state_test.wasm")
        .expect("Read state test WASM");

    println!("\n=== UNOPTIMIZED LOOPS TEST ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 100);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 20 (4 * 5)");

    assert_eq!(result, 20, "Unoptimized loop test should return 20");
}

#[test]
fn test_inline_function() {
    let device = Device::system_default().expect("No Metal device found");
    let wasm_bytes = fs::read("/tmp/inline_test/target/wasm32-unknown-unknown/release/inline_test.wasm")
        .expect("Read inline test WASM");

    println!("\n=== INLINE FUNCTION TEST ===");
    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("Bytecode size: {} bytes", bytecode.len());
    decode_and_show(&bytecode, 100);

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("\nGPU Result: {}", result);
    println!("Expected: 30");
}
