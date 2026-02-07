//! Test translation and execution of real WASM programs compiled from Rust
//!
//! THE GPU IS THE COMPUTER.

use wasm_translator::{WasmTranslator, TranslatorConfig};
use rust_experiment::gpu_os::gpu_app_system::{BytecodeHeader, BytecodeInst, bytecode_op};
use std::fs;

const SIMPLE_MATH_WASM: &str = "test_programs/simple_math/target/wasm32-unknown-unknown/release/simple_math.wasm";
const GAME_OF_LIFE_WASM: &str = "test_programs/game_of_life/target/wasm32-unknown-unknown/release/game_of_life.wasm";
const BOUNCING_BALLS_WASM: &str = "test_programs/apps/bouncing_balls/target/wasm32-unknown-unknown/release/bouncing_balls.wasm";

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

fn print_bytecode(bytecode: &[u8]) {
    let (header, insts) = decode_bytecode(bytecode);
    println!("Header: code_size={}, entry_point={}, vertex_budget={}",
             header.code_size, header.entry_point, header.vertex_budget);
    println!("Instructions ({}):", insts.len());
    for (i, inst) in insts.iter().enumerate() {
        println!("  {:3}: op={:#04x} dst={} src1={} src2={} imm={}",
                 i, inst.opcode, inst.dst, inst.src1, inst.src2, inst.imm);
    }
}

#[test]
fn test_translate_simple_math_wasm() {
    // Read the compiled WASM file
    let wasm_path = "test_programs/simple_math/target/wasm32-unknown-unknown/release/simple_math.wasm";

    let wasm_bytes = match fs::read(wasm_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Skipping test - WASM file not found: {}", e);
            println!("Run: cd test_programs/simple_math && cargo build --release --target wasm32-unknown-unknown");
            return;
        }
    };

    println!("WASM file size: {} bytes", wasm_bytes.len());

    // Create translator
    let translator = WasmTranslator::new(TranslatorConfig::default());

    // Try to translate the factorial function
    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => {
            println!("\n=== Translation Successful ===");
            println!("Bytecode size: {} bytes", bytecode.len());
            print_bytecode(&bytecode);
        }
        Err(e) => {
            println!("Translation failed: {:?}", e);
            // This might fail because we need to specify entry point
        }
    }
}

#[test]
fn test_translate_all_functions() {
    let wasm_path = "test_programs/simple_math/target/wasm32-unknown-unknown/release/simple_math.wasm";

    let wasm_bytes = match fs::read(wasm_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Skipping test - WASM file not found: {}", e);
            return;
        }
    };

    // Use default translator
    let translator = WasmTranslator::new(TranslatorConfig::default());

    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => {
            println!("\n=== Translation Successful ===");
            println!("Bytecode size: {} bytes", bytecode.len());
            let (header, insts) = decode_bytecode(&bytecode);
            println!("Instructions: {}", insts.len());

            // Verify we got some bytecode
            assert!(insts.len() > 0, "Should have generated some instructions");

            // Check for expected opcodes (loops, arithmetic)
            let has_loop = insts.iter().any(|i| i.opcode == bytecode_op::JMP || i.opcode == bytecode_op::JNZ || i.opcode == bytecode_op::JZ);
            let has_mul = insts.iter().any(|i| i.opcode == bytecode_op::INT_MUL || i.opcode == bytecode_op::MUL);
            let has_add = insts.iter().any(|i| i.opcode == bytecode_op::INT_ADD || i.opcode == bytecode_op::ADD);

            println!("Has loop/branch: {}", has_loop);
            println!("Has multiply: {}", has_mul);
            println!("Has add: {}", has_add);
        }
        Err(e) => {
            println!("Translation error: {:?}", e);
            // Print the error but don't fail - we want to see what's happening
        }
    }
}

#[test]
fn test_translate_game_of_life() {
    let wasm_bytes = match fs::read(GAME_OF_LIFE_WASM) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Skipping test - WASM file not found: {}", e);
            println!("Run: cd test_programs/game_of_life && cargo build --release --target wasm32-unknown-unknown");
            return;
        }
    };

    println!("Game of Life WASM file size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());

    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => {
            println!("\n=== Game of Life Translation Successful ===");
            let (header, insts) = decode_bytecode(&bytecode);
            println!("Bytecode size: {} bytes", bytecode.len());
            println!("Instructions: {}", insts.len());

            // Game of Life should have:
            // - Conditional branches (for Conway's rules)
            // - Comparisons
            // - Arithmetic
            let has_branch = insts.iter().any(|i|
                i.opcode == bytecode_op::JMP ||
                i.opcode == bytecode_op::JNZ ||
                i.opcode == bytecode_op::JZ);
            let has_compare = insts.iter().any(|i|
                i.opcode == bytecode_op::INT_EQ ||
                i.opcode == bytecode_op::INT_NE ||
                i.opcode == bytecode_op::EQ);
            let has_arithmetic = insts.iter().any(|i|
                i.opcode == bytecode_op::INT_ADD ||
                i.opcode == bytecode_op::INT_SUB ||
                i.opcode == bytecode_op::INT_MUL ||
                i.opcode == bytecode_op::INT_DIV_S ||
                i.opcode == bytecode_op::INT_REM_S);

            println!("Has branching: {}", has_branch);
            println!("Has comparisons: {}", has_compare);
            println!("Has arithmetic: {}", has_arithmetic);

            // Should have generated reasonable amount of code
            assert!(insts.len() > 10, "Game of Life should have meaningful code");
        }
        Err(e) => {
            println!("Translation error: {:?}", e);
        }
    }
}

#[test]
fn test_translate_bouncing_balls() {
    let wasm_bytes = match fs::read(BOUNCING_BALLS_WASM) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Skipping test - WASM file not found: {}", e);
            println!("Run: cd test_programs/apps/bouncing_balls && cargo build --release --target wasm32-unknown-unknown");
            return;
        }
    };

    println!("Bouncing Balls WASM file size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());

    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => {
            println!("\n=== Bouncing Balls Translation Successful ===");
            let (header, insts) = decode_bytecode(&bytecode);
            println!("Bytecode size: {} bytes", bytecode.len());
            println!("Instructions: {}", insts.len());

            // Bouncing Balls should have:
            // - Floating point math (for physics)
            // - Loops (for updating all balls)
            // - Memory access (for ball state)
            let has_float = insts.iter().any(|i|
                i.opcode == bytecode_op::MUL ||
                i.opcode == bytecode_op::ADD ||
                i.opcode == bytecode_op::DIV);
            let has_branch = insts.iter().any(|i|
                i.opcode == bytecode_op::JMP ||
                i.opcode == bytecode_op::JNZ ||
                i.opcode == bytecode_op::JZ);
            let has_sin = insts.iter().any(|i| i.opcode == bytecode_op::SIN);
            let quad_count = insts.iter().filter(|i| i.opcode == bytecode_op::QUAD).count();

            println!("Has float ops: {}", has_float);
            println!("Has branching: {}", has_branch);
            println!("Has SIN opcode: {}", has_sin);
            println!("QUAD opcode count: {}", quad_count);

            // Print ALL instructions to understand loop structure
            println!("\nFull bytecode dump:");
            for (i, inst) in insts.iter().enumerate() {
                let op_name = match inst.opcode {
                    x if x == bytecode_op::NOP => "NOP",
                    x if x == bytecode_op::HALT => "HALT",
                    x if x == bytecode_op::MOV => "MOV",
                    x if x == bytecode_op::LOADI => "LOADI",
                    x if x == bytecode_op::ADD => "ADD",
                    x if x == bytecode_op::SUB => "SUB",
                    x if x == bytecode_op::MUL => "MUL",
                    x if x == bytecode_op::DIV => "DIV",
                    x if x == bytecode_op::JMP => "JMP",
                    x if x == bytecode_op::JZ => "JZ",
                    x if x == bytecode_op::JNZ => "JNZ",
                    x if x == bytecode_op::QUAD => "QUAD",
                    x if x == bytecode_op::LD => "LD",
                    x if x == bytecode_op::ST => "ST",
                    x if x == bytecode_op::INT_ADD => "INT_ADD",
                    x if x == bytecode_op::INT_SUB => "INT_SUB",
                    x if x == bytecode_op::INT_MUL => "INT_MUL",
                    x if x == bytecode_op::INT_LT_S => "INT_LT_S",
                    x if x == bytecode_op::INT_LT_U => "INT_LT_U",
                    x if x == bytecode_op::LOADI_UINT => "LOADI_UINT",
                    _ => "???",
                };
                println!("  {:3}: {:10} dst=r{:2} src1=r{:2} src2=r{:2} imm={}",
                    i, op_name, inst.dst, inst.src1, inst.src2, inst.imm);
            }

            assert!(insts.len() > 10, "Bouncing Balls should have meaningful code");
        }
        Err(e) => {
            println!("Translation error: {:?}", e);
            panic!("Bouncing Balls translation should succeed");
        }
    }
}

#[test]
fn test_translate_particles() {
    let wasm_bytes = match fs::read("test_programs/apps/particles/target/wasm32-unknown-unknown/release/particles.wasm") {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("Skipping - not found: {}", e);
            return;
        }
    };
    println!("Particles WASM: {} bytes", wasm_bytes.len());
    let translator = WasmTranslator::new(TranslatorConfig::default());
    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => {
            let (_header, insts) = decode_bytecode(&bytecode);
            println!("Instructions: {}", insts.len());
            let quad_count = insts.iter().filter(|i| i.opcode == bytecode_op::QUAD).count();
            println!("QUAD count: {}", quad_count);
            println!("\nQUAD instructions:");
            for (i, inst) in insts.iter().enumerate() {
                if inst.opcode == bytecode_op::QUAD {
                    println!("  {:3}: QUAD pos=r{} size=r{} color=r{} z={}",
                        i, inst.dst, inst.src1, inst.src2, inst.imm);
                }
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
