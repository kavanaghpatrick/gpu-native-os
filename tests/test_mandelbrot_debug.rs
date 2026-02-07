//! Debug test for Mandelbrot bytecode execution
//!
//! Dumps the bytecode and tests execution step by step.

use metal::Device;
use wasm_translator::{WasmTranslator, TranslatorConfig};
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type};
use std::fs;

// Bytecode opcodes from gpu_app_system.rs
const OP_HALT: u8 = 0xFF;
const OP_LOADI_INT: u8 = 0xDF;
const OP_LOADI_UINT: u8 = 0xE0;
const OP_LOADI_F32: u8 = 0xDD;
const OP_MOV: u8 = 0xDE;
const OP_LD: u8 = 0x20;
const OP_ST: u8 = 0x21;
const OP_JMP: u8 = 0x30;
const OP_JZ: u8 = 0x31;
const OP_JNZ: u8 = 0x32;
const OP_INT_ADD: u8 = 0x01;
const OP_INT_SUB: u8 = 0x02;
const OP_INT_MUL: u8 = 0x03;
const OP_INT_LT_S: u8 = 0x08;
const OP_INT_GT_S: u8 = 0x09;
const OP_INT_LE_S: u8 = 0x0A;
const OP_INT_GE_S: u8 = 0x0B;
const OP_INT_EQ: u8 = 0x0C;
const OP_INT_NE: u8 = 0x0D;
const OP_FLOAT_ADD: u8 = 0x40;
const OP_FLOAT_SUB: u8 = 0x41;
const OP_FLOAT_MUL: u8 = 0x42;
const OP_FLOAT_DIV: u8 = 0x43;
const OP_FLOAT_LT: u8 = 0x44;
const OP_FLOAT_GT: u8 = 0x45;
const OP_FLOAT_LE: u8 = 0x46;
const OP_FLOAT_GE: u8 = 0x47;
const OP_INT_TO_FLOAT: u8 = 0x50;
const OP_EMIT_QUAD: u8 = 0xA0;

fn opcode_name(opcode: u8) -> &'static str {
    match opcode {
        0x00 => "NOP",
        0xFF => "HALT",
        0xDF => "LOADI_INT",
        0xE0 => "LOADI_UINT",
        0xDD => "LOADI_F32",
        0xDE => "MOV",
        0x01 => "INT_ADD",
        0x02 => "INT_SUB",
        0x03 => "INT_MUL",
        0x04 => "MUL",
        0x05 => "DIV",
        0x08 => "INT_LT_S",
        0x09 => "INT_LT_U",
        0x0A => "INT_LE_S",
        0x0B => "INT_LE_U",
        0x0C => "INT_EQ",
        0x0D => "INT_NE",
        0x13 => "LOADI",
        0x40 => "FLOAT_ADD",
        0x41 => "FLOAT_SUB",
        0x42 => "FLOAT_MUL",
        0x43 => "FLOAT_DIV",
        0x44 => "FLOAT_LT",
        0x45 => "FLOAT_GT",
        0x46 => "FLOAT_LE",
        0x47 => "FLOAT_GE",
        0x50 => "INT_TO_F",
        0x51 => "UINT_TO_F",
        0x52 => "F_TO_INT",
        0x53 => "F_TO_UINT",
        0x60 => "JMP",
        0x61 => "JZ",
        0x62 => "JNZ",
        0x80 => "LD",
        0x81 => "ST",
        0xA0 => "QUAD",
        0xC0 => "BIT_AND",
        0xC1 => "BIT_OR",
        0xCB => "SHR_U",
        0xCE => "SHL",
        0xD4 => "INT_DIV_S",
        0xD5 => "INT_DIV_U",
        0xDA => "INT_REM_S",
        0xDB => "INT_REM_U",
        _ => "UNKNOWN",
    }
}

#[test]
fn test_mandelbrot_bytecode_dump() {
    let wasm_path = "test_programs/apps/mandelbrot_interactive/target/wasm32-unknown-unknown/release/mandelbrot_interactive.wasm";

    let wasm_bytes = fs::read(wasm_path).expect("Failed to read WASM file");
    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    // Parse header
    let code_size = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    let entry_point = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
    let vertex_budget = u32::from_le_bytes([bytecode[8], bytecode[9], bytecode[10], bytecode[11]]);
    let flags = u32::from_le_bytes([bytecode[12], bytecode[13], bytecode[14], bytecode[15]]);

    println!("═══════════════════════════════════════════════════════════════");
    println!("MANDELBROT BYTECODE ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Header:");
    println!("  code_size: {} instructions", code_size);
    println!("  entry_point: {}", entry_point);
    println!("  vertex_budget: {}", vertex_budget);
    println!("  flags: {}", flags);
    println!();

    // Count opcodes
    let mut opcode_counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut instructions = Vec::new();

    for i in 0..code_size as usize {
        let offset = 16 + i * 8;
        let opcode = bytecode[offset];
        let dst = bytecode[offset + 1];
        let src1 = bytecode[offset + 2];
        let src2 = bytecode[offset + 3];
        let imm = f32::from_le_bytes([
            bytecode[offset + 4],
            bytecode[offset + 5],
            bytecode[offset + 6],
            bytecode[offset + 7],
        ]);

        *opcode_counts.entry(opcode).or_insert(0) += 1;
        instructions.push((i, opcode, dst, src1, src2, imm));
    }

    println!("Opcode frequency:");
    let mut counts: Vec<_> = opcode_counts.iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (opcode, count) in counts.iter() {
        println!("  {:15} (0x{:02X}): {}", opcode_name(**opcode), opcode, count);
    }
    println!();

    // Print instructions around loop entry (50-90) with raw bytes
    println!("Instructions 50-65 (around suspicious MOV):");
    for i in 50..65 {
        let offset = 16 + i * 8;
        if offset + 8 > bytecode.len() { break; }

        let op = bytecode[offset];
        let d = bytecode[offset + 1];
        let s1 = bytecode[offset + 2];
        let s2 = bytecode[offset + 3];
        let imm_bytes = [bytecode[offset+4], bytecode[offset+5], bytecode[offset+6], bytecode[offset+7]];
        let imm_f32 = f32::from_le_bytes(imm_bytes);
        let imm_u32 = u32::from_le_bytes(imm_bytes);

        println!("  {:4}: {:02x} {:02x} {:02x} {:02x} {:02x}{:02x}{:02x}{:02x} | {:15} r{} r{} r{} | imm_f32={:.6} imm_u32={}",
                 i, op, d, s1, s2, imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3],
                 opcode_name(op), d, s1, s2, imm_f32, imm_u32);
    }

    // Look for JMP/JZ instructions to find loops
    println!();
    println!("Control flow (JMP/JZ/JNZ):");
    for (i, opcode, dst, src1, src2, imm) in instructions.iter() {
        if *opcode == 0x60 || *opcode == 0x61 || *opcode == 0x62 {  // JMP, JZ, JNZ
            let target = *imm as u32;
            let direction = if target < *i as u32 { "BACK (loop)" } else { "FWD" };
            println!("  {:4}: {:15} r{} -> target {} ({})",
                     i, opcode_name(*opcode), src1, target, direction);
        }
    }

    // Check for backward jumps (loops)
    println!();
    println!("Backward jumps (actual loops):");
    let mut loop_count = 0;
    for (i, opcode, dst, src1, src2, imm) in instructions.iter() {
        if *opcode == 0x60 || *opcode == 0x61 || *opcode == 0x62 {
            let target = *imm as u32;
            if target < *i as u32 {
                println!("  Loop found: inst {} jumps back to {}", i, target);
                loop_count += 1;
            }
        }
    }
    println!("Total backward jumps (loops): {}", loop_count);
}

#[test]
fn test_mandelbrot_execution() {
    let device = Device::system_default().expect("No Metal device found");

    let wasm_path = "test_programs/apps/mandelbrot_interactive/target/wasm32-unknown-unknown/release/mandelbrot_interactive.wasm";
    let wasm_bytes = fs::read(wasm_path).expect("Failed to read WASM file");
    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("═══════════════════════════════════════════════════════════════");
    println!("MANDELBROT GPU EXECUTION");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Bytecode size: {} bytes", bytecode.len());

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 4096 = 64*64)", result);

    assert!(result >= 4096, "Mandelbrot should render 4096 quads, got {}", result);
}
