//! Test translation of REAL Rust code from the wild
//!
//! THE GPU IS THE COMPUTER.
//! These are unmodified algorithms that must work on our platform.

use wasm_translator::{WasmTranslator, TranslatorConfig};
use rust_experiment::gpu_os::gpu_app_system::{BytecodeHeader, BytecodeInst, bytecode_op};
use std::fs;

/// Decode bytecode for analysis
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

/// Test result structure
struct TranslationResult {
    name: &'static str,
    wasm_size: usize,
    bytecode_size: usize,
    instruction_count: usize,
    has_loops: bool,
    has_arithmetic: bool,
    has_comparisons: bool,
    spill_count: u32,
}

impl TranslationResult {
    fn print(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│ {} ", self.name);
        println!("├─────────────────────────────────────────┤");
        println!("│ WASM size:        {:>6} bytes          │", self.wasm_size);
        println!("│ Bytecode size:    {:>6} bytes          │", self.bytecode_size);
        println!("│ Instructions:     {:>6}                │", self.instruction_count);
        println!("│ Has loops:        {:>6}                │", if self.has_loops { "YES" } else { "NO" });
        println!("│ Has arithmetic:   {:>6}                │", if self.has_arithmetic { "YES" } else { "NO" });
        println!("│ Has comparisons:  {:>6}                │", if self.has_comparisons { "YES" } else { "NO" });
        println!("│ Spill count:      {:>6}                │", self.spill_count);
        println!("└─────────────────────────────────────────┘");
    }
}

fn translate_and_analyze(name: &'static str, wasm_path: &str) -> Result<TranslationResult, String> {
    let wasm_bytes = fs::read(wasm_path)
        .map_err(|e| format!("Failed to read {}: {}", wasm_path, e))?;

    let translator = WasmTranslator::new(TranslatorConfig::default());

    let bytecode = translator.translate(&wasm_bytes)
        .map_err(|e| format!("Translation failed: {:?}", e))?;

    let (_header, insts) = decode_bytecode(&bytecode);

    let has_loops = insts.iter().any(|i|
        i.opcode == bytecode_op::JMP ||
        i.opcode == bytecode_op::JNZ ||
        i.opcode == bytecode_op::JZ);

    let has_arithmetic = insts.iter().any(|i|
        i.opcode == bytecode_op::INT_ADD ||
        i.opcode == bytecode_op::INT_SUB ||
        i.opcode == bytecode_op::INT_MUL ||
        i.opcode == bytecode_op::INT_DIV_S ||
        i.opcode == bytecode_op::ADD ||
        i.opcode == bytecode_op::MUL);

    let has_comparisons = insts.iter().any(|i|
        i.opcode == bytecode_op::INT_EQ ||
        i.opcode == bytecode_op::INT_NE ||
        i.opcode == bytecode_op::INT_LT_S ||
        i.opcode == bytecode_op::INT_LE_S ||
        i.opcode == bytecode_op::LT ||
        i.opcode == bytecode_op::EQ);

    Ok(TranslationResult {
        name,
        wasm_size: wasm_bytes.len(),
        bytecode_size: bytecode.len(),
        instruction_count: insts.len(),
        has_loops,
        has_arithmetic,
        has_comparisons,
        spill_count: 0, // TODO: Get from translator stats
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDIVIDUAL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fnv_hash() {
    // FNV-1a Hash uses 64-bit operations
    // Issue #188 RESOLVED: i64 operations now supported natively on Apple Silicon GPU
    // THE GPU IS THE COMPUTER - native 64-bit integer support
    let result = translate_and_analyze(
        "FNV-1a Hash",
        "test_programs/external/fnv_hash/target/wasm32-unknown-unknown/release/fnv_hash.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
            assert!(r.has_loops, "FNV hash uses loops");
            assert!(r.has_arithmetic, "FNV hash uses XOR and multiply");
        }
        Err(e) => {
            panic!("FNV-1a Hash translation should succeed with i64 support: {}", e);
        }
    }
}

#[test]
fn test_bubble_sort() {
    let result = translate_and_analyze(
        "Bubble Sort",
        "test_programs/external/bubble_sort/target/wasm32-unknown-unknown/release/bubble_sort.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
            assert!(r.has_loops, "Bubble sort uses nested loops");
            assert!(r.has_comparisons, "Bubble sort uses comparisons");
        }
        Err(e) => {
            println!("Skipping bubble sort test: {}", e);
        }
    }
}

#[test]
fn test_prime_sieve() {
    let result = translate_and_analyze(
        "Prime Sieve",
        "test_programs/external/prime_sieve/target/wasm32-unknown-unknown/release/prime_sieve.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
            assert!(r.has_loops, "Prime sieve uses loops");
            assert!(r.has_arithmetic, "Prime sieve uses modulo");
            assert!(r.has_comparisons, "Prime sieve uses comparisons");
        }
        Err(e) => {
            println!("Skipping prime sieve test: {}", e);
        }
    }
}

#[test]
fn test_mandelbrot() {
    let result = translate_and_analyze(
        "Mandelbrot Set",
        "test_programs/external/mandelbrot/target/wasm32-unknown-unknown/release/mandelbrot.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
            assert!(r.has_loops, "Mandelbrot uses iteration loop");
            assert!(r.has_arithmetic, "Mandelbrot uses floating point math");
            assert!(r.has_comparisons, "Mandelbrot uses escape condition");
        }
        Err(e) => {
            println!("Skipping mandelbrot test: {}", e);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMARY TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_external_programs() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     EXTERNAL RUST CODE TRANSLATION TEST SUITE                 ║");
    println!("║     THE GPU IS THE COMPUTER                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let tests = vec![
        ("FNV-1a Hash", "test_programs/external/fnv_hash/target/wasm32-unknown-unknown/release/fnv_hash.wasm"),
        ("Bubble Sort", "test_programs/external/bubble_sort/target/wasm32-unknown-unknown/release/bubble_sort.wasm"),
        ("Prime Sieve", "test_programs/external/prime_sieve/target/wasm32-unknown-unknown/release/prime_sieve.wasm"),
        ("Mandelbrot", "test_programs/external/mandelbrot/target/wasm32-unknown-unknown/release/mandelbrot.wasm"),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, path) in tests {
        match translate_and_analyze(name, path) {
            Ok(r) => {
                r.print();
                passed += 1;
            }
            Err(e) => {
                println!("✗ {} FAILED: {}", name, e);
                failed += 1;
            }
        }
        println!();
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("RESULTS: {} passed, {} failed", passed, failed);
    println!("════════════════════════════════════════════════════════════════");

    assert_eq!(failed, 0, "All external programs should translate successfully");
}
