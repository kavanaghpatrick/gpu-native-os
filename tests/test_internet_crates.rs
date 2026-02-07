//! Test REAL crates from crates.io
//!
//! THE GPU IS THE COMPUTER.
//! These are UNMODIFIED algorithms from the Rust ecosystem.

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
    crate_name: &'static str,
    wasm_size: usize,
    bytecode_size: usize,
    instruction_count: usize,
    has_64bit: bool,
}

impl TranslationResult {
    fn print(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│ {} ", self.name);
        println!("│ crates.io: {} ", self.crate_name);
        println!("├─────────────────────────────────────────┤");
        println!("│ WASM size:        {:>6} bytes          │", self.wasm_size);
        println!("│ Bytecode size:    {:>6} bytes          │", self.bytecode_size);
        println!("│ Instructions:     {:>6}                │", self.instruction_count);
        println!("│ Uses 64-bit:      {:>6}                │", if self.has_64bit { "YES" } else { "NO" });
        println!("└─────────────────────────────────────────┘");
    }
}

fn translate_and_analyze(name: &'static str, crate_name: &'static str, wasm_path: &str) -> Result<TranslationResult, String> {
    let wasm_bytes = fs::read(wasm_path)
        .map_err(|e| format!("Failed to read {}: {}", wasm_path, e))?;

    let translator = WasmTranslator::new(TranslatorConfig::default());

    let bytecode = translator.translate(&wasm_bytes)
        .map_err(|e| format!("Translation failed: {:?}", e))?;

    let (_header, insts) = decode_bytecode(&bytecode);

    let has_64bit = insts.iter().any(|i|
        i.opcode == bytecode_op::INT64_ADD ||
        i.opcode == bytecode_op::INT64_MUL ||
        i.opcode == bytecode_op::INT64_XOR ||
        i.opcode == bytecode_op::INT64_OR ||
        i.opcode == bytecode_op::INT64_AND ||
        i.opcode == bytecode_op::INT64_SHL ||
        i.opcode == bytecode_op::INT64_SHR_U);

    Ok(TranslationResult {
        name,
        crate_name,
        wasm_size: wasm_bytes.len(),
        bytecode_size: bytecode.len(),
        instruction_count: insts.len(),
        has_64bit,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDIVIDUAL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_xxhash() {
    // xxhash-rust: Fast non-cryptographic hash
    // https://crates.io/crates/xxhash-rust
    let result = translate_and_analyze(
        "xxHash32",
        "xxhash-rust",
        "test_programs/external/xxhash/target/wasm32-unknown-unknown/release/xxhash_test.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
        }
        Err(e) => {
            panic!("xxHash translation failed: {}", e);
        }
    }
}

#[test]
fn test_fastrand() {
    // fastrand: Fast random number generator
    // https://crates.io/crates/fastrand
    let result = translate_and_analyze(
        "fastrand RNG",
        "fastrand",
        "test_programs/external/fastrand/target/wasm32-unknown-unknown/release/fastrand_test.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
        }
        Err(e) => {
            panic!("fastrand translation failed: {}", e);
        }
    }
}

#[test]
fn test_siphasher() {
    // siphasher: SipHash implementation (used by Rust's HashMap!)
    // https://crates.io/crates/siphasher
    let result = translate_and_analyze(
        "SipHash-1-3",
        "siphasher",
        "test_programs/external/siphasher/target/wasm32-unknown-unknown/release/siphasher_test.wasm"
    );

    match result {
        Ok(r) => {
            r.print();
            assert!(r.instruction_count > 0, "Should generate instructions");
            // SipHash uses 64-bit operations
            assert!(r.has_64bit, "SipHash should use 64-bit operations");
        }
        Err(e) => {
            panic!("SipHash translation failed: {}", e);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMARY TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_internet_crates() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     REAL CRATES FROM CRATES.IO - TRANSLATION TEST             ║");
    println!("║     THE GPU IS THE COMPUTER                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let tests = vec![
        ("xxHash32", "xxhash-rust", "test_programs/external/xxhash/target/wasm32-unknown-unknown/release/xxhash_test.wasm"),
        ("fastrand RNG", "fastrand", "test_programs/external/fastrand/target/wasm32-unknown-unknown/release/fastrand_test.wasm"),
        ("SipHash-1-3", "siphasher", "test_programs/external/siphasher/target/wasm32-unknown-unknown/release/siphasher_test.wasm"),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, crate_name, path) in tests {
        match translate_and_analyze(name, crate_name, path) {
            Ok(r) => {
                r.print();
                passed += 1;
            }
            Err(e) => {
                println!("✗ {} ({}) FAILED: {}", name, crate_name, e);
                failed += 1;
            }
        }
        println!();
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("RESULTS: {} passed, {} failed", passed, failed);
    println!("════════════════════════════════════════════════════════════════");

    assert_eq!(failed, 0, "All internet crates should translate successfully");
}
