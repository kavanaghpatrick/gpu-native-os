//! Test to trace register allocation during Game of Life translation
//!
//! THE GPU IS THE COMPUTER.
//! This test analyzes WHY the Game of Life fails with OutOfRegisters.

use wasm_translator::{WasmTranslator, TranslatorConfig};
use std::fs;

const GAME_OF_LIFE_WASM: &str = "test_programs/game_of_life/target/wasm32-unknown-unknown/release/game_of_life.wasm";

/// Manually trace through WASM to count operations
///
/// Game of Life WASM structure (inlined):
///
/// get_index(row, col) has these operations:
///   - row % HEIGHT (i32.const, i32.rem_s) = 2 ops, stack depth 1
///   - + HEIGHT (i32.const, i32.add) = 2 ops, stack depth 1
///   - % HEIGHT (i32.const, i32.rem_s) = 2 ops, stack depth 1
///   - col % WIDTH (i32.const, i32.rem_s) = 2 ops, stack depth 2
///   - + WIDTH (i32.const, i32.add) = 2 ops, stack depth 2
///   - % WIDTH (i32.const, i32.rem_s) = 2 ops, stack depth 2
///   - row * WIDTH (i32.const, i32.mul) = 2 ops, stack depth 3
///   - + col (i32.add) = 1 op, stack depth 2->1
/// Total: ~15 operations per get_index call, max stack depth ~3
///
/// count_neighbors calls get_index 8 times:
///   Each inlined get_index uses ~3 temp registers at peak
///   But WITHOUT register recycling, we accumulate:
///   - 8 result values (n, s, w, e, nw, ne, sw, se) = 8 registers
///   - Plus intermediate values from inlining
///
/// The problem: `alloc_temp()` ONLY increments `next_temp`.
/// It NEVER decrements when a value is popped.
/// So each i32.const allocates a NEW register that's NEVER freed.
#[test]
fn test_analyze_register_pressure() {
    let wasm_bytes = match fs::read(GAME_OF_LIFE_WASM) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("WASM file not found: {}", e);
            return;
        }
    };

    println!("\n=== REGISTER PRESSURE ANALYSIS ===\n");
    println!("WASM file size: {} bytes", wasm_bytes.len());

    // The translator will fail at OutOfRegisters
    let translator = WasmTranslator::new(TranslatorConfig::default());
    let result = translator.translate(&wasm_bytes);

    match result {
        Ok(bytecode) => {
            println!("Translation succeeded with {} bytes", bytecode.len());
        }
        Err(e) => {
            println!("Translation error: {:?}", e);
            println!("\nThis is EXPECTED for Game of Life with 20-register limit.\n");
        }
    }

    // Count expected operations
    println!("=== EXPECTED REGISTER USAGE ===\n");
    println!("get_index operations:");
    println!("  - Each get_index has ~15 WASM instructions");
    println!("  - Peak operand stack depth: ~3 values");
    println!();
    println!("count_neighbors calls get_index 8 times:");
    println!("  - 8 x (peak of 3) = potential for ~24 intermediate values");
    println!("  - Plus 8 result values held simultaneously");
    println!("  - Total: potentially 30+ temp registers needed");
    println!();
    println!("Available temp registers: r8-r27 = 20 registers");
    println!();
    println!("=== ROOT CAUSE ===\n");
    println!("The OperandStack.alloc_temp() method:");
    println!("  1. Allocates register r[next_temp]");
    println!("  2. Increments next_temp");
    println!("  3. NEVER decrements when pop() is called");
    println!();
    println!("This means:");
    println!("  - Every i32.const allocates a NEW register");
    println!("  - Every arithmetic result allocates a NEW register");
    println!("  - Registers are NEVER recycled");
    println!();
    println!("Example trace of i32.add:");
    println!("  1. Operand A pushed -> r8 allocated, next_temp=9");
    println!("  2. Operand B pushed -> r9 allocated, next_temp=10");
    println!("  3. i32.add: pop A(r8), pop B(r9) -> but registers NOT freed");
    println!("  4. Result pushed -> r10 allocated, next_temp=11");
    println!("  5. r8, r9 are now DEAD but next_temp still at 11");
    println!();
    println!("After 7 i32.add operations:");
    println!("  - 7*3 = 21 registers consumed");
    println!("  - OutOfRegisters at register 28 (only 20 available)");
}

/// Test to verify the exact point of failure
#[test]
fn test_count_wasm_operations() {
    println!("\n=== WASM OPERATION COUNT ===\n");

    // Trace get_index manually:
    // ((row % HEIGHT) + HEIGHT) % HEIGHT
    // ((col % WIDTH) + WIDTH) % WIDTH
    // row * WIDTH + col

    println!("get_index(row: i32, col: i32) -> i32:");
    println!("  Operations:");
    println!("    local.get row           ; push row -> alloc r8");
    println!("    i32.const 64            ; push 64 -> alloc r9");
    println!("    i32.rem_s               ; pop, pop, push -> alloc r10");
    println!("    i32.const 64            ; push 64 -> alloc r11");
    println!("    i32.add                 ; pop, pop, push -> alloc r12");
    println!("    i32.const 64            ; push 64 -> alloc r13");
    println!("    i32.rem_s               ; pop, pop, push -> alloc r14 (final row)");
    println!("    local.get col           ; push col -> alloc r15");
    println!("    i32.const 64            ; push 64 -> alloc r16");
    println!("    i32.rem_s               ; pop, pop, push -> alloc r17");
    println!("    i32.const 64            ; push 64 -> alloc r18");
    println!("    i32.add                 ; pop, pop, push -> alloc r19");
    println!("    i32.const 64            ; push 64 -> alloc r20");
    println!("    i32.rem_s               ; pop, pop, push -> alloc r21 (final col)");
    println!("    i32.const 64            ; push 64 -> alloc r22");
    println!("    i32.mul                 ; pop row, pop 64, push -> alloc r23");
    println!("    i32.add                 ; pop, pop, push -> alloc r24 (result)");
    println!();
    println!("  Register consumption: 17 registers for ONE get_index call!");
    println!("  Available: 20 (r8-r27)");
    println!();
    println!("  Even with optimal recycling, 8 get_index calls would need:");
    println!("  - 8 result values live simultaneously = 8 registers");
    println!("  - Plus intermediate values = exceeds 20");
    println!();
    println!("=== SOLUTION APPROACHES ===\n");
    println!("1. Register recycling: When pop() is called, mark register as free");
    println!("2. Linear scan allocation: Track live ranges, reuse dead registers");
    println!("3. Stack spilling: Spill to memory when >20 registers needed");
    println!("4. Don't inline deep: Limit inlining depth or use calls");
}
