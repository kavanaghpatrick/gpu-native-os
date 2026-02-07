//! Test to profile and demonstrate OutOfRegisters error
//!
//! THE GPU IS THE COMPUTER.
//! This test creates register pressure scenarios to understand allocation limits.

use wasm_translator::{WasmTranslator, TranslatorConfig, TranslateError};

/// Helper to count instructions until OutOfRegisters
fn translate_wat_and_count(wat: &str) -> Result<usize, TranslateError> {
    // Parse WAT to WASM binary
    let wasm_bytes = wat::parse_str(wat)
        .map_err(|e| TranslateError::Parse(e.to_string()))?;

    let translator = WasmTranslator::new(TranslatorConfig::default());
    match translator.translate(&wasm_bytes) {
        Ok(bytecode) => Ok(bytecode.len()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_register_limit_no_blocks() {
    println!("\n=== Testing register limits WITHOUT block boundaries ===\n");

    // This WAT has 21 consecutive LocalGet operations without any block boundaries
    // Each LocalGet allocates a new temp register (r8-r27 = 20 registers)
    // The 21st should fail with OutOfRegisters

    let wat = r#"
    (module
      (func (export "main") (param $x i32) (result i32)
        ;; Each local.get allocates a temp register
        ;; We have 20 temps (r8-r27)
        ;; After 20, we should run out
        (i32.add
          (i32.add
            (i32.add
              (i32.add
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.add
                        (i32.add
                          (i32.add
                            (i32.add
                              (i32.add
                                (i32.add
                                  (i32.add
                                    (i32.add
                                      (i32.add
                                        (i32.add
                                          (i32.add
                                            (i32.add
                                              (i32.add
                                                (local.get $x)
                                                (local.get $x))
                                              (local.get $x))
                                            (local.get $x))
                                          (local.get $x))
                                        (local.get $x))
                                      (local.get $x))
                                    (local.get $x))
                                  (local.get $x))
                                (local.get $x))
                              (local.get $x))
                            (local.get $x))
                          (local.get $x))
                        (local.get $x))
                      (local.get $x))
                    (local.get $x))
                  (local.get $x))
                (local.get $x))
              (local.get $x))
            (local.get $x))
          (local.get $x))
      )
    )
    "#;

    match translate_wat_and_count(wat) {
        Ok(size) => println!("SUCCESS: Generated {} bytes of bytecode (fixed version with block resets)", size),
        Err(e) => println!("EXPECTED FAILURE: {:?}", e),
    }
}

#[test]
fn test_register_limit_with_blocks() {
    println!("\n=== Testing register recycling WITH block boundaries ===\n");

    // This WAT uses blocks to recycle registers
    // Each block resets the temp counter, so we never run out

    let wat = r#"
    (module
      (func (export "main") (param $x i32) (result i32)
        ;; Use blocks to recycle registers
        (block $b1 (result i32)
          (i32.add
            (i32.add
              (i32.add
                (i32.add
                  (local.get $x)
                  (local.get $x))
                (local.get $x))
              (local.get $x))
            (local.get $x))
        )
      )
    )
    "#;

    match translate_wat_and_count(wat) {
        Ok(size) => println!("SUCCESS: Generated {} bytes with block recycling", size),
        Err(e) => println!("FAILURE: {:?}", e),
    }
}

#[test]
fn test_find_exact_limit() {
    println!("\n=== Finding exact register limit ===\n");

    // Build progressively deeper nested adds until we hit the limit
    // Note: Each i32.add:
    //   1. Pops 2 regs from stack
    //   2. Allocates 1 new temp for result
    //   3. Net stack effect: -1 (consumes 2, produces 1)

    // The maximum depth depends on how many values are simultaneously live
    // With nested adds: (add (add (add (get) (get)) (get)) (get))
    // Maximum stack depth = log2(leaves) values need to be live at once

    for depth in 1..=30 {
        // Create WAT with 'depth' nested local.gets
        let mut inner = "(local.get $x)".to_string();
        for _ in 1..depth {
            inner = format!("(i32.add {} (local.get $x))", inner);
        }

        let wat = format!(r#"
        (module
          (func (export "main") (param $x i32) (result i32)
            {}
          )
        )
        "#, inner);

        match translate_wat_and_count(&wat) {
            Ok(size) => println!("Depth {:2}: SUCCESS ({} bytes)", depth, size),
            Err(TranslateError::OutOfRegisters) => {
                println!("Depth {:2}: OutOfRegisters - LIMIT REACHED", depth);
                println!("\n>>> FAILURE POINT: {} nested operations <<<", depth);
                break;
            }
            Err(e) => println!("Depth {:2}: Other error: {:?}", depth, e),
        }
    }
}

#[test]
fn test_stack_growth_pattern() {
    println!("\n=== Testing stack growth pattern ===\n");
    println!("Pattern: Push many values, then consume them");

    // This pattern maximizes simultaneous register usage:
    // Push N values, then add them all together

    for count in 1..=25 {
        // Create WAT that pushes 'count' values then adds them
        let gets: String = (0..count).map(|_| "(local.get $x)").collect::<Vec<_>>().join(" ");
        let adds: String = (1..count).map(|_| "i32.add").collect::<Vec<_>>().join(" ");

        let wat = format!(r#"
        (module
          (func (export "main") (param $x i32) (result i32)
            {} {}
          )
        )
        "#, gets, adds);

        match translate_wat_and_count(&wat) {
            Ok(size) => println!("Push {:2} values: SUCCESS ({} bytes)", count, size),
            Err(TranslateError::OutOfRegisters) => {
                println!("Push {:2} values: OutOfRegisters", count);
                println!("\n>>> FAILURE POINT: {} simultaneous values <<<", count);
                break;
            }
            Err(e) => println!("Push {:2} values: {:?}", count, e),
        }
    }
}

#[test]
fn test_game_of_life_like_pattern() {
    println!("\n=== Testing Game of Life-like register pressure ===\n");

    // Simulate the pattern that Game of Life uses:
    // - Multiple coordinate calculations
    // - Each needs intermediate results
    // - All combined at the end

    // get_index(row, col) has ~10 operations:
    // ((row % H) + H) % H -> ~5 ops
    // ((col % W) + W) % W -> ~5 ops
    // result = row * W + col

    // count_neighbors calls get_index 8 times
    // If all results are kept live, that's 8 * ~15 = 120 intermediate values

    let wat = r#"
    (module
      (func (export "main") (param $idx i32) (result i32)
        (local $row i32)
        (local $col i32)

        ;; row = idx / 64
        (local.set $row (i32.div_s (local.get $idx) (i32.const 64)))
        ;; col = idx % 64
        (local.set $col (i32.rem_s (local.get $idx) (i32.const 64)))

        ;; Simulate get_index for 4 neighbors (simplified)
        ;; Each calculates ((row +/- 1) % 64 + 64) % 64 * 64 + ((col +/- 1) % 64 + 64) % 64

        (i32.add
          (i32.add
            (i32.add
              ;; North neighbor index
              (i32.add
                (i32.mul
                  (i32.rem_s
                    (i32.add
                      (i32.rem_s
                        (i32.sub (local.get $row) (i32.const 1))
                        (i32.const 64))
                      (i32.const 64))
                    (i32.const 64))
                  (i32.const 64))
                (local.get $col))
              ;; South neighbor index
              (i32.add
                (i32.mul
                  (i32.rem_s
                    (i32.add
                      (i32.rem_s
                        (i32.add (local.get $row) (i32.const 1))
                        (i32.const 64))
                      (i32.const 64))
                    (i32.const 64))
                  (i32.const 64))
                (local.get $col)))
            ;; West neighbor index
            (i32.add
              (i32.mul (local.get $row) (i32.const 64))
              (i32.rem_s
                (i32.add
                  (i32.rem_s
                    (i32.sub (local.get $col) (i32.const 1))
                    (i32.const 64))
                  (i32.const 64))
                (i32.const 64))))
          ;; East neighbor index
          (i32.add
            (i32.mul (local.get $row) (i32.const 64))
            (i32.rem_s
              (i32.add
                (i32.rem_s
                  (i32.add (local.get $col) (i32.const 1))
                  (i32.const 64))
                (i32.const 64))
              (i32.const 64))))
      )
    )
    "#;

    match translate_wat_and_count(wat) {
        Ok(size) => println!("Game of Life pattern: SUCCESS ({} bytes)", size),
        Err(e) => println!("Game of Life pattern: FAILURE - {:?}", e),
    }
}
