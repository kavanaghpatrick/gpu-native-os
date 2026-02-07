# PRD: WASM Translator Safeguards

## Overview

**THE GPU IS THE COMPUTER.**

This PRD defines safeguards for the WASM-to-GPU bytecode translator to ensure robustness against edge cases identified during architecture validation.

## Background

The register recycling fix (free_pool) works correctly for LLVM-optimized WASM because:
1. LLVM uses locals for values that persist across blocks
2. Operand stack stays shallow (typically < 10)
3. Block boundaries have empty/near-empty operand stacks

However, Codex's analysis identified edge cases that could break these assumptions:
- Debug builds (`-Copt-level=0`)
- Manual WASM via `global_asm!`/`wasm_bindgen`
- Future LLVM changes
- Multi-value blocks (WASM proposal)

## Goals

1. **Detect violations early** - Assert when assumptions are violated
2. **Graceful degradation** - Spill to GPU memory instead of failing
3. **Validate with real code** - Test unmodified Rust projects from the wild

## Non-Goals

- Full liveness analysis (too complex, not needed for LLVM-generated WASM)
- Stack-based bytecode rewrite (current approach is sound)

---

## Safeguard 1: Stack-Empty Assertion

### Problem
`reset_temps()` assumes the operand stack is empty at block boundaries. If this assumption is violated, registers could be incorrectly reused.

### Solution
Add debug assertion that warns (not fails) when stack is non-empty at block boundaries.

### Pseudocode

```rust
// In stack.rs
pub fn reset_temps(&mut self) {
    // SAFEGUARD: Warn if stack is non-empty at block boundary
    // This indicates LLVM behavior we didn't expect
    #[cfg(debug_assertions)]
    if !self.stack.is_empty() {
        eprintln!(
            "[WARN] reset_temps called with non-empty stack: depth={} registers={:?}",
            self.stack.len(),
            self.stack
        );
        // Don't reset if stack has live values - preserve them
        // Just clear the free pool to prevent reuse of potentially live registers
        self.free_pool.clear();
        return;
    }

    self.next_temp = 8;
    self.free_pool.clear();
}
```

### Test

```rust
#[test]
fn test_reset_temps_with_empty_stack() {
    let mut stack = OperandStack::new();
    stack.reset_temps();
    assert_eq!(stack.depth(), 0);
}

#[test]
fn test_reset_temps_preserves_live_values() {
    let mut stack = OperandStack::new();
    let r1 = stack.alloc_and_push().unwrap();
    let r2 = stack.alloc_and_push().unwrap();

    // Stack has 2 values - reset should preserve them
    stack.reset_temps();

    assert_eq!(stack.depth(), 2);
    assert_eq!(stack.peek().unwrap(), r2);
}
```

---

## Safeguard 2: Spill Fallback

### Problem
When all 20 temp registers are exhausted and free_pool is empty, translation fails with `OutOfRegisters`.

### Solution
Instead of failing, spill to a per-thread memory area in the GPU state buffer.

### Pseudocode

```rust
// In stack.rs
pub struct OperandStack {
    stack: Vec<u8>,
    next_temp: u8,
    max_temp: u8,
    free_pool: Vec<u8>,
    // NEW: Spill tracking
    spill_count: u32,
    spill_base: u32,  // Base address in GPU state buffer
}

// Special register value indicating spilled value
const SPILLED_MARKER: u8 = 0xFE;

impl OperandStack {
    pub fn alloc_temp(&mut self) -> Result<u8, TranslateError> {
        // 1. Try free pool first
        if let Some(reg) = self.free_pool.pop() {
            return Ok(reg);
        }

        // 2. Try allocating new register
        if self.next_temp < self.max_temp {
            let reg = self.next_temp;
            self.next_temp += 1;
            return Ok(reg);
        }

        // 3. SAFEGUARD: Spill to memory instead of failing
        #[cfg(debug_assertions)]
        eprintln!("[SPILL] Register pressure exceeded, spilling to GPU memory");

        // Return marker indicating this value is spilled
        // The emitter will generate load/store instructions
        self.spill_count += 1;
        Ok(SPILLED_MARKER)
    }

    /// Check if a register is actually a spill marker
    pub fn is_spilled(&self, reg: u8) -> bool {
        reg == SPILLED_MARKER
    }

    /// Get the next spill address
    pub fn next_spill_addr(&mut self) -> u32 {
        let addr = self.spill_base + self.spill_count;
        self.spill_count += 1;
        addr
    }
}

// In translate.rs - handle spilled values
fn emit_binary_op(&mut self, op: BinaryOp) -> Result<(), TranslateError> {
    let b = self.stack.pop()?;
    let a = self.stack.pop()?;
    let dst = self.stack.alloc_and_push()?;

    // Load spilled values into scratch registers
    let actual_a = if self.stack.is_spilled(a) {
        let scratch = self.stack.scratch(0);
        self.emit.ld(scratch, /* spill addr */, 0.0);
        scratch
    } else {
        a
    };

    let actual_b = if self.stack.is_spilled(b) {
        let scratch = self.stack.scratch(1);
        self.emit.ld(scratch, /* spill addr */, 0.0);
        scratch
    } else {
        b
    };

    // Emit the actual operation
    self.emit.binary(op, dst, actual_a, actual_b);

    // If dst is spilled, store it
    if self.stack.is_spilled(dst) {
        self.emit.st(/* spill addr */, dst, 0.0);
    }

    Ok(())
}
```

### Test

```rust
#[test]
fn test_spill_fallback_under_pressure() {
    let mut stack = OperandStack::new();

    // Exhaust all 20 temp registers
    for _ in 0..20 {
        stack.alloc_and_push().unwrap();
    }

    // 21st allocation should spill, not fail
    let result = stack.alloc_temp();
    assert!(result.is_ok(), "Should spill instead of failing");

    let reg = result.unwrap();
    assert!(stack.is_spilled(reg), "Should be marked as spilled");
}
```

---

## Safeguard 3: Debug Build Compatibility

### Problem
Debug builds (`-Copt-level=0`) generate more verbose WASM with:
- More block/loop constructs
- Deeper operand stacks
- Less aggressive use of `select`

### Solution
Test the translator with debug-built WASM to ensure it handles these patterns.

### Test Plan

```rust
#[test]
fn test_translate_debug_build() {
    // Compile test program with opt-level=0
    // test_programs/debug_test/Cargo.toml:
    // [profile.dev]
    // opt-level = 0

    let wasm_path = "test_programs/debug_test/target/wasm32-unknown-unknown/debug/debug_test.wasm";
    let wasm_bytes = fs::read(wasm_path).expect("Debug WASM not found");

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let result = translator.translate(&wasm_bytes);

    // Should succeed (possibly with spills) not fail
    assert!(result.is_ok(), "Debug build should translate: {:?}", result.err());
}
```

---

## Safeguard 4: Real-World Rust Code Testing

### Problem
We've only tested our own simple programs. Real Rust code may have patterns we haven't anticipated.

### Solution
Test with unmodified Rust code from popular crates/projects.

### Candidates

| Project | Why | Complexity |
|---------|-----|------------|
| `no_std` math libraries | Pure computation, no allocator | Low |
| `heapless` data structures | Fixed-size, no_std | Medium |
| `micromath` | Embedded math, no_std | Low |
| `hash32` | Simple hashing, no_std | Low |

### Test Plan

```rust
#[test]
fn test_real_world_micromath() {
    // Clone and build micromath with WASM target
    // Extract a simple function and test translation
}

#[test]
fn test_real_world_hash32() {
    // FNV-1a hash is simple and commonly used
}
```

---

## Implementation Order

1. **Safeguard 1: Stack-empty assertion** (30 min)
   - Modify `reset_temps()` in `stack.rs`
   - Add unit tests

2. **Safeguard 3: Debug build test** (30 min)
   - Create debug test program
   - Verify current translator handles it

3. **Safeguard 4: Real-world testing** (1-2 hours)
   - Set up test harness for external crates
   - Test 3-4 real projects

4. **Safeguard 2: Spill fallback** (2-3 hours) - ONLY IF NEEDED
   - Only implement if real-world testing reveals failures
   - More complex, may not be necessary

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Debug build translation | Works (possibly with warnings) |
| Real-world crate translation | 3+ crates translate successfully |
| Spill events in optimized code | 0 (free_pool should suffice) |
| Regression in existing tests | None |

---

## Files to Modify

- `wasm_translator/src/stack.rs` - Safeguards 1 & 2
- `wasm_translator/src/translate.rs` - Spill handling (if needed)
- `tests/test_real_wasm_programs.rs` - Real-world tests
- `test_programs/debug_test/` - Debug build test fixture

---

## Risks

| Risk | Mitigation |
|------|------------|
| Spill overhead slows GPU execution | Only triggers for pathological code |
| Real-world code uses unsupported WASM features | Document limitations, add graceful errors |
| Debug assertion false positives | Make it a warning, not a failure |
