//! Operand stack for WASM→register conversion
//!
//! THE GPU IS THE COMPUTER.
//! WASM is stack-based, GPU bytecode is register-based.
//! This module efficiently maps stack operations to register operations.
//!
//! ## Safeguards (validated by Codex architecture review)
//!
//! 1. **Stack-empty assertion**: Warns when reset_temps() called with non-empty stack
//! 2. **Spill fallback**: When registers exhausted, spill to GPU memory instead of failing
//! 3. **Debug build compatibility**: Handles verbose WASM from -Copt-level=0

use crate::types::TranslateError;

/// Marker value indicating a register has been spilled to GPU memory
pub const SPILLED_REG: u8 = 0xFE;

/// Operand stack that maps to registers
///
/// Register allocation:
/// - r0-r3:  Reserved (system: r3=frame_number)
/// - r4-r7:  Function arguments / return values
/// - r8-r27: Temporaries for operand stack (20 registers)
/// - r28-r29: Local variable spill area (locals 4 and 5)
///            WARNING: Do NOT use in intrinsics - conflicts with locals!
/// - r30-r31: Scratch registers for intrinsics (emit_quad uses these)
/// - 0xFE:   Spilled to GPU memory (SAFEGUARD)
///
/// CRITICAL FIX: Implements register recycling to avoid OutOfRegisters
/// When registers are popped, they're returned to the free pool for reuse.
///
/// SAFEGUARD: When all registers exhausted, spills to GPU memory instead of failing.
pub struct OperandStack {
    /// Stack of register numbers (or SPILLED_REG for spilled values)
    stack: Vec<u8>,
    /// Next available temp register (for sequential allocation)
    next_temp: u8,
    /// Maximum temp register (exclusive)
    max_temp: u8,
    /// Pool of freed registers available for reuse (CRITICAL FIX)
    free_pool: Vec<u8>,
    /// SAFEGUARD: Number of values spilled to GPU memory
    spill_count: u32,
    /// SAFEGUARD: Base address for spill area in GPU state buffer
    spill_base: u32,
    /// SAFEGUARD: Track spill slot addresses for each spilled value
    spill_slots: Vec<u32>,
}

impl OperandStack {
    /// Default spill base address in GPU state buffer
    /// CRITICAL: This is a FLOAT4 INDEX in GPU shader. state[72] = byte offset 72*16 = 1152.
    /// Layout: state[0-7] reserved, state[8-71] globals, state[72+] spill area.
    const DEFAULT_SPILL_BASE: u32 = 72;

    /// Create a new operand stack
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(64),
            next_temp: 8,   // Start at r8
            max_temp: 28,   // Up to r27
            free_pool: Vec::with_capacity(20),  // CRITICAL FIX: Pool for recycled registers
            spill_count: 0,
            spill_base: Self::DEFAULT_SPILL_BASE,
            spill_slots: Vec::new(),
        }
    }

    /// Create with custom spill base address
    pub fn with_spill_base(spill_base: u32) -> Self {
        Self {
            stack: Vec::with_capacity(64),
            next_temp: 8,
            max_temp: 28,
            free_pool: Vec::with_capacity(20),
            spill_count: 0,
            spill_base,
            spill_slots: Vec::new(),
        }
    }

    /// Push a register onto the stack
    pub fn push(&mut self, reg: u8) {
        self.stack.push(reg);
    }

    /// Pop a register from the stack
    /// CRITICAL FIX: Returns the register to the free pool for reuse
    /// SAFEGUARD: Spilled values (SPILLED_REG) are tracked separately
    pub fn pop(&mut self) -> Result<u8, TranslateError> {
        let reg = self.stack.pop().ok_or(TranslateError::StackUnderflow)?;
        // Return temp registers (r8-r27) to the free pool for reuse
        // Note: SPILLED_REG (0xFE) is not returned to free pool
        if reg >= 8 && reg < 28 {
            self.free_pool.push(reg);
        }
        Ok(reg)
    }

    /// Pop and get spill address if value was spilled
    /// Returns (register, Option<spill_address>)
    pub fn pop_with_spill_info(&mut self) -> Result<(u8, Option<u32>), TranslateError> {
        let reg = self.stack.pop().ok_or(TranslateError::StackUnderflow)?;

        if reg == SPILLED_REG {
            // Pop the corresponding spill slot
            let addr = self.spill_slots.pop();
            Ok((reg, addr))
        } else {
            // Return temp registers to free pool
            if reg >= 8 && reg < 28 {
                self.free_pool.push(reg);
            }
            Ok((reg, None))
        }
    }

    /// Peek at the top register without popping
    pub fn peek(&self) -> Result<u8, TranslateError> {
        self.stack.last().copied().ok_or(TranslateError::StackUnderflow)
    }

    /// Allocate a temporary register
    /// CRITICAL FIX: First tries to reuse a freed register, then allocates new
    /// SAFEGUARD: If all registers exhausted, returns SPILLED_REG and tracks spill slot
    pub fn alloc_temp(&mut self) -> Result<u8, TranslateError> {
        // 1. First, try to reuse a freed register from the pool
        if let Some(reg) = self.free_pool.pop() {
            return Ok(reg);
        }

        // 2. Try allocating a new register
        if self.next_temp < self.max_temp {
            let reg = self.next_temp;
            self.next_temp += 1;
            return Ok(reg);
        }

        // 3. SAFEGUARD: Spill to GPU memory instead of failing
        #[cfg(debug_assertions)]
        eprintln!("[SPILL] Register pressure exceeded (depth={}), spilling to GPU memory slot {}",
                  self.stack.len(), self.spill_count);

        // Track the spill slot address
        let spill_addr = self.spill_base + self.spill_count;
        self.spill_slots.push(spill_addr);
        self.spill_count += 1;

        Ok(SPILLED_REG)
    }

    /// Check if a register value indicates a spilled value
    pub fn is_spilled(reg: u8) -> bool {
        reg == SPILLED_REG
    }

    /// Get the spill address for the most recently allocated spill slot
    /// Call this immediately after alloc_temp() returns SPILLED_REG
    pub fn last_spill_addr(&self) -> Option<u32> {
        self.spill_slots.last().copied()
    }

    /// Get spill address by index (for tracking multiple spills)
    pub fn spill_addr_at(&self, idx: usize) -> Option<u32> {
        self.spill_slots.get(idx).copied()
    }

    /// Get total number of spills (for statistics/debugging)
    pub fn spill_count(&self) -> u32 {
        self.spill_count
    }

    /// Get allocation state for debugging
    pub fn debug_state(&self) -> (u8, u8, usize) {
        (self.next_temp, self.max_temp, self.stack.len())
    }

    /// Allocate a temp and push it onto the stack
    pub fn alloc_and_push(&mut self) -> Result<u8, TranslateError> {
        let reg = self.alloc_temp()?;
        self.push(reg);
        Ok(reg)
    }

    /// Get current stack depth
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Reset temp allocation (for new basic block)
    /// CRITICAL FIX: Also clears the free pool since we're starting fresh
    ///
    /// SAFEGUARD: If stack is non-empty, this indicates unexpected LLVM behavior
    /// (e.g., debug builds, manual WASM). We warn but preserve live values.
    pub fn reset_temps(&mut self) {
        // SAFEGUARD: Warn if stack is non-empty at block boundary
        // LLVM-optimized code should have empty operand stack at block boundaries
        // Non-empty stack suggests debug build or non-standard WASM
        #[cfg(debug_assertions)]
        if !self.stack.is_empty() {
            eprintln!(
                "[WARN] reset_temps called with non-empty stack: depth={} registers={:?}",
                self.stack.len(),
                self.stack
            );
            // SAFEGUARD: Don't reset next_temp below the highest live register
            // This prevents accidental reuse of live registers
            let max_live = self.stack.iter()
                .filter(|&&r| r >= 8 && r < 28)
                .max()
                .copied()
                .unwrap_or(7);
            if max_live >= 8 {
                self.next_temp = max_live + 1;
                self.free_pool.clear();
                return;
            }
        }

        // Normal case: stack is empty, safe to reset
        self.next_temp = 8;
        self.free_pool.clear();
    }

    /// Check if stack is empty (for assertions)
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Truncate stack to given depth (for block end)
    pub fn truncate(&mut self, depth: usize) {
        self.stack.truncate(depth);
    }

    /// Get scratch register (doesn't affect stack)
    pub fn scratch(&self, idx: usize) -> u8 {
        30 + (idx as u8 % 2)  // r30 or r31
    }

    /// Save current stack state (for branch handling in call_indirect)
    /// Returns a snapshot that can be restored later
    pub fn save_state(&self) -> StackState {
        StackState {
            stack: self.stack.clone(),
            next_temp: self.next_temp,
            free_pool: self.free_pool.clone(),
            spill_count: self.spill_count,
            spill_slots: self.spill_slots.clone(),
        }
    }

    /// Restore stack state from a previous save
    pub fn restore_state(&mut self, state: StackState) {
        self.stack = state.stack;
        self.next_temp = state.next_temp;
        self.free_pool = state.free_pool;
        self.spill_count = state.spill_count;
        self.spill_slots = state.spill_slots;
    }
}

/// Snapshot of stack state for branching code paths
#[derive(Clone)]
pub struct StackState {
    stack: Vec<u8>,
    next_temp: u8,
    free_pool: Vec<u8>,
    spill_count: u32,
    spill_slots: Vec<u32>,
}

impl Default for OperandStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Local variable mapping
///
/// Maps WASM locals to registers or spill slots
pub struct LocalMap {
    /// Local index -> register number
    /// First N locals map to r4-r7 (args), then r28-r29
    registers: Vec<u8>,
    /// Spill offset for overflow locals
    spill_base: u32,
}

impl LocalMap {
    /// Create local map for function parameters and locals
    pub fn new(param_count: u32, local_count: u32, spill_base: u32) -> Self {
        let total = param_count + local_count;
        let mut registers = Vec::with_capacity(total as usize);

        // First 4 locals go to r4-r7 (args)
        for i in 0..std::cmp::min(total, 4) {
            registers.push(4 + i as u8);
        }

        // Next locals go to r28-r29 or spill
        for i in 4..total {
            if i < 6 {
                registers.push(28 + (i - 4) as u8);
            } else {
                // Spilled - use placeholder, will load/store from memory
                registers.push(0xFF);
            }
        }

        Self {
            registers,
            spill_base,
        }
    }

    /// Get register for local, or None if spilled
    pub fn get(&self, local_idx: u32) -> Option<u8> {
        let reg = *self.registers.get(local_idx as usize)?;
        if reg == 0xFF {
            None
        } else {
            Some(reg)
        }
    }

    /// Get spill address for local (if spilled)
    pub fn spill_addr(&self, local_idx: u32) -> u32 {
        self.spill_base + (local_idx.saturating_sub(6))
    }

    /// Check if local is in a register
    pub fn is_register(&self, local_idx: u32) -> bool {
        self.get(local_idx).is_some()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS FOR SAFEGUARDS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut stack = OperandStack::new();
        let r1 = stack.alloc_and_push().unwrap();
        assert_eq!(r1, 8); // First temp register
        let r2 = stack.alloc_and_push().unwrap();
        assert_eq!(r2, 9);
        assert_eq!(stack.depth(), 2);
    }

    #[test]
    fn test_free_pool_recycling() {
        let mut stack = OperandStack::new();

        // Allocate and push 3 registers
        let r1 = stack.alloc_and_push().unwrap();
        let r2 = stack.alloc_and_push().unwrap();
        let r3 = stack.alloc_and_push().unwrap();

        // Pop them (returns to free pool)
        stack.pop().unwrap();
        stack.pop().unwrap();
        stack.pop().unwrap();

        // Next allocations should reuse from free pool
        let r4 = stack.alloc_and_push().unwrap();
        let r5 = stack.alloc_and_push().unwrap();

        // Should be recycled registers (order may vary due to LIFO)
        assert!(r4 == r1 || r4 == r2 || r4 == r3);
        assert!(r5 == r1 || r5 == r2 || r5 == r3);
    }

    #[test]
    fn test_spill_fallback_under_pressure() {
        let mut stack = OperandStack::new();

        // Exhaust all 20 temp registers (r8-r27)
        for i in 0..20 {
            let reg = stack.alloc_and_push().unwrap();
            assert_eq!(reg, 8 + i as u8, "Register {} should be r{}", i, 8 + i);
        }

        // 21st allocation should spill, not fail
        let result = stack.alloc_temp();
        assert!(result.is_ok(), "Should spill instead of failing");

        let reg = result.unwrap();
        assert!(OperandStack::is_spilled(reg), "Should be marked as spilled");
        assert_eq!(reg, SPILLED_REG);

        // Should have tracked the spill
        assert_eq!(stack.spill_count(), 1);
        assert!(stack.last_spill_addr().is_some());
    }

    #[test]
    fn test_multiple_spills() {
        let mut stack = OperandStack::new();

        // Exhaust registers
        for _ in 0..20 {
            stack.alloc_and_push().unwrap();
        }

        // Spill 5 more
        for i in 0..5 {
            let reg = stack.alloc_and_push().unwrap();
            assert!(OperandStack::is_spilled(reg));
            assert_eq!(stack.spill_count(), (i + 1) as u32);
        }

        assert_eq!(stack.spill_count(), 5);
        assert_eq!(stack.depth(), 25); // 20 in registers + 5 spilled
    }

    #[test]
    fn test_pop_with_spill_info() {
        let mut stack = OperandStack::new();

        // Exhaust registers and spill one
        for _ in 0..20 {
            stack.alloc_and_push().unwrap();
        }
        stack.alloc_and_push().unwrap(); // This spills

        // Pop the spilled value
        let (reg, spill_addr) = stack.pop_with_spill_info().unwrap();
        assert!(OperandStack::is_spilled(reg));
        assert!(spill_addr.is_some());

        // Pop a regular register
        let (reg2, spill_addr2) = stack.pop_with_spill_info().unwrap();
        assert!(!OperandStack::is_spilled(reg2));
        assert!(spill_addr2.is_none());
    }

    #[test]
    fn test_reset_temps_with_empty_stack() {
        let mut stack = OperandStack::new();

        // Allocate some registers
        stack.alloc_and_push().unwrap();
        stack.alloc_and_push().unwrap();

        // Pop them all
        stack.pop().unwrap();
        stack.pop().unwrap();

        // Reset should work normally
        stack.reset_temps();

        // Next allocation should start at r8 again
        let reg = stack.alloc_and_push().unwrap();
        assert_eq!(reg, 8);
    }

    #[test]
    fn test_reset_temps_preserves_live_values() {
        let mut stack = OperandStack::new();

        // Push values that stay on stack
        let r1 = stack.alloc_and_push().unwrap();
        let r2 = stack.alloc_and_push().unwrap();

        // Reset with non-empty stack (simulates debug build behavior)
        stack.reset_temps();

        // Stack should still have the values
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.peek().unwrap(), r2);

        // New allocation should not clobber live registers
        let r3 = stack.alloc_and_push().unwrap();
        assert!(r3 > r2, "New register {} should be higher than live register {}", r3, r2);
    }

    #[test]
    fn test_is_empty() {
        let mut stack = OperandStack::new();
        assert!(stack.is_empty());

        stack.alloc_and_push().unwrap();
        assert!(!stack.is_empty());

        stack.pop().unwrap();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_custom_spill_base() {
        let stack = OperandStack::with_spill_base(0x2000);
        assert_eq!(stack.spill_count(), 0);
    }

    #[test]
    fn test_scratch_registers() {
        let stack = OperandStack::new();
        assert_eq!(stack.scratch(0), 30);
        assert_eq!(stack.scratch(1), 31);
        assert_eq!(stack.scratch(2), 30); // Wraps
    }
}
