//! WASM instruction translator
//!
//! THE GPU IS THE COMPUTER.
//! Translates WASM instructions to GPU bytecode.

use crate::emit::Emitter;
use crate::stack::{LocalMap, OperandStack};
use crate::types::*;
use wasmparser::Operator;
use std::collections::HashSet;

/// Translation context for a single function
pub struct TranslationContext<'a> {
    pub emit: Emitter,
    pub stack: OperandStack,
    pub locals: LocalMap,
    pub block_stack: Vec<BlockContext>,
    pub config: &'a TranslatorConfig,
    /// Label for function epilogue (return value handling)
    pub epilogue_label: usize,
    /// Module reference for function inlining (Phase 5)
    pub module: Option<&'a WasmModule>,
    /// Parsed function bodies for inlining (Phase 5)
    pub function_ops: Option<&'a Vec<Vec<Operator<'a>>>>,
    /// Local counts per defined function (for inlining)
    pub function_local_counts: Option<&'a Vec<u32>>,
    /// Call stack for recursion detection (Phase 5)
    pub call_stack: HashSet<u32>,
    /// Counter for unique spill bases when inlining functions
    inline_spill_counter: u32,
}

impl<'a> TranslationContext<'a> {
    pub fn new(
        param_count: u32,
        local_count: u32,
        config: &'a TranslatorConfig,
    ) -> Self {
        let mut emit = Emitter::new();
        let epilogue_label = emit.new_label();
        Self {
            emit,
            stack: OperandStack::new(),
            locals: LocalMap::new(param_count, local_count, config.globals_base + 256),
            block_stack: Vec::new(),
            config,
            epilogue_label,
            module: None,
            function_ops: None,
            function_local_counts: None,
            call_stack: HashSet::new(),
            inline_spill_counter: 1,  // Start at 1, entry function uses 0
        }
    }

    /// Set module and function ops for function call support (Phase 5)
    pub fn with_module(
        mut self,
        module: &'a WasmModule,
        function_ops: &'a Vec<Vec<Operator<'a>>>,
        function_local_counts: &'a Vec<u32>,
    ) -> Self {
        self.module = Some(module);
        self.function_ops = Some(function_ops);
        self.function_local_counts = Some(function_local_counts);
        self
    }

    /// Get branch target label for relative depth
    fn get_branch_target(&self, relative_depth: u32) -> usize {
        let idx = self.block_stack.len() - 1 - relative_depth as usize;
        let block = &self.block_stack[idx];
        // For loops, branch to start; for blocks/if, branch to end
        if block.kind == BlockKind::Loop {
            block.start_label.unwrap()
        } else {
            block.end_label
        }
    }

    /// Translate a single WASM operator
    pub fn translate_operator(&mut self, op: &Operator) -> Result<(), TranslateError> {
        match op {
            // ═══════════════════════════════════════════════════════════════
            // CONTROL FLOW
            // ═══════════════════════════════════════════════════════════════

            Operator::Unreachable => {
                self.emit.halt();
            }

            Operator::Nop => {
                self.emit.nop();
            }

            Operator::Block { .. } => {
                let end_label = self.emit.new_label();
                // CRITICAL FIX: Reset temps at block entry for register recycling
                self.stack.reset_temps();
                self.block_stack.push(BlockContext {
                    kind: BlockKind::Block,
                    start_label: None,
                    else_label: None,
                    end_label,
                    stack_depth: self.stack.depth(),
                    result_count: 0,  // Regular blocks don't use result_count
                });
            }

            Operator::Loop { .. } => {
                let start_label = self.emit.new_label();
                let end_label = self.emit.new_label();
                self.emit.define_label(start_label);
                // CRITICAL FIX: Reset temps at loop entry for register recycling
                // Loop bodies may execute many times - must reuse registers
                self.stack.reset_temps();
                self.block_stack.push(BlockContext {
                    kind: BlockKind::Loop,
                    start_label: Some(start_label),
                    else_label: None,
                    end_label,
                    stack_depth: self.stack.depth(),
                    result_count: 0,  // Loops don't use result_count
                });
            }

            Operator::If { .. } => {
                let cond = self.stack.pop()?;
                let else_label = self.emit.new_label();
                let end_label = self.emit.new_label();
                self.emit.jz_label(cond, else_label);
                // CRITICAL FIX: Reset temps at if entry for register recycling
                self.stack.reset_temps();
                self.block_stack.push(BlockContext {
                    kind: BlockKind::If,
                    start_label: None,
                    else_label: Some(else_label),
                    end_label,
                    stack_depth: self.stack.depth(),
                    result_count: 0,  // If blocks don't use result_count
                });
            }

            Operator::Else => {
                let ctx = self.block_stack.last_mut()
                    .ok_or(TranslateError::Invalid("else without if".into()))?;
                let else_label = ctx.else_label.take()
                    .ok_or(TranslateError::Invalid("else without if".into()))?;
                self.emit.jmp_label(ctx.end_label);
                self.emit.define_label(else_label);
                // CRITICAL FIX: Reset temps at else entry for register recycling
                self.stack.reset_temps();
            }

            Operator::End => {
                if let Some(ctx) = self.block_stack.pop() {
                    // If there was an else_label that wasn't used, define it now
                    if let Some(else_label) = ctx.else_label {
                        self.emit.define_label(else_label);
                    }
                    self.emit.define_label(ctx.end_label);

                    // CRITICAL FIX: Recycle registers at block boundaries
                    // Reset temp allocation to avoid OutOfRegisters error
                    // Values consumed in the block are no longer live, so we can reuse registers
                    self.stack.reset_temps();
                }
                // Note: End for function is handled separately
            }

            Operator::Br { relative_depth } => {
                let target = self.get_branch_target(*relative_depth);
                self.emit.jmp_label(target);
            }

            Operator::BrIf { relative_depth } => {
                let cond = self.stack.pop()?;
                let target = self.get_branch_target(*relative_depth);
                self.emit.jnz_label(cond, target);
            }

            Operator::Return => {
                // Check if we're inside an inlined function
                // If so, jump to the inline function's end label, not the main epilogue
                let inline_block = self.block_stack.iter().rev()
                    .find(|b| b.kind == BlockKind::InlineFunction);

                if let Some(block) = inline_block {
                    // Inside an inlined function - leave return value on stack
                    // The value will be used by the caller after the inline block ends
                    let end_label = block.end_label;
                    self.emit.jmp_label(end_label);
                } else {
                    // Top-level function - move return value to r4 and jump to epilogue
                    if self.stack.depth() > 0 {
                        let val = self.stack.pop()?;
                        self.emit.mov(4, val);
                    }
                    self.emit.jmp_label(self.epilogue_label);
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // LOCAL/GLOBAL VARIABLES
            // ═══════════════════════════════════════════════════════════════

            Operator::LocalGet { local_index } => {
                let dst = self.stack.alloc_and_push()?;
                if let Some(src) = self.locals.get(*local_index) {
                    self.emit.mov(dst, src);
                } else {
                    // Spilled local - load from memory
                    // Issue #213 fix: Use loadi (float VALUE) for addresses
                    let addr = self.locals.spill_addr(*local_index);
                    self.emit.loadi(30, addr as f32);
                    self.emit.ld(dst, 30, 0.0);
                }
            }

            Operator::LocalSet { local_index } => {
                let val = self.stack.pop()?;
                if let Some(dst) = self.locals.get(*local_index) {
                    self.emit.mov(dst, val);
                } else {
                    // Spilled - store to memory
                    // Issue #213 fix: Use loadi (float VALUE) for addresses
                    let addr = self.locals.spill_addr(*local_index);
                    self.emit.loadi(30, addr as f32);
                    self.emit.st(30, val, 0.0);
                }
            }

            Operator::LocalTee { local_index } => {
                let val = self.stack.peek()?;
                if let Some(dst) = self.locals.get(*local_index) {
                    self.emit.mov(dst, val);
                } else {
                    // Issue #213 fix: Use loadi (float VALUE) for addresses
                    let addr = self.locals.spill_addr(*local_index);
                    self.emit.loadi(30, addr as f32);
                    self.emit.st(30, val, 0.0);
                }
            }

            Operator::GlobalGet { global_index } => {
                let dst = self.stack.alloc_and_push()?;
                // Issue #213 fix: Use loadi (float VALUE) for addresses
                let addr = self.config.globals_base + *global_index;
                self.emit.loadi(30, addr as f32);
                self.emit.ld(dst, 30, 0.0);
            }

            Operator::GlobalSet { global_index } => {
                let val = self.stack.pop()?;
                // Issue #213 fix: Use loadi (float VALUE) for addresses
                let addr = self.config.globals_base + *global_index;
                self.emit.loadi(30, addr as f32);
                self.emit.st(30, val, 0.0);
            }

            // ═══════════════════════════════════════════════════════════════
            // MEMORY OPERATIONS
            // ═══════════════════════════════════════════════════════════════

            Operator::I32Load { memarg } => {
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // LD4 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.ld4(dst, addr, 0.0);
            }

            Operator::I32Store { memarg } => {
                let val = self.stack.pop()?;
                let addr = self.stack.pop()?;
                // ST4 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.st4(addr, val, 0.0);
            }

            Operator::F32Load { memarg } => {
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // LD4 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.ld4(dst, addr, 0.0);
            }

            Operator::F32Store { memarg } => {
                let val = self.stack.pop()?;
                let addr = self.stack.pop()?;
                // ST4 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.st4(addr, val, 0.0);
            }

            // ═══════════════════════════════════════════════════════════════
            // 64-BIT MEMORY OPERATIONS (Issue #188)
            // THE GPU IS THE COMPUTER - 64-bit values span two 32-bit words
            // ═══════════════════════════════════════════════════════════════

            Operator::I64Load { memarg } => {
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate word address for low 32 bits
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.loadi_uint(30, 2);
                self.emit.shr_u(addr, addr, 30);

                // Load low 32 bits into dst.x
                self.emit.ld(dst, addr, 0.0);

                // Load high 32 bits into dst.y (next word)
                self.emit.loadi_uint(30, 1);
                self.emit.int_add(addr, addr, 30);  // addr + 1
                self.emit.ld(31, addr, 0.0);  // Load into r31
                // Copy r31.x to dst.y using sety
                // Actually we need to read from r31 and set dst.y
                // For now, use a simpler approach: store/load pair
                // Or use the proper approach with memory
                // Simplest: directly set dst.y from loaded value
                self.emit.sety(dst, 0.0);  // Clear first (will be overwritten)
                // We need to copy r31.x into dst.y - this requires a special operation
                // Let's use a workaround: reconstruct the 64-bit value
                // r31 has the high bits in .x, dst has low bits in .x
                // We need to combine them into dst.xy
                // Use INT64_EXTEND_U to get low part as 64-bit, then OR with shifted high
                self.emit.int64_extend_u(dst, dst);  // dst.xy = zext(dst.x)
                self.emit.int64_extend_u(31, 31);     // r31.xy = zext(r31.x)
                // Shift high part left by 32
                self.emit.loadi_uint(29, 32);
                self.emit.int64_shl(31, 31, 29);     // r31.xy <<= 32
                self.emit.int64_or(dst, dst, 31);    // dst.xy |= r31.xy
            }

            Operator::I64Store { memarg } => {
                let val = self.stack.pop()?;
                let addr = self.stack.pop()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate word address
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.loadi_uint(30, 2);
                self.emit.shr_u(addr, addr, 30);

                // Extract and store low 32 bits (val.x)
                self.emit.int64_wrap(31, val);      // r31.x = low 32 bits
                self.emit.st(addr, 31, 0.0);        // Store low

                // Extract and store high 32 bits
                self.emit.loadi_uint(30, 32);
                self.emit.int64_shr_u(31, val, 30);  // r31.xy = val >> 32
                self.emit.int64_wrap(31, 31);        // r31.x = high 32 bits

                // Store at next word
                self.emit.loadi_uint(30, 1);
                self.emit.int_add(addr, addr, 30);   // addr + 1
                self.emit.st(addr, 31, 0.0);         // Store high
            }

            Operator::I64Load32U { memarg } => {
                // Load 32 bits from memory and zero-extend to 64-bit
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate word address
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.loadi_uint(30, 2);
                self.emit.shr_u(addr, addr, 30);

                // Load 32-bit value
                self.emit.ld(dst, addr, 0.0);

                // Zero-extend to 64-bit
                self.emit.int64_extend_u(dst, dst);
            }

            Operator::I64Load32S { memarg } => {
                // Load 32 bits from memory and sign-extend to 64-bit
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate word address
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.loadi_uint(30, 2);
                self.emit.shr_u(addr, addr, 30);

                // Load 32-bit value
                self.emit.ld(dst, addr, 0.0);

                // Sign-extend to 64-bit
                self.emit.int64_extend_s(dst, dst);
            }

            Operator::I64Load16U { memarg } | Operator::I64Load16S { memarg } => {
                // Load 16 bits from memory and extend to 64-bit
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate byte address
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);

                // Get halfword offset within word (addr & 2) >> 1
                self.emit.loadi_uint(30, 2);
                self.emit.bit_and(31, addr, 30);  // r31 = 0 or 2

                // Get word address (addr >> 2)
                self.emit.shr_u(addr, addr, 30);

                // Load the word
                self.emit.ld(dst, addr, 0.0);

                // Shift right to get the halfword (halfword_offset * 8)
                self.emit.loadi_uint(30, 3);
                self.emit.shl(31, 31, 30);  // r31 = 0 or 16
                self.emit.shr_u(dst, dst, 31);

                // Mask to get just 16 bits
                self.emit.loadi_uint(30, 0xFFFF);
                self.emit.bit_and(dst, dst, 30);

                // Sign extend if signed
                if matches!(op, Operator::I64Load16S { .. }) {
                    // Sign extend from 16 to 32 bits first
                    self.emit.loadi_uint(30, 0x8000);
                    self.emit.bit_and(31, dst, 30);
                    let no_sign = self.emit.new_label();
                    let done = self.emit.new_label();
                    self.emit.jz_label(31, no_sign);
                    self.emit.loadi_int(30, -65536);  // 0xFFFF0000
                    self.emit.bit_or(dst, dst, 30);
                    self.emit.jmp_label(done);
                    self.emit.define_label(no_sign);
                    self.emit.define_label(done);
                    // Then sign-extend to 64-bit
                    self.emit.int64_extend_s(dst, dst);
                } else {
                    // Zero-extend to 64-bit
                    self.emit.int64_extend_u(dst, dst);
                }
            }

            Operator::I64Load8U { memarg } | Operator::I64Load8S { memarg } => {
                // Load 8 bits from memory and extend to 64-bit
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let base = self.config.memory_base + memarg.offset as u32;

                // Calculate byte address
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);

                // Get byte offset within word (addr & 3)
                self.emit.loadi_uint(30, 3);
                self.emit.bit_and(31, addr, 30);

                // Get word address (addr >> 2)
                self.emit.loadi_uint(30, 2);
                self.emit.shr_u(addr, addr, 30);

                // Load the word
                self.emit.ld(dst, addr, 0.0);

                // Shift right to get the byte (byte_offset * 8)
                self.emit.loadi_uint(30, 3);
                self.emit.shl(31, 31, 30);
                self.emit.shr_u(dst, dst, 31);

                // Mask to get just the byte
                self.emit.loadi_uint(30, 0xFF);
                self.emit.bit_and(dst, dst, 30);

                // Sign extend if signed
                if matches!(op, Operator::I64Load8S { .. }) {
                    self.emit.loadi_uint(30, 0x80);
                    self.emit.bit_and(31, dst, 30);
                    let no_sign = self.emit.new_label();
                    let done = self.emit.new_label();
                    self.emit.jz_label(31, no_sign);
                    self.emit.loadi_int(30, -256);
                    self.emit.bit_or(dst, dst, 30);
                    self.emit.jmp_label(done);
                    self.emit.define_label(no_sign);
                    self.emit.define_label(done);
                    self.emit.int64_extend_s(dst, dst);
                } else {
                    self.emit.int64_extend_u(dst, dst);
                }
            }

            Operator::MemorySize { .. } => {
                // GPU-native memory size - Issue #210
                let dst = self.stack.alloc_and_push()?;
                self.emit.memory_size(dst);
            }

            Operator::MemoryGrow { .. } => {
                // GPU-native memory grow - Issue #210
                // Returns old size on success, -1 on failure
                let delta = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.memory_grow(dst, delta);
                // delta register already returned to free pool by pop()
            }

            Operator::MemoryCopy { .. } => {
                // memory.copy(dst, src, len) - bulk copy bytes
                // Issue #260: Must handle overlapping regions with memmove semantics
                // If src < dst && src + len > dst, copy backwards
                let len = self.stack.pop()?;
                let src = self.stack.pop()?;
                let dst = self.stack.pop()?;

                // Add memory base (byte offset) to addresses
                let base = self.config.memory_base;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(dst, dst, 30);  // dst += base
                self.emit.int_add(src, src, 30);  // src += base

                // Labels for forward vs backward copy
                let forward_copy = self.emit.new_label();
                let backward_copy = self.emit.new_label();
                let copy_done = self.emit.new_label();

                // Check for backward copy needed: src < dst && src + len > dst
                // r28 = src < dst
                self.emit.int_lt_u(28, src, dst);
                self.emit.loadi_uint(30, 0);
                self.emit.int_eq(31, 28, 30);  // r31 = !(src < dst)
                self.emit.jnz_label(31, forward_copy);

                // r28 = src + len
                self.emit.int_add(28, src, len);
                // r31 = (src + len) > dst  (overlap condition)
                // a > b ≡ b < a, so (r28 > dst) ≡ (dst < r28)
                self.emit.int_lt_u(31, dst, 28);
                self.emit.loadi_uint(30, 0);
                self.emit.int_eq(28, 31, 30);  // r28 = !((src+len) > dst)
                self.emit.jnz_label(28, forward_copy);

                // ═══════════════════════════════════════════════════════════════
                // BACKWARD COPY (overlap case): copy from end to start
                // ═══════════════════════════════════════════════════════════════
                self.emit.define_label(backward_copy);

                // Move pointers to end: src += len, dst += len
                self.emit.int_add(src, src, len);
                self.emit.int_add(dst, dst, len);

                // r29 = bytes remaining
                self.emit.mov(29, len);

                // Backward byte loop (simpler than word copy for backward)
                let back_byte_loop = self.emit.new_label();
                let back_done = self.emit.new_label();
                self.emit.define_label(back_byte_loop);
                self.emit.loadi_uint(30, 0);
                self.emit.int_eq(31, 29, 30);  // r31 = (remaining == 0)
                self.emit.jnz_label(31, back_done);

                // Decrement pointers first (we're at end)
                self.emit.loadi_uint(30, 1);
                self.emit.int_sub(src, src, 30);
                self.emit.int_sub(dst, dst, 30);
                self.emit.int_sub(29, 29, 30);

                // Copy byte
                self.emit.ld1(28, src, 0.0);
                self.emit.st1(dst, 28, 0.0);

                self.emit.jmp_label(back_byte_loop);
                self.emit.define_label(back_done);
                self.emit.jmp_label(copy_done);

                // ═══════════════════════════════════════════════════════════════
                // FORWARD COPY (non-overlapping or safe overlap)
                // ═══════════════════════════════════════════════════════════════
                self.emit.define_label(forward_copy);

                let loop_start = self.emit.new_label();
                let loop_end = self.emit.new_label();

                // r29 = bytes remaining
                self.emit.mov(29, len);

                // Loop: while bytes >= 4
                self.emit.define_label(loop_start);
                self.emit.loadi_uint(30, 4);
                self.emit.int_lt_u(31, 29, 30);  // r31 = (remaining < 4)
                self.emit.jnz_label(31, loop_end);

                // Load word from src using LD4 (byte address)
                self.emit.ld4(28, src, 0.0);   // r28 = word from src

                // Store word to dst using ST4 (byte address)
                self.emit.st4(dst, 28, 0.0);   // store word

                // Advance pointers by 4 bytes
                self.emit.loadi_uint(30, 4);
                self.emit.int_add(src, src, 30);
                self.emit.int_add(dst, dst, 30);
                self.emit.int_sub(29, 29, 30);
                self.emit.jmp_label(loop_start);

                self.emit.define_label(loop_end);

                // Handle remaining bytes (0-3) byte by byte
                let byte_loop = self.emit.new_label();
                let byte_done = self.emit.new_label();
                self.emit.define_label(byte_loop);
                self.emit.loadi_uint(30, 0);
                self.emit.int_eq(31, 29, 30);  // r31 = (remaining == 0)
                self.emit.jnz_label(31, byte_done);
                self.emit.ld1(28, src, 0.0);   // load byte
                self.emit.st1(dst, 28, 0.0);   // store byte
                self.emit.loadi_uint(30, 1);
                self.emit.int_add(src, src, 30);
                self.emit.int_add(dst, dst, 30);
                self.emit.int_sub(29, 29, 30);
                self.emit.jmp_label(byte_loop);
                self.emit.define_label(byte_done);

                self.emit.define_label(copy_done);
            }

            Operator::MemoryFill { .. } => {
                // memory.fill(dst, val, len) - fill bytes with value
                let len = self.stack.pop()?;
                let val = self.stack.pop()?;
                let dst = self.stack.pop()?;

                // Add memory base (byte offset)
                let base = self.config.memory_base;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(dst, dst, 30);

                // Replicate byte value to all 4 bytes of a word
                // val = val | (val << 8) | (val << 16) | (val << 24)
                self.emit.loadi_uint(30, 0xFF);
                self.emit.bit_and(val, val, 30);  // val &= 0xFF
                self.emit.mov(31, val);
                self.emit.loadi_uint(30, 8);
                self.emit.shl(31, 31, 30);
                self.emit.bit_or(val, val, 31);  // val |= val << 8
                self.emit.mov(31, val);
                self.emit.loadi_uint(30, 16);
                self.emit.shl(31, 31, 30);
                self.emit.bit_or(val, val, 31);  // val |= val << 16

                // Fill loop using ST4 (word at a time)
                let loop_start = self.emit.new_label();
                let loop_end = self.emit.new_label();
                self.emit.mov(29, len);

                self.emit.define_label(loop_start);
                self.emit.loadi_uint(30, 4);
                self.emit.int_lt_u(31, 29, 30);
                self.emit.jnz_label(31, loop_end);

                self.emit.st4(dst, val, 0.0);  // ST4 takes byte address

                self.emit.loadi_uint(30, 4);
                self.emit.int_add(dst, dst, 30);
                self.emit.int_sub(29, 29, 30);
                self.emit.jmp_label(loop_start);

                self.emit.define_label(loop_end);

                // Handle remaining bytes (0-3) byte by byte
                let byte_loop = self.emit.new_label();
                let byte_done = self.emit.new_label();
                self.emit.define_label(byte_loop);
                self.emit.loadi_uint(30, 0);
                self.emit.int_eq(31, 29, 30);  // r31 = (remaining == 0)
                self.emit.jnz_label(31, byte_done);
                // Use just the low byte of val
                self.emit.loadi_uint(30, 0xFF);
                self.emit.bit_and(28, val, 30);
                self.emit.st1(dst, 28, 0.0);   // store byte
                self.emit.loadi_uint(30, 1);
                self.emit.int_add(dst, dst, 30);
                self.emit.int_sub(29, 29, 30);
                self.emit.jmp_label(byte_loop);
                self.emit.define_label(byte_done);
            }

            // ═══════════════════════════════════════════════════════════════
            // BYTE-LEVEL MEMORY OPERATIONS
            // GPU memory is word-addressed, so we pack/unpack bytes
            // ═══════════════════════════════════════════════════════════════

            Operator::I32Load8S { memarg } | Operator::I32Load8U { memarg } => {
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // LD1 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.ld1(dst, addr, 0.0);

                // Sign extend if signed
                if matches!(op, Operator::I32Load8S { .. }) {
                    // Sign extend: if bit 7 is set, OR with 0xFFFFFF00
                    self.emit.loadi_uint(30, 0x80);
                    self.emit.bit_and(31, dst, 30);  // Check bit 7
                    let no_sign = self.emit.new_label();
                    let done = self.emit.new_label();
                    self.emit.jz_label(31, no_sign);
                    self.emit.loadi_int(30, -256);  // 0xFFFFFF00
                    self.emit.bit_or(dst, dst, 30);
                    self.emit.jmp_label(done);
                    self.emit.define_label(no_sign);
                    self.emit.define_label(done);
                }
            }

            Operator::I32Store8 { memarg } => {
                let val = self.stack.pop()?;
                let addr = self.stack.pop()?;
                // ST1 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.st1(addr, val, 0.0);
            }

            Operator::I32Load16S { memarg } | Operator::I32Load16U { memarg } => {
                let addr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // LD2 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.ld2(dst, addr, 0.0);

                // Sign extend if I32Load16S
                if matches!(op, Operator::I32Load16S { .. }) {
                    // Sign extend: if bit 15 is set, OR with 0xFFFF0000
                    self.emit.loadi_uint(30, 0x8000);
                    self.emit.bit_and(31, dst, 30);  // Check bit 15
                    let no_sign = self.emit.new_label();
                    let done = self.emit.new_label();
                    self.emit.jz_label(31, no_sign);
                    self.emit.loadi_int(30, -65536);  // 0xFFFF0000
                    self.emit.bit_or(dst, dst, 30);
                    self.emit.jmp_label(done);
                    self.emit.define_label(no_sign);
                    self.emit.define_label(done);
                }
            }

            Operator::I32Store16 { memarg } => {
                let val = self.stack.pop()?;
                let addr = self.stack.pop()?;
                // ST2 takes byte address directly
                let base = self.config.memory_base + memarg.offset as u32;
                self.emit.loadi_uint(30, base);
                self.emit.int_add(addr, addr, 30);
                self.emit.st2(addr, val, 0.0);
            }

            // ═══════════════════════════════════════════════════════════════
            // I32 CONSTANTS
            // ═══════════════════════════════════════════════════════════════

            Operator::I32Const { value } => {
                let dst = self.stack.alloc_and_push()?;
                self.emit.loadi_int(dst, *value);
            }

            // ═══════════════════════════════════════════════════════════════
            // I32 ARITHMETIC
            // ═══════════════════════════════════════════════════════════════

            Operator::I32Add => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_add(dst, a, b);
            }

            Operator::I32Sub => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_sub(dst, a, b);
            }

            Operator::I32Mul => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_mul(dst, a, b);
            }

            Operator::I32DivS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_div_s(dst, a, b);
            }

            Operator::I32DivU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_div_u(dst, a, b);
            }

            Operator::I32RemS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_rem_s(dst, a, b);
            }

            Operator::I32RemU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_rem_u(dst, a, b);
            }

            // ═══════════════════════════════════════════════════════════════
            // I32 BITWISE
            // ═══════════════════════════════════════════════════════════════

            Operator::I32And => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.bit_and(dst, a, b);
            }

            Operator::I32Or => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.bit_or(dst, a, b);
            }

            Operator::I32Xor => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.bit_xor(dst, a, b);
            }

            Operator::I32Shl => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.shl(dst, a, b);
            }

            Operator::I32ShrS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.shr_s(dst, a, b);
            }

            Operator::I32ShrU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.shr_u(dst, a, b);
            }

            Operator::I32Rotl => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.rotl(dst, a, b);
            }

            Operator::I32Rotr => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.rotr(dst, a, b);
            }

            Operator::I32Clz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.clz(dst, a);
            }

            Operator::I32Ctz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.ctz(dst, a);
            }

            Operator::I32Popcnt => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.popcnt(dst, a);
            }

            // ═══════════════════════════════════════════════════════════════
            // I32 COMPARISON
            // ═══════════════════════════════════════════════════════════════

            Operator::I32Eqz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.loadi_int(30, 0);
                self.emit.int_eq(dst, a, 30);
            }

            Operator::I32Eq => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_eq(dst, a, b);
            }

            Operator::I32Ne => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_ne(dst, a, b);
            }

            Operator::I32LtS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_lt_s(dst, a, b);
            }

            Operator::I32LtU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_lt_u(dst, a, b);
            }

            Operator::I32LeS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_le_s(dst, a, b);
            }

            Operator::I32LeU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_le_u(dst, a, b);
            }

            Operator::I32GtS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // gt = lt with swapped operands
                self.emit.int_lt_s(dst, b, a);
            }

            Operator::I32GtU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_lt_u(dst, b, a);
            }

            Operator::I32GeS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // ge = le with swapped operands
                self.emit.int_le_s(dst, b, a);
            }

            Operator::I32GeU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_le_u(dst, b, a);
            }

            // ═══════════════════════════════════════════════════════════════
            // F32 CONSTANTS AND ARITHMETIC
            // ═══════════════════════════════════════════════════════════════

            Operator::F32Const { value } => {
                let dst = self.stack.alloc_and_push()?;
                // Convert bits back to f32 properly (not cast, which converts the integer value)
                self.emit.loadi(dst, f32::from_bits(value.bits()));
            }

            Operator::F32Add => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.add(dst, a, b);
            }

            Operator::F32Sub => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.sub(dst, a, b);
            }

            Operator::F32Mul => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.mul(dst, a, b);
            }

            Operator::F32Div => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.div(dst, a, b);
            }

            Operator::F32Neg => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.fneg(dst, a);
            }

            Operator::F32Abs => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.abs(dst, a);
            }

            Operator::F32Ceil => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.ceil(dst, a);
            }

            Operator::F32Floor => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.floor(dst, a);
            }

            Operator::F32Trunc => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.trunc(dst, a);
            }

            Operator::F32Nearest => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.nearest(dst, a);
            }

            Operator::F32Sqrt => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.sqrt(dst, a);
            }

            Operator::F32Copysign => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.copysign(dst, a, b);
            }

            Operator::F32Min => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.fmin(dst, a, b);
            }

            Operator::F32Max => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.fmax(dst, a, b);
            }

            // ═══════════════════════════════════════════════════════════════
            // F32 COMPARISON
            // ═══════════════════════════════════════════════════════════════

            Operator::F32Eq => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.eq(dst, a, b);
            }

            Operator::F32Lt => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.lt(dst, a, b);
            }

            Operator::F32Gt => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.gt(dst, a, b);
            }

            Operator::F32Le => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.le(dst, a, b);
            }

            Operator::F32Ge => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.ge(dst, a, b);
            }

            Operator::F32Ne => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.ne(dst, a, b);
            }

            // ═══════════════════════════════════════════════════════════════
            // CONVERSIONS
            // ═══════════════════════════════════════════════════════════════

            Operator::I32TruncF32S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f_to_int(dst, a);
            }

            Operator::I32TruncF32U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f_to_uint(dst, a);
            }

            // Saturating truncation - same as regular truncation for our GPU model
            // (GPU hardware handles saturation automatically)
            Operator::I32TruncSatF32S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f_to_int(dst, a);
            }

            Operator::I32TruncSatF32U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f_to_uint(dst, a);
            }

            Operator::I32TruncSatF64S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i32_s(dst, a);
            }

            Operator::I32TruncSatF64U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i32_u(dst, a);
            }

            Operator::I64TruncSatF32S | Operator::I64TruncSatF64S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i64_s(dst, a);  // Use f64 path for i64 result
            }

            Operator::I64TruncSatF32U | Operator::I64TruncSatF64U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i64_u(dst, a);  // Use f64 path for i64 result
            }

            Operator::F32ConvertI32S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int_to_f(dst, a);
            }

            Operator::F32ConvertI32U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.uint_to_f(dst, a);
            }

            Operator::I32ReinterpretF32 | Operator::F32ReinterpretI32 => {
                // No-op for register values: bits stay the same in our register model
                // But we MUST maintain stack semantics (pop and push) - Issue #247 fix
                let val = self.stack.pop()?;
                self.stack.push(val);
            }

            // ═══════════════════════════════════════════════════════════════
            // STACK MANIPULATION
            // ═══════════════════════════════════════════════════════════════

            Operator::Drop => {
                self.stack.pop()?;
            }

            Operator::Select => {
                let cond = self.stack.pop()?;
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // if cond != 0: dst = val1; else: dst = val2
                // Simple: always use val1 if cond, else val2
                let skip = self.emit.new_label();
                let end = self.emit.new_label();
                self.emit.jz_label(cond, skip);
                self.emit.mov(dst, val1);
                self.emit.jmp_label(end);
                self.emit.define_label(skip);
                self.emit.mov(dst, val2);
                self.emit.define_label(end);
            }

            // ═══════════════════════════════════════════════════════════════
            // I64 OPERATIONS (Issue #188)
            // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64
            // 64-bit values use two registers: even for low bits, even+1 for high bits
            // ═══════════════════════════════════════════════════════════════

            Operator::I64Const { value } => {
                let dst = self.stack.alloc_and_push()?;
                // Load 64-bit constant using the xy components of the register
                // Low 32 bits go in .x, high 32 bits go in .y
                let low = (*value as u64 & 0xFFFFFFFF) as u32;
                let high = ((*value as u64) >> 32) as u32;
                // Use LOADI_INT for low, SETY for high
                self.emit.loadi_uint(dst, low);
                self.emit.sety(dst, f32::from_bits(high));
            }

            Operator::I64Add => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_add(dst, a, b);
            }

            Operator::I64Sub => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_sub(dst, a, b);
            }

            Operator::I64Mul => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_mul(dst, a, b);
            }

            Operator::I64DivS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_div_s(dst, a, b);
            }

            Operator::I64DivU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_div_u(dst, a, b);
            }

            Operator::I64RemU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_rem_u(dst, a, b);
            }

            Operator::I64RemS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_rem_s(dst, a, b);
            }

            Operator::I64And => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_and(dst, a, b);
            }

            Operator::I64Or => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_or(dst, a, b);
            }

            Operator::I64Xor => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_xor(dst, a, b);
            }

            Operator::I64Shl => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_shl(dst, a, b);
            }

            Operator::I64ShrS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_shr_s(dst, a, b);
            }

            Operator::I64ShrU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_shr_u(dst, a, b);
            }

            Operator::I64Eq => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_eq(dst, a, b);
            }

            Operator::I64Ne => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_ne(dst, a, b);
            }

            Operator::I64LtU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_lt_u(dst, a, b);
            }

            Operator::I64LtS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_lt_s(dst, a, b);
            }

            Operator::I64GtU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // gt = lt with swapped operands
                self.emit.int64_lt_u(dst, b, a);
            }

            Operator::I64GtS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_lt_s(dst, b, a);
            }

            Operator::I64LeU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_le_u(dst, a, b);
            }

            Operator::I64LeS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_le_s(dst, a, b);
            }

            Operator::I64GeU => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // ge = le with swapped operands
                self.emit.int64_le_u(dst, b, a);
            }

            Operator::I64GeS => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_le_s(dst, b, a);
            }

            Operator::I64Eqz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_eqz(dst, a);
            }

            Operator::I64Rotr => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_rotr(dst, a, b);
            }

            Operator::I64Rotl => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // rotl(a, b) = rotr(a, 64 - b)
                // But it's cleaner to implement directly
                self.emit.int64_rotl(dst, a, b);
            }

            Operator::I64Clz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_clz(dst, a);
            }

            Operator::I64Ctz => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_ctz(dst, a);
            }

            Operator::I64Popcnt => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_popcnt(dst, a);
            }

            Operator::I32WrapI64 => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_wrap(dst, a);
            }

            Operator::I64ExtendI32U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_extend_u(dst, a);
            }

            Operator::I64ExtendI32S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_extend_s(dst, a);
            }

            // Sign extension operations (required for WASM MVP)
            Operator::I32Extend8S => {
                // Sign-extend from 8 bits to 32 bits
                // ((val << 24) >> 24) using arithmetic shift
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let tmp = self.stack.scratch(0);
                self.emit.loadi_int(tmp, 24);
                self.emit.shl(dst, a, tmp);           // dst = a << 24
                self.emit.shr_s(dst, dst, tmp);       // dst = dst >> 24 (arithmetic)
            }

            Operator::I32Extend16S => {
                // Sign-extend from 16 bits to 32 bits
                // ((val << 16) >> 16) using arithmetic shift
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let tmp = self.stack.scratch(0);
                self.emit.loadi_int(tmp, 16);
                self.emit.shl(dst, a, tmp);           // dst = a << 16
                self.emit.shr_s(dst, dst, tmp);       // dst = dst >> 16 (arithmetic)
            }

            Operator::I64Extend8S => {
                // Sign-extend from 8 bits to 64 bits
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let tmp = self.stack.scratch(0);
                self.emit.loadi_int(tmp, 56);
                self.emit.int64_shl(dst, a, tmp);
                self.emit.int64_shr_s(dst, dst, tmp);
            }

            Operator::I64Extend16S => {
                // Sign-extend from 16 bits to 64 bits
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                let tmp = self.stack.scratch(0);
                self.emit.loadi_int(tmp, 48);
                self.emit.int64_shl(dst, a, tmp);
                self.emit.int64_shr_s(dst, dst, tmp);
            }

            Operator::I64Extend32S => {
                // Sign-extend from 32 bits to 64 bits
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.int64_extend_s(dst, a);
            }

            // ═══════════════════════════════════════════════════════════════
            // F64 OPERATIONS - DOUBLE-SINGLE EMULATION (Issue #27)
            // Metal does NOT support native double precision.
            // We use double-single representation: value = hi + lo stored in xy
            // This provides ~47 bits of mantissa precision (vs 52 for native f64)
            // ═══════════════════════════════════════════════════════════════

            Operator::F64Const { value } => {
                let dst = self.stack.alloc_and_push()?;
                let f64_val = f64::from_bits(value.bits());
                self.emit.loadi_f64(dst, f64_val);  // Stores hi in .x, lo in .y
            }

            Operator::F64Add => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_add(dst, a, b);  // Double-single addition
            }

            Operator::F64Sub => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_sub(dst, a, b);  // Double-single subtraction
            }

            Operator::F64Mul => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_mul(dst, a, b);  // Double-single multiplication
            }

            Operator::F64Div => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_div(dst, a, b);  // Double-single division
            }

            Operator::F64Sqrt => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_sqrt(dst, a);  // Double-single square root
            }

            Operator::F64Neg => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // Negate both hi and lo components
                // TODO: Add f64_neg to emit.rs for proper double-single negation
                self.emit.fneg(dst, a);  // Negates .x only - NEEDS FIX
            }

            // F64 unary ops - operate on hi component, lo stays 0 for these
            // These ops work on the double-single value but may lose lo precision
            Operator::F64Abs => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_abs for proper double-single abs
                self.emit.abs(dst, a);  // Applies abs to .x only
            }

            Operator::F64Ceil => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_ceil for proper double-single ceil
                self.emit.ceil(dst, a);
            }

            Operator::F64Floor => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_floor for proper double-single floor
                self.emit.floor(dst, a);
            }

            Operator::F64Trunc => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_trunc for proper double-single trunc
                self.emit.trunc(dst, a);
            }

            Operator::F64Nearest => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_nearest for proper double-single nearest
                self.emit.nearest(dst, a);
            }

            // F64 binary ops
            Operator::F64Copysign => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                // TODO: Add f64_copysign for proper double-single copysign
                self.emit.copysign(dst, a, b);
            }

            Operator::F64Min => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_min(dst, a, b);  // Double-single aware min
            }

            Operator::F64Max => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_max(dst, a, b);  // Double-single aware max
            }

            // F64 comparisons - double-single aware (compares both hi and lo)
            Operator::F64Eq => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_eq(dst, a, b);
            }

            Operator::F64Ne => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_ne(dst, a, b);
            }

            Operator::F64Lt => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_lt(dst, a, b);
            }

            Operator::F64Gt => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_gt(dst, a, b);
            }

            Operator::F64Le => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_le(dst, a, b);
            }

            Operator::F64Ge => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_ge(dst, a, b);
            }

            // F64 promote (f32 -> f64) - convert to double-single format
            Operator::F64PromoteF32 => {
                // f32 -> double-single: hi = f32 value, lo = 0
                let val = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, val);   // Copy .x (f32 value)
                self.emit.sety(dst, 0.0);  // Set .y = 0 (lo part)
            }

            // F32 demote (f64 -> f32) - extract hi part
            Operator::F32DemoteF64 => {
                // double-single -> f32: just take .x (hi) part
                // TODO: Add hi + lo for better precision in the f32 result
                let val = self.stack.pop()?;
                self.stack.push(val);  // .x already has the hi value
            }

            // F64 conversion from integers - use double-single conversion
            Operator::F64ConvertI32S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_from_i32_s(dst, a);  // i32 -> double-single
            }

            Operator::F64ConvertI32U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_from_i32_u(dst, a);  // u32 -> double-single
            }

            Operator::F64ConvertI64S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_from_i64_s(dst, a);  // i64 -> double-single
            }

            Operator::F64ConvertI64U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_from_i64_u(dst, a);  // u64 -> double-single
            }

            // F64 conversion to integers - use double-single conversion
            Operator::I32TruncF64S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i32_s(dst, a);  // double-single -> i32
            }

            Operator::I32TruncF64U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i32_u(dst, a);  // double-single -> u32
            }

            Operator::I64TruncF64S => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i64_s(dst, a);  // double-single -> i64
            }

            Operator::I64TruncF64U => {
                let a = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_to_i64_u(dst, a);  // double-single -> u64
            }

            // F64 reinterpret operations (approximations for double-single)
            // These reconstruct the f64 value from IEEE 754 bits and vice versa.
            // Not a true zero-cost reinterpret, but gives semantically correct results.
            Operator::F64ReinterpretI64 => {
                // i64 bits -> f64 value
                let src = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.f64_reinterpret_i64(dst, src);
            }
            Operator::I64ReinterpretF64 => {
                // f64 value -> i64 bits
                let src = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.i64_reinterpret_f64(dst, src);
            }

            // ═══════════════════════════════════════════════════════════════
            // FUNCTION CALLS (Phase 5 - Issue #178)
            // THE GPU IS THE COMPUTER - intrinsics map to GPU ops, helpers inline
            // ═══════════════════════════════════════════════════════════════

            Operator::Call { function_index } => {
                self.translate_call(*function_index)?;
            }

            // ═══════════════════════════════════════════════════════════════
            // INDIRECT FUNCTION CALLS (Issue #189)
            // THE GPU IS THE COMPUTER - call_indirect resolves vtables at compile time
            //
            // WASM call_indirect: pop table index, look up function, call it
            // Our approach: at translation time, check if all possible table entries
            // can be known statically. If so, emit a switch-case that inlines each.
            // For single-entry cases (common for Rust vtables), inline directly.
            // ═══════════════════════════════════════════════════════════════
            Operator::CallIndirect { type_index, table_index } => {
                self.translate_call_indirect(*type_index, *table_index)?;
            }

            _ => {
                return Err(TranslateError::Unsupported(format!("{:?}", op)));
            }
        }

        Ok(())
    }

    /// Finish translation and return bytecode
    pub fn finish(mut self) -> Vec<u8> {
        // Define epilogue label (jumped to by Return)
        self.emit.define_label(self.epilogue_label);

        // Store return value (r4) to state[3] (after allocator header)
        // State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | heap[4+]
        // This allows us to read the result after execution
        // Issue #213 fix: Use loadi (float VALUE 3.0) instead of loadi_uint (integer BITS 0x3)
        // LD/ST now use uint(float) conversion, so we need float values not bit patterns
        //
        // Note: i32 values are now stored as float VALUES by LOADI_INT (Issue #213 fix).
        // No conversion needed here - r4 already contains a normalized float value.
        self.emit.loadi(30, 3.0);     // state index 3.0 (as float)
        self.emit.st(30, 4, 0.0);     // state[3] = r4 (already a float value)

        // Halt execution
        self.emit.halt();

        self.emit.finish(10000)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FUNCTION CALL SUPPORT (Phase 5 - Issue #178)
    // THE GPU IS THE COMPUTER - intrinsics are GPU ops, helpers are inlined
    // ═══════════════════════════════════════════════════════════════════════════

    /// Translate a function call
    fn translate_call(&mut self, func_idx: u32) -> Result<(), TranslateError> {
        let module = self.module
            .ok_or_else(|| TranslateError::Invalid("module not set for function calls".into()))?;

        // Check if it's an import (GPU intrinsic)
        if module.is_import(func_idx) {
            let import = module.get_import(func_idx)
                .ok_or_else(|| TranslateError::Invalid("import not found".into()))?;

            if let Some(intrinsic) = import.intrinsic {
                return self.emit_intrinsic(intrinsic);
            } else {
                return Err(TranslateError::Unsupported(
                    format!("unsupported import: {}:{}", import.module, import.name)
                ));
            }
        }

        // It's a defined function - inline it
        self.inline_function(func_idx)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INDIRECT FUNCTION CALL SUPPORT (Issue #189)
    // THE GPU IS THE COMPUTER - resolve vtable dispatch at compile time
    //
    // call_indirect pops a table index from the stack and calls the function
    // at that position in the function table. Since the table is statically
    // initialized from element segments, we can:
    // 1. For single-entry tables: just inline that function
    // 2. For small tables: emit if-else chain to pick the right function
    // 3. For dynamic indices: emit runtime switch (less common in practice)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Translate an indirect function call
    fn translate_call_indirect(&mut self, type_index: u32, table_index: u32) -> Result<(), TranslateError> {
        let module = self.module
            .ok_or_else(|| TranslateError::Invalid("module not set for call_indirect".into()))?;

        // We only support table 0 for now
        if table_index != 0 {
            return Err(TranslateError::Unsupported(
                format!("call_indirect with table index {} (only table 0 supported)", table_index)
            ));
        }

        // Pop the table index from the stack
        let idx_reg = self.stack.pop()?;

        // Get the function table
        let func_table = &module.func_table;

        // Collect all non-None entries with their indices
        let entries: Vec<(usize, u32)> = func_table.iter()
            .enumerate()
            .filter_map(|(i, f)| f.map(|func_idx| (i, func_idx)))
            .collect();

        if entries.is_empty() {
            // Empty table - this shouldn't happen in valid WASM, but emit trap
            self.emit.halt();
            return Ok(());
        }

        // Get the expected function type for validation
        let expected_type = module.types.get(type_index as usize)
            .ok_or_else(|| TranslateError::Invalid("type index out of bounds".into()))?;

        // Optimization: if all entries have the same function, just inline it
        let all_same = entries.iter().all(|(_, f)| *f == entries[0].1);
        if all_same {
            // All entries point to the same function - just inline it
            // The table index becomes dead code (we already validated it)
            return self.translate_call(entries[0].1);
        }

        // General case: emit a switch-case based on the table index
        // For each possible table entry, check if idx == entry_index, then inline that function
        //
        // CRITICAL: We must save and restore stack state for each branch because:
        // - At compile time, we generate code for ALL branches
        // - At runtime, only ONE branch executes
        // - Each branch's translate_call pops arguments from the stack
        // - Without save/restore, the second branch would see an empty stack

        let end_label = self.emit.new_label();

        // Save stack state BEFORE any branch (after popping table index)
        let saved_state = self.stack.save_state();

        for (i, (table_slot, func_idx)) in entries.iter().enumerate() {
            // Restore stack state for this branch (so each branch sees the same args)
            if i > 0 {
                self.stack.restore_state(saved_state.clone());
            }

            // Check if idx_reg == table_slot
            let skip_label = self.emit.new_label();

            // Load table slot constant
            self.emit.loadi_uint(30, *table_slot as u32);

            // Compare: r31 = (idx_reg == table_slot)
            self.emit.int_eq(31, idx_reg, 30);

            // If not equal, skip to next case
            self.emit.jz_label(31, skip_label);

            // Validate function type matches (at compile time)
            if let Some(func_type) = module.get_func_type(*func_idx) {
                if func_type.params.len() != expected_type.params.len() ||
                   func_type.results.len() != expected_type.results.len() {
                    return Err(TranslateError::Invalid(
                        format!("type mismatch in call_indirect: expected {:?} params, got {:?}",
                                expected_type.params.len(), func_type.params.len())
                    ));
                }
            }

            // Inline the function at this table slot
            self.translate_call(*func_idx)?;

            // Jump to end
            self.emit.jmp_label(end_label);

            // Define skip label for this case
            self.emit.define_label(skip_label);
        }

        // If we reach here, the table index was out of bounds - trap
        self.emit.halt();

        // Define the end label
        self.emit.define_label(end_label);

        Ok(())
    }

    /// Emit GPU intrinsic operation
    fn emit_intrinsic(&mut self, intrinsic: GpuIntrinsic) -> Result<(), TranslateError> {
        match intrinsic {
            GpuIntrinsic::ThreadId => {
                // thread_id() -> i32
                // GPU r1 contains thread ID AS A FLOAT (e.g., 5.0f for thread 5)
                // WASM expects an integer, so we must convert float→int
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 1);  // r1 = thread_id (as float)
                self.emit.f_to_int(dst, dst);  // Convert float → int bit pattern
            }

            GpuIntrinsic::ThreadgroupSize => {
                // threadgroup_size() -> i32
                // GPU r2 contains threadgroup size AS A FLOAT (e.g., 256.0f)
                // WASM expects an integer, so we must convert float→int
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 2);  // r2 = threadgroup_size (as float)
                self.emit.f_to_int(dst, dst);  // Convert float → int bit pattern
            }

            GpuIntrinsic::Frame => {
                // frame() -> i32
                // GPU r3 contains frame number AS A FLOAT (e.g., 60.0f for frame 60)
                // WASM expects an integer, so we must convert float→int to get proper bit pattern
                // Otherwise INT_ADD/INT_SUB will reinterpret 60.0f bits as 0x42700000 = 1.1 billion (WRONG!)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 3);  // r3 = frame (as float)
                self.emit.f_to_int(dst, dst);  // Convert float 60.0 → int bit pattern for 60
            }

            GpuIntrinsic::Sin => {
                // sin(x: f32) -> f32
                let src = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.sin(dst, src);
            }

            GpuIntrinsic::Cos => {
                // cos(x: f32) -> f32
                let src = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.cos(dst, src);
            }

            GpuIntrinsic::Sqrt => {
                // sqrt(x: f32) -> f32
                let src = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.sqrt(dst, src);
            }

            // ═══════════════════════════════════════════════════════════════
            // ALLOCATOR INTRINSICS (Phase 6 - Issue #179)
            // THE GPU IS THE COMPUTER - GPU-resident memory allocator
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::Alloc => {
                // __rust_alloc(size: usize, align: usize) -> *mut u8
                let align = self.stack.pop()?;
                let size = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.alloc(dst, size, align);
            }

            GpuIntrinsic::Dealloc => {
                // __rust_dealloc(ptr: *mut u8, size: usize, align: usize)
                let align = self.stack.pop()?;
                let size = self.stack.pop()?;
                let ptr = self.stack.pop()?;
                self.emit.dealloc(ptr, size, align);
            }

            GpuIntrinsic::Realloc => {
                // __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8
                let new_size = self.stack.pop()?;
                let _align = self.stack.pop()?;  // align not used in our implementation
                let old_size = self.stack.pop()?;
                let ptr = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.realloc(dst, ptr, old_size, new_size);
            }

            GpuIntrinsic::AllocZeroed => {
                // __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8
                let align = self.stack.pop()?;
                let size = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.alloc_zero(dst, size, align);
            }

            // ═══════════════════════════════════════════════════════════════
            // DEBUG I/O INTRINSICS (Phase 7 - Issue #180)
            // THE GPU IS THE COMPUTER - debug output via ring buffer
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::DebugI32 => {
                // __gpu_debug_i32(value: i32)
                let src = self.stack.pop()?;
                self.emit.dbg_i32(src);
            }

            GpuIntrinsic::DebugF32 => {
                // __gpu_debug_f32(value: f32)
                let src = self.stack.pop()?;
                self.emit.dbg_f32(src);
            }

            GpuIntrinsic::DebugStr => {
                // __gpu_debug_str(ptr: i32, len: i32)
                let len = self.stack.pop()?;
                let ptr = self.stack.pop()?;
                self.emit.dbg_str(ptr, len);
            }

            GpuIntrinsic::DebugBool => {
                // __gpu_debug_bool(value: i32)
                let src = self.stack.pop()?;
                self.emit.dbg_bool(src);
            }

            GpuIntrinsic::DebugNewline => {
                // __gpu_debug_newline()
                self.emit.dbg_nl();
            }

            GpuIntrinsic::DebugFlush => {
                // __gpu_debug_flush()
                self.emit.dbg_flush();
            }

            // ═══════════════════════════════════════════════════════════════
            // AUTOMATIC CODE TRANSFORMATION INTRINSICS (Phase 8 - Issue #182)
            // THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::WorkPush => {
                // __gpu_work_push(item: i32, queue: i32)
                let queue_reg = self.stack.pop()?;
                let item_reg = self.stack.pop()?;
                self.emit.work_push(item_reg, queue_reg);
            }

            GpuIntrinsic::WorkPop => {
                // __gpu_work_pop(queue: i32) -> i32
                let queue_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.work_pop(dst, queue_reg);
            }

            GpuIntrinsic::Barrier => {
                // __gpu_barrier() - threadgroup synchronization
                self.emit.barrier();
            }

            GpuIntrinsic::FrameWait => {
                // __gpu_frame_wait(frames: i32)
                let frames_reg = self.stack.pop()?;
                self.emit.frame_wait(frames_reg);
            }

            GpuIntrinsic::Spinlock => {
                // __gpu_spinlock(lock_addr: i32) -> i32 (returns 1 if acquired, 0 if timeout)
                let lock_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.spinlock(lock_reg);
                // Result is implicitly in same register as lock_reg for now
                // Use mov to copy result to dst
                self.emit.mov(dst, lock_reg);
            }

            GpuIntrinsic::Spinunlock => {
                // __gpu_spinunlock(lock_addr: i32)
                let lock_reg = self.stack.pop()?;
                self.emit.spinunlock(lock_reg);
            }

            GpuIntrinsic::RcClone => {
                // __gpu_rc_clone(refcount_addr: i32) -> i32 (returns old refcount)
                let addr_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.rc_clone(dst, addr_reg);
            }

            GpuIntrinsic::RcDrop => {
                // __gpu_rc_drop(refcount_addr: i32) -> i32 (returns old refcount)
                let addr_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.rc_drop(dst, addr_reg);
            }

            GpuIntrinsic::RequestQueue => {
                // __gpu_request_queue(type: i32, data: i32)
                let data_reg = self.stack.pop()?;
                let type_reg = self.stack.pop()?;
                self.emit.request_queue(type_reg, data_reg);
            }

            GpuIntrinsic::RequestPoll => {
                // __gpu_request_poll(id: i32) -> i32
                let id_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.request_poll(dst, id_reg);
            }

            // ═══════════════════════════════════════════════════════════════
            // RENDERING INTRINSICS (Phase 9 - GPU App Framework)
            // THE GPU IS THE COMPUTER - emit graphics primitives from WASM apps
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::EmitQuad => {
                // emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32)
                // Stack order (top to bottom): color, h, w, y, x
                //
                // QUAD opcode format:
                //   s1 = position register (xy = x, y)
                //   s2 = size register (xy = w, h)
                //   d = color register (float4 RGBA)
                //   imm = depth (default 0.0)

                let color_reg = self.stack.pop()?;  // u32 packed RGBA (0xRRGGBBAA)
                let h_reg = self.stack.pop()?;
                let w_reg = self.stack.pop()?;
                let y_reg = self.stack.pop()?;
                let x_reg = self.stack.pop()?;

                // Use PACK2 to combine scalar values into float2:
                // CRITICAL FIX: Use scratch registers r30/r31 instead of r28/r29
                // r28/r29 are used for local variable spill area and would collide
                // with locals 4 and 5 in functions with more than 4 locals!
                // r30.xy = (x_reg.x, y_reg.x) for position
                // r31.xy = (w_reg.x, h_reg.x) for size
                self.emit.pack2(30, x_reg, y_reg);  // r30.xy = (x, y)
                self.emit.pack2(31, w_reg, h_reg);  // r31.xy = (w, h)

                // Color: color_reg contains u32 packed as 0xRRGGBBAA
                // The GPU shader reads this as float4 where the bits represent RGBA
                // For proper rendering, we'd need to unpack to normalized floats
                // For now, pass the raw bits - the write_quad function handles it

                // Emit QUAD: pos_reg=r30, size_reg=r31, color_reg, depth=0.0
                self.emit.quad(30, 31, color_reg, 0.0);
            }

            GpuIntrinsic::GetCursorX => {
                // get_cursor_x() -> f32
                // GPU frame state: cursor_x is at known offset in frame state buffer
                // For now, use a dedicated register r16 for cursor_x (set by runtime)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 16);  // r16 = cursor_x (convention)
            }

            GpuIntrinsic::GetCursorY => {
                // get_cursor_y() -> f32
                // r17 = cursor_y (convention)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 17);  // r17 = cursor_y
            }

            GpuIntrinsic::GetMouseDown => {
                // get_mouse_down() -> i32
                // r18 = mouse_down AS A FLOAT (0.0 or 1.0)
                // WASM expects an integer, so we must convert float→int
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 18);  // r18 = mouse_down (as float)
                self.emit.f_to_int(dst, dst);  // Convert float → int bit pattern
            }

            GpuIntrinsic::GetTime => {
                // get_time() -> f32
                // r19 = time (convention)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 19);  // r19 = time
            }

            GpuIntrinsic::GetScreenWidth => {
                // get_screen_width() -> f32
                // r20 = screen_width (convention)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 20);  // r20 = screen_width
            }

            GpuIntrinsic::GetScreenHeight => {
                // get_screen_height() -> f32
                // r21 = screen_height (convention)
                let dst = self.stack.alloc_and_push()?;
                self.emit.mov(dst, 21);  // r21 = screen_height
            }

            // ═══════════════════════════════════════════════════════════════
            // WASI INTRINSICS (Issue #207 - GPU-Native WASI)
            // THE GPU IS THE COMPUTER - WASI system calls on GPU
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::WasiFdWrite => {
                // fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32
                let nwritten_reg = self.stack.pop()?;
                let _iovs_len_reg = self.stack.pop()?;  // Not used - simplification
                let iovs_reg = self.stack.pop()?;
                let fd_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_fd_write(dst, fd_reg, iovs_reg, nwritten_reg);
            }

            GpuIntrinsic::WasiFdRead => {
                // fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32
                let nread_reg = self.stack.pop()?;
                let _iovs_len_reg = self.stack.pop()?;
                let iovs_reg = self.stack.pop()?;
                let fd_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_fd_read(dst, fd_reg, iovs_reg, nread_reg);
            }

            GpuIntrinsic::WasiProcExit => {
                // proc_exit(code: i32) -> !
                let code_reg = self.stack.pop()?;
                self.emit.wasi_proc_exit(code_reg);
            }

            GpuIntrinsic::WasiEnvironSizesGet => {
                // environ_sizes_get(count: i32, size: i32) -> i32
                let size_reg = self.stack.pop()?;
                let count_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_environ_sizes_get(dst, count_reg, size_reg);
            }

            GpuIntrinsic::WasiEnvironGet => {
                // environ_get(environ: i32, buf: i32) -> i32
                let buf_reg = self.stack.pop()?;
                let environ_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_environ_get(dst, environ_reg, buf_reg);
            }

            GpuIntrinsic::WasiArgsSizesGet => {
                // args_sizes_get(count: i32, size: i32) -> i32
                let size_reg = self.stack.pop()?;
                let count_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_args_sizes_get(dst, count_reg, size_reg);
            }

            GpuIntrinsic::WasiArgsGet => {
                // args_get(argv: i32, buf: i32) -> i32
                let buf_reg = self.stack.pop()?;
                let argv_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_args_get(dst, argv_reg, buf_reg);
            }

            GpuIntrinsic::WasiClockTimeGet => {
                // clock_time_get(clock_id: i32, precision: i64, time: i32) -> i32
                let time_reg = self.stack.pop()?;
                let _precision_reg = self.stack.pop()?;  // Not used
                let clock_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_clock_time_get(dst, clock_reg, time_reg);
            }

            GpuIntrinsic::WasiRandomGet => {
                // random_get(buf: i32, len: i32) -> i32
                let len_reg = self.stack.pop()?;
                let buf_reg = self.stack.pop()?;
                let dst = self.stack.alloc_and_push()?;
                self.emit.wasi_random_get(dst, buf_reg, len_reg);
            }

            // ═══════════════════════════════════════════════════════════════
            // PANIC HANDLING INTRINSICS (Issue #209 - GPU-Native Panic)
            // THE GPU IS THE COMPUTER - panic handling on GPU
            // ═══════════════════════════════════════════════════════════════

            GpuIntrinsic::Panic => {
                // panic(msg_ptr: i32, msg_len: i32) -> !
                // Note: Rust's panic ABI varies, but we handle the simple case
                let msg_len_reg = self.stack.pop()?;
                let msg_ptr_reg = self.stack.pop()?;
                self.emit.panic(msg_ptr_reg, msg_len_reg);
                // Panic never returns, so no result pushed
            }

            GpuIntrinsic::Unreachable => {
                // unreachable() -> !
                // This is called for WASM unreachable instruction and Rust unreachable!()
                self.emit.unreachable();
                // Unreachable never returns, so no result pushed
            }
        }

        Ok(())
    }

    /// Inline a defined function
    fn inline_function(&mut self, func_idx: u32) -> Result<(), TranslateError> {
        let module = self.module
            .ok_or_else(|| TranslateError::Invalid("module not set for inlining".into()))?;
        let function_ops = self.function_ops
            .ok_or_else(|| TranslateError::Invalid("function_ops not set for inlining".into()))?;
        let function_local_counts = self.function_local_counts
            .ok_or_else(|| TranslateError::Invalid("function_local_counts not set for inlining".into()))?;

        // Check for recursion
        if self.call_stack.contains(&func_idx) {
            return Err(TranslateError::Unsupported(
                format!("recursion detected: function {} calls itself", func_idx)
            ));
        }

        // Mark this function as being processed
        self.call_stack.insert(func_idx);

        // Get the function's operators
        let import_count = module.import_count();
        let defined_idx = func_idx.checked_sub(import_count)
            .ok_or_else(|| TranslateError::Invalid("cannot inline import".into()))?;

        let ops = function_ops.get(defined_idx as usize)
            .ok_or_else(|| TranslateError::Invalid(
                format!("function {} not found", func_idx)
            ))?;

        // Get function type for parameter handling
        let func_type = module.get_func_type(func_idx)
            .ok_or_else(|| TranslateError::Invalid("function type not found".into()))?;

        // Get local count for this function
        let local_count = *function_local_counts.get(defined_idx as usize)
            .ok_or_else(|| TranslateError::Invalid("local count not found".into()))?;

        // Pop arguments from stack into temporary registers
        // Arguments are passed in reverse order on the WASM stack
        let param_count = func_type.params.len() as u32;
        let mut arg_regs = Vec::with_capacity(param_count as usize);
        for i in 0..param_count {
            if let Ok(reg) = self.stack.pop() {
                arg_regs.push(reg);
            } else {
                eprintln!("[INLINE ERROR] func_idx={}, need {} params, only got {}, stack_depth={}",
                    func_idx, param_count, i, self.stack.depth());
                return Err(TranslateError::StackUnderflow);
            }
        }
        arg_regs.reverse(); // Now arg_regs[0] = first argument

        // For inline functions, we treat parameters as if they're in r4..rN
        // Copy arguments to parameter registers (r4, r5, r6, r7)
        for (i, &arg_reg) in arg_regs.iter().enumerate() {
            if i < 4 {
                let param_reg = 4 + i as u8;
                if arg_reg != param_reg {
                    self.emit.mov(param_reg, arg_reg);
                }
            } else {
                // Spill to memory for >4 params
                // Issue #213 fix: Use loadi (float VALUE) for addresses
                let spill_addr = self.config.globals_base + 256 + (i as u32);
                self.emit.loadi(30, spill_addr as f32);
                self.emit.st(30, arg_reg, 0.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════════
        // CRITICAL FIX: Create a new LocalMap for the inlined function
        // Each inlined function needs its own local variable space to avoid collisions
        // with the caller's locals or other inlined functions.
        // ═══════════════════════════════════════════════════════════════════════════
        let saved_locals = std::mem::replace(
            &mut self.locals,
            LocalMap::new(
                param_count,
                local_count,
                // Use a unique spill base for each inlined function
                // This ensures locals from different functions don't collide in memory
                //
                // CRITICAL FIX: Use the spill area (state[72-263]) for inline functions
                // Main function uses state[264-327] (globals_base + 256)
                // Previously used: globals_base + 512 = 520+ which overlaps linear memory!
                //
                // Memory layout:
                //   state[8-71] = globals (64 slots)
                //   state[72-263] = inline function spill (192 slots, 32 per function max)
                //   state[264-327] = main function spill (64 slots)
                //   state[328+] = linear memory (byte 5248+)
                72 + self.inline_spill_counter * 32,
            ),
        );
        self.inline_spill_counter += 1;

        // Create inline block for proper End/Return handling
        // InlineFunction block type tells Return to jump here instead of epilogue
        let inline_end = self.emit.new_label();
        let result_count = func_type.results.len();
        self.block_stack.push(BlockContext {
            kind: BlockKind::InlineFunction,
            start_label: None,
            else_label: None,
            end_label: inline_end,
            stack_depth: self.stack.depth(),
            result_count,
        });

        // Translate each operator (except the final End which ends the function)
        let trace_func = std::env::var("TRACE_FUNC").ok().and_then(|v| v.parse::<u32>().ok());

        for (i, op) in ops.iter().enumerate() {
            // Skip the final End - it ends the function, not a block
            if i == ops.len() - 1 {
                if let Operator::End = op {
                    break;
                }
            }
            // Detailed tracing for a specific function
            if trace_func == Some(func_idx) {
                eprintln!("[TRACE] func[{}] op[{}] stack_before={}: {:?}",
                    func_idx, i, self.stack.depth(), op);
            }
            if let Err(e) = self.translate_operator(op) {
                eprintln!("[INLINE ERROR] func_idx={}, op[{}]: {:?}, stack_depth={}",
                    func_idx, i, op, self.stack.depth());
                return Err(e);
            }
        }

        // Pop the inline block
        if let Some(ctx) = self.block_stack.pop() {
            self.emit.define_label(ctx.end_label);

            // ═══════════════════════════════════════════════════════════════════════════
            // CRITICAL FIX: Handle trap-only functions (e.g., panic handlers)
            //
            // Some functions contain only `Unreachable` (trap) and are declared to return
            // a value. When inlined, the stack is empty but result_count > 0.
            //
            // WASM semantics require the stack to balance even for dead code paths.
            // We push a placeholder value that will never be used (code after the trap
            // is unreachable anyway).
            // ═══════════════════════════════════════════════════════════════════════════
            if ctx.result_count > 0 && self.stack.depth() < ctx.result_count {
                let missing = ctx.result_count - self.stack.depth();
                for _ in 0..missing {
                    // Push a placeholder register with value 0
                    // This code is unreachable in practice, but needed for stack balance
                    let placeholder = self.stack.alloc_and_push()?;
                    self.emit.loadi(placeholder, 0.0);
                }
            }
        }

        // Restore the caller's LocalMap
        self.locals = saved_locals;

        // If function returns a value, it should be on the stack
        // The translated code will have pushed the result

        // Remove from call stack
        self.call_stack.remove(&func_idx);

        Ok(())
    }
}
