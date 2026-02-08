//! Bytecode emitter for WASM translator
//!
//! THE GPU IS THE COMPUTER.
//! Wraps BytecodeAssembler with label support for control flow.

use rust_experiment::gpu_os::gpu_app_system::BytecodeAssembler;
use std::collections::HashMap;

/// Bytecode emitter with label support
pub struct Emitter {
    asm: BytecodeAssembler,
    /// Label name -> PC
    labels: HashMap<usize, usize>,
    /// Instructions needing patch: (instruction PC, label ID)
    patches: Vec<(usize, usize)>,
    /// Next label ID
    next_label: usize,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            asm: BytecodeAssembler::new(),
            labels: HashMap::new(),
            patches: Vec::new(),
            next_label: 0,
        }
    }

    /// Get current PC
    pub fn pc(&self) -> usize {
        self.asm.pc()
    }

    /// Create a new label, returns label ID
    pub fn new_label(&mut self) -> usize {
        let id = self.next_label;
        self.next_label += 1;
        id
    }

    /// Define a label at current PC
    pub fn define_label(&mut self, label: usize) {
        self.labels.insert(label, self.pc());
    }

    /// Emit NOP
    pub fn nop(&mut self) {
        self.asm.nop();
    }

    /// Emit HALT
    pub fn halt(&mut self) {
        self.asm.halt();
    }

    /// Emit MOV dst, src
    pub fn mov(&mut self, dst: u8, src: u8) {
        self.asm.mov(dst, src);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGER OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn loadi_int(&mut self, dst: u8, val: i32) {
        self.asm.loadi_int(dst, val);
    }

    pub fn loadi_uint(&mut self, dst: u8, val: u32) {
        self.asm.loadi_uint(dst, val);
    }

    /// Load packed RGBA color (0xRRGGBBAA) and decompose to float4(R/255, G/255, B/255, A/255)
    /// Bits are preserved in the immediate field, avoiding float precision loss.
    pub fn loadi_rgba(&mut self, dst: u8, packed_color: u32) {
        self.asm.loadi_rgba(dst, packed_color);
    }

    /// Load 4 bytes from memory as RGBA color: dst = float4(R/255, G/255, B/255, A/255)
    pub fn ld_rgba(&mut self, dst: u8, addr_reg: u8, offset: f32) {
        self.asm.ld_rgba(dst, addr_reg, offset);
    }

    pub fn int_add(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_add(dst, a, b);
    }

    pub fn int_sub(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_sub(dst, a, b);
    }

    pub fn int_mul(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_mul(dst, a, b);
    }

    pub fn int_div_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_div_s(dst, a, b);
    }

    pub fn int_div_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_div_u(dst, a, b);
    }

    pub fn int_rem_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_rem_s(dst, a, b);
    }

    pub fn int_rem_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_rem_u(dst, a, b);
    }

    pub fn int_neg(&mut self, dst: u8, src: u8) {
        self.asm.int_neg(dst, src);
    }

    // Bitwise
    pub fn bit_and(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.bit_and(dst, a, b);
    }

    pub fn bit_or(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.bit_or(dst, a, b);
    }

    pub fn bit_xor(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.bit_xor(dst, a, b);
    }

    pub fn bit_not(&mut self, dst: u8, src: u8) {
        self.asm.bit_not(dst, src);
    }

    pub fn shl(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.shl(dst, a, b);
    }

    pub fn shr_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.shr_u(dst, a, b);
    }

    pub fn shr_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.shr_s(dst, a, b);
    }

    pub fn rotl(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.rotl(dst, a, b);
    }

    pub fn rotr(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.rotr(dst, a, b);
    }

    pub fn clz(&mut self, dst: u8, src: u8) {
        self.asm.clz(dst, src);
    }

    pub fn ctz(&mut self, dst: u8, src: u8) {
        self.asm.ctz(dst, src);
    }

    pub fn popcnt(&mut self, dst: u8, src: u8) {
        self.asm.popcnt(dst, src);
    }

    // Comparison
    pub fn int_eq(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_eq(dst, a, b);
    }

    pub fn int_ne(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_ne(dst, a, b);
    }

    pub fn int_lt_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_lt_s(dst, a, b);
    }

    pub fn int_lt_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_lt_u(dst, a, b);
    }

    pub fn int_le_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_le_s(dst, a, b);
    }

    pub fn int_le_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int_le_u(dst, a, b);
    }

    // Conversion
    pub fn int_to_f(&mut self, dst: u8, src: u8) {
        self.asm.int_to_f(dst, src);
    }

    pub fn uint_to_f(&mut self, dst: u8, src: u8) {
        self.asm.uint_to_f(dst, src);
    }

    pub fn f_to_int(&mut self, dst: u8, src: u8) {
        self.asm.f_to_int(dst, src);
    }

    pub fn f_to_uint(&mut self, dst: u8, src: u8) {
        self.asm.f_to_uint(dst, src);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT INTEGER OPERATIONS (Issue #188)
    // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64/uint64
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn int64_add(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_add(dst, a, b);
    }

    pub fn int64_sub(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_sub(dst, a, b);
    }

    pub fn int64_mul(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_mul(dst, a, b);
    }

    pub fn int64_div_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_div_s(dst, a, b);
    }

    pub fn int64_div_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_div_u(dst, a, b);
    }

    pub fn int64_rem_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_rem_u(dst, a, b);
    }

    pub fn int64_rem_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_rem_s(dst, a, b);
    }

    pub fn int64_and(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_and(dst, a, b);
    }

    pub fn int64_or(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_or(dst, a, b);
    }

    pub fn int64_xor(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_xor(dst, a, b);
    }

    pub fn int64_shl(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_shl(dst, a, b);
    }

    pub fn int64_shr_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_shr_u(dst, a, b);
    }

    pub fn int64_shr_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_shr_s(dst, a, b);
    }

    pub fn int64_eq(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_eq(dst, a, b);
    }

    pub fn int64_ne(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_ne(dst, a, b);
    }

    pub fn int64_lt_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_lt_u(dst, a, b);
    }

    pub fn int64_lt_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_lt_s(dst, a, b);
    }

    pub fn int64_le_u(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_le_u(dst, a, b);
    }

    pub fn int64_le_s(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_le_s(dst, a, b);
    }

    pub fn int64_eqz(&mut self, dst: u8, src: u8) {
        self.asm.int64_eqz(dst, src);
    }

    pub fn int64_rotr(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_rotr(dst, a, b);
    }

    pub fn int64_rotl(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.int64_rotl(dst, a, b);
    }

    pub fn int64_clz(&mut self, dst: u8, src: u8) {
        self.asm.int64_clz(dst, src);
    }

    pub fn int64_ctz(&mut self, dst: u8, src: u8) {
        self.asm.int64_ctz(dst, src);
    }

    pub fn int64_popcnt(&mut self, dst: u8, src: u8) {
        self.asm.int64_popcnt(dst, src);
    }

    pub fn int64_wrap(&mut self, dst: u8, src: u8) {
        self.asm.int64_wrap(dst, src);
    }

    pub fn int64_extend_u(&mut self, dst: u8, src: u8) {
        self.asm.int64_extend_u(dst, src);
    }

    pub fn int64_extend_s(&mut self, dst: u8, src: u8) {
        self.asm.int64_extend_s(dst, src);
    }

    /// Set the Y component of a register (for 64-bit constant high bits)
    pub fn sety(&mut self, dst: u8, val: f32) {
        self.asm.sety(dst, val);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FLOAT OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn loadi(&mut self, dst: u8, val: f32) {
        self.asm.loadi(dst, val);
    }

    pub fn add(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.add(dst, a, b);
    }

    pub fn sub(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.sub(dst, a, b);
    }

    pub fn mul(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.mul(dst, a, b);
    }

    pub fn div(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.div(dst, a, b);
    }

    pub fn eq(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.eq(dst, a, b);
    }

    pub fn lt(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.lt(dst, a, b);
    }

    pub fn gt(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.gt(dst, a, b);
    }

    /// Float less-than-or-equal: dst = (a <= b)
    /// Emulated as: NOT(a > b) using int_eq with 0
    pub fn le(&mut self, dst: u8, a: u8, b: u8) {
        // le(a,b) = !gt(a,b)
        // Step 1: dst = (a > b) returns 1 if true, 0 if false
        self.asm.gt(dst, a, b);
        // Step 2: dst = (dst == 0) inverts: 0->1, 1->0
        self.asm.loadi_int(30, 0);        // r30 = 0
        self.asm.int_eq(dst, dst, 30);    // dst = (dst == 0)
    }

    /// Float greater-than-or-equal: dst = (a >= b)
    /// Emulated as: NOT(a < b)
    pub fn ge(&mut self, dst: u8, a: u8, b: u8) {
        // ge(a,b) = !lt(a,b)
        self.asm.lt(dst, a, b);           // dst = (a < b)
        self.asm.loadi_int(30, 0);        // r30 = 0
        self.asm.int_eq(dst, dst, 30);    // dst = (dst == 0)
    }

    /// Float not-equal: dst = (a != b)
    /// Emulated as: NOT(a == b)
    pub fn ne(&mut self, dst: u8, a: u8, b: u8) {
        // ne(a,b) = !eq(a,b)
        self.asm.eq(dst, a, b);           // dst = (a == b)
        self.asm.loadi_int(30, 0);        // r30 = 0
        self.asm.int_eq(dst, dst, 30);    // dst = (dst == 0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MATH INTRINSICS (Phase 5 - Issue #178)
    // THE GPU IS THE COMPUTER - native GPU math operations
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn sin(&mut self, dst: u8, src: u8) {
        self.asm.sin(dst, src);
    }

    pub fn cos(&mut self, dst: u8, src: u8) {
        self.asm.cos(dst, src);
    }

    pub fn sqrt(&mut self, dst: u8, src: u8) {
        self.asm.sqrt(dst, src);
    }

    // Float unary/binary math ops (Issue #198)
    pub fn abs(&mut self, dst: u8, src: u8) {
        self.asm.abs(dst, src);
    }

    pub fn ceil(&mut self, dst: u8, src: u8) {
        self.asm.ceil(dst, src);
    }

    pub fn floor(&mut self, dst: u8, src: u8) {
        self.asm.floor(dst, src);
    }

    pub fn trunc(&mut self, dst: u8, src: u8) {
        self.asm.trunc(dst, src);
    }

    pub fn nearest(&mut self, dst: u8, src: u8) {
        self.asm.nearest(dst, src);
    }

    pub fn copysign(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.copysign(dst, a, b);
    }

    pub fn fmin(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.fmin(dst, a, b);
    }

    pub fn fmax(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.fmax(dst, a, b);
    }

    pub fn fneg(&mut self, dst: u8, src: u8) {
        self.asm.fneg(dst, src);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT FLOAT OPERATIONS (Issue #189)
    // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support double precision
    // ═══════════════════════════════════════════════════════════════════════════

    // F64 arithmetic
    pub fn f64_add(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_add(dst, a, b);
    }

    pub fn f64_sub(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_sub(dst, a, b);
    }

    pub fn f64_mul(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_mul(dst, a, b);
    }

    pub fn f64_div(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_div(dst, a, b);
    }

    pub fn f64_sqrt(&mut self, dst: u8, src: u8) {
        self.asm.f64_sqrt(dst, src);
    }

    // F64 comparison (double-single aware)
    pub fn f64_eq(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_eq(dst, a, b);
    }

    pub fn f64_ne(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_ne(dst, a, b);
    }

    pub fn f64_lt(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_lt(dst, a, b);
    }

    pub fn f64_gt(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_gt(dst, a, b);
    }

    pub fn f64_le(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_le(dst, a, b);
    }

    pub fn f64_ge(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_ge(dst, a, b);
    }

    // F64 min/max (double-single aware)
    pub fn f64_min(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_min(dst, a, b);
    }

    pub fn f64_max(&mut self, dst: u8, a: u8, b: u8) {
        self.asm.f64_max(dst, a, b);
    }

    // F64 unary operations (double-single aware, Issue #294)
    pub fn f64_neg(&mut self, dst: u8, src: u8) {
        self.asm.f64_neg(dst, src);
    }

    pub fn f64_abs(&mut self, dst: u8, src: u8) {
        self.asm.f64_abs(dst, src);
    }

    pub fn f64_ceil(&mut self, dst: u8, src: u8) {
        self.asm.f64_ceil(dst, src);
    }

    pub fn f64_floor(&mut self, dst: u8, src: u8) {
        self.asm.f64_floor(dst, src);
    }

    pub fn f64_trunc(&mut self, dst: u8, src: u8) {
        self.asm.f64_trunc(dst, src);
    }

    pub fn f64_nearest(&mut self, dst: u8, src: u8) {
        self.asm.f64_nearest(dst, src);
    }

    pub fn f64_copysign(&mut self, dst: u8, mag: u8, sign: u8) {
        self.asm.f64_copysign(dst, mag, sign);
    }

    // F64 conversion from integers
    pub fn f64_from_i32_s(&mut self, dst: u8, src: u8) {
        self.asm.f64_from_i32_s(dst, src);
    }

    pub fn f64_from_i32_u(&mut self, dst: u8, src: u8) {
        self.asm.f64_from_i32_u(dst, src);
    }

    pub fn f64_from_i64_s(&mut self, dst: u8, src: u8) {
        self.asm.f64_from_i64_s(dst, src);
    }

    pub fn f64_from_i64_u(&mut self, dst: u8, src: u8) {
        self.asm.f64_from_i64_u(dst, src);
    }

    // F64 conversion to integers - trapping versions (per WASM spec)
    pub fn f64_to_i32_s(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i32_s(dst, src);
    }

    pub fn f64_to_i32_u(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i32_u(dst, src);
    }

    pub fn f64_to_i64_s(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i64_s(dst, src);
    }

    pub fn f64_to_i64_u(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i64_u(dst, src);
    }

    // F64 conversion to integers - saturating versions (per WASM spec)
    pub fn f64_to_i32_s_sat(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i32_s_sat(dst, src);
    }

    pub fn f64_to_i32_u_sat(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i32_u_sat(dst, src);
    }

    pub fn f64_to_i64_s_sat(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i64_s_sat(dst, src);
    }

    pub fn f64_to_i64_u_sat(&mut self, dst: u8, src: u8) {
        self.asm.f64_to_i64_u_sat(dst, src);
    }

    // F32 conversion - saturating versions (per WASM spec)
    pub fn f_to_int_sat(&mut self, dst: u8, src: u8) {
        self.asm.f_to_int_sat(dst, src);
    }

    pub fn f_to_uint_sat(&mut self, dst: u8, src: u8) {
        self.asm.f_to_uint_sat(dst, src);
    }

    // F64 reinterpret operations (approximations for double-single)
    pub fn f64_reinterpret_i64(&mut self, dst: u8, src: u8) {
        self.asm.f64_reinterpret_i64(dst, src);
    }

    pub fn i64_reinterpret_f64(&mut self, dst: u8, src: u8) {
        self.asm.i64_reinterpret_f64(dst, src);
    }

    /// Load 64-bit float immediate constant
    pub fn loadi_f64(&mut self, dst: u8, value: f64) {
        self.asm.loadi_f64(dst, value);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MEMORY
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn ld(&mut self, dst: u8, addr: u8, offset: f32) {
        self.asm.ld(dst, addr, offset);
    }

    pub fn st(&mut self, addr: u8, src: u8, offset: f32) {
        self.asm.st(addr, src, offset);
    }

    /// Load byte from byte address
    pub fn ld1(&mut self, dst: u8, addr: u8, offset: f32) {
        self.asm.ld1(dst, addr, offset);
    }

    /// Store byte to byte address
    pub fn st1(&mut self, addr: u8, src: u8, offset: f32) {
        self.asm.st1(addr, src, offset);
    }

    /// Load 16-bit halfword from byte address
    pub fn ld2(&mut self, dst: u8, addr: u8, offset: f32) {
        self.asm.ld2(dst, addr, offset);
    }

    /// Store 16-bit halfword to byte address
    pub fn st2(&mut self, addr: u8, src: u8, offset: f32) {
        self.asm.st2(addr, src, offset);
    }

    /// Load 32-bit word from byte address
    pub fn ld4(&mut self, dst: u8, addr: u8, offset: f32) {
        self.asm.ld4(dst, addr, offset);
    }

    /// Store 32-bit word to byte address
    pub fn st4(&mut self, addr: u8, src: u8, offset: f32) {
        self.asm.st4(addr, src, offset);
    }

    /// Get current memory size in pages (64KB per page)
    /// dst = memory_size()
    pub fn memory_size(&mut self, dst: u8) {
        self.asm.memory_size(dst);
    }

    /// Grow memory by delta pages
    /// dst = memory_grow(delta_reg) - returns old size or -1 on failure
    pub fn memory_grow(&mut self, dst: u8, delta_reg: u8) {
        self.asm.memory_grow(dst, delta_reg);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ALLOCATOR OPERATIONS (Phase 6 - Issue #179)
    // THE GPU IS THE COMPUTER - GPU-resident memory allocator for Rust alloc crate
    // ═══════════════════════════════════════════════════════════════════════════

    /// Allocate memory: dst = gpu_alloc(size_reg, align_reg)
    pub fn alloc(&mut self, dst: u8, size_reg: u8, align_reg: u8) {
        self.asm.alloc(dst, size_reg, align_reg);
    }

    /// Free memory: gpu_dealloc(ptr_reg, size_reg, align_reg)
    pub fn dealloc(&mut self, ptr_reg: u8, size_reg: u8, align_reg: u8) {
        self.asm.dealloc(ptr_reg, size_reg, align_reg);
    }

    /// Reallocate memory: dst = gpu_realloc(ptr_reg, old_size_reg, new_size_reg)
    pub fn realloc(&mut self, dst: u8, ptr_reg: u8, old_size_reg: u8, new_size_reg: u8) {
        self.asm.realloc(dst, ptr_reg, old_size_reg, new_size_reg);
    }

    /// Allocate zeroed memory: dst = gpu_alloc_zeroed(size_reg, align_reg)
    pub fn alloc_zero(&mut self, dst: u8, size_reg: u8, align_reg: u8) {
        self.asm.alloc_zero(dst, size_reg, align_reg);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG I/O OPERATIONS (Phase 7 - Issue #180)
    // THE GPU IS THE COMPUTER - debug output via ring buffer
    // Lock-free writes include thread ID for multi-thread debugging
    // ═══════════════════════════════════════════════════════════════════════════

    /// Debug print i32 value from register
    pub fn dbg_i32(&mut self, src_reg: u8) {
        self.asm.dbg_i32(src_reg);
    }

    /// Debug print f32 value from register
    pub fn dbg_f32(&mut self, src_reg: u8) {
        self.asm.dbg_f32(src_reg);
    }

    /// Debug print string from memory (ptr_reg = pointer, len_reg = length)
    pub fn dbg_str(&mut self, ptr_reg: u8, len_reg: u8) {
        self.asm.dbg_str(ptr_reg, len_reg);
    }

    /// Debug print bool value from register
    pub fn dbg_bool(&mut self, src_reg: u8) {
        self.asm.dbg_bool(src_reg);
    }

    /// Debug newline marker
    pub fn dbg_nl(&mut self) {
        self.asm.dbg_nl();
    }

    /// Debug flush marker
    pub fn dbg_flush(&mut self) {
        self.asm.dbg_flush();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOMATIC CODE TRANSFORMATION OPERATIONS (Phase 8 - Issue #182)
    // THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
    // ═══════════════════════════════════════════════════════════════════════════

    /// Push work item to queue (async/await transformation)
    pub fn work_push(&mut self, item_reg: u8, queue_reg: u8) {
        self.asm.work_push(item_reg, queue_reg);
    }

    /// Pop work item from queue (async/await transformation)
    pub fn work_pop(&mut self, dst: u8, queue_reg: u8) {
        self.asm.work_pop(dst, queue_reg);
    }

    /// Threadgroup barrier (Condvar::wait transformation)
    pub fn barrier(&mut self) {
        self.asm.barrier();
    }

    /// Frame-based timing (thread::sleep transformation)
    pub fn frame_wait(&mut self, frames_reg: u8) {
        self.asm.frame_wait(frames_reg);
    }

    /// Acquire spinlock (Mutex::lock transformation)
    pub fn spinlock(&mut self, lock_reg: u8) {
        self.asm.spinlock(lock_reg);
    }

    /// Release spinlock (Mutex::unlock transformation)
    pub fn spinunlock(&mut self, lock_reg: u8) {
        self.asm.spinunlock(lock_reg);
    }

    /// Atomic increment for Rc::clone
    pub fn rc_clone(&mut self, dst: u8, refcount_addr_reg: u8) {
        self.asm.rc_clone(dst, refcount_addr_reg);
    }

    /// Atomic decrement for Rc::drop
    pub fn rc_drop(&mut self, dst: u8, refcount_addr_reg: u8) {
        self.asm.rc_drop(dst, refcount_addr_reg);
    }

    /// Queue I/O request
    pub fn request_queue(&mut self, type_reg: u8, data_reg: u8) {
        self.asm.request_queue(type_reg, data_reg);
    }

    /// Poll I/O request status
    pub fn request_poll(&mut self, dst: u8, id_reg: u8) {
        self.asm.request_poll(dst, id_reg);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDERING INTRINSICS (Phase 9 - GPU App Framework)
    // THE GPU IS THE COMPUTER - emit graphics primitives from WASM apps
    // ═══════════════════════════════════════════════════════════════════════════

    /// Set X component of register to immediate value
    pub fn setx(&mut self, dst: u8, val: f32) {
        self.asm.setx(dst, val);
    }

    /// Set Z component of register to immediate value
    pub fn setz(&mut self, dst: u8, val: f32) {
        self.asm.setz(dst, val);
    }

    /// Set W component of register to immediate value
    pub fn setw(&mut self, dst: u8, val: f32) {
        self.asm.setw(dst, val);
    }

    /// Pack two scalar .x values into dst.xy: dst.xy = (s1.x, s2.x)
    /// Used for building float2 position/size from separate registers
    pub fn pack2(&mut self, dst: u8, s1: u8, s2: u8) {
        self.asm.pack2(dst, s1, s2);
    }

    /// Emit a quad (2 triangles, 6 vertices)
    pub fn quad(&mut self, pos_reg: u8, size_reg: u8, color_reg: u8, depth: f32) {
        self.asm.quad(pos_reg, size_reg, color_reg, depth);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // WASI OPERATIONS (Issue #207 - GPU-Native WASI)
    // THE GPU IS THE COMPUTER - WASI system calls on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// fd_write(fd_reg, iovs_reg, iovs_len_reg, nwritten_reg) -> errno in dst
    pub fn wasi_fd_write(&mut self, dst: u8, fd_reg: u8, iovs_reg: u8, nwritten_reg: u8) {
        self.asm.wasi_fd_write(dst, fd_reg, iovs_reg, nwritten_reg);
    }

    /// fd_read -> always returns EBADF
    pub fn wasi_fd_read(&mut self, dst: u8, fd_reg: u8, iovs_reg: u8, nread_reg: u8) {
        self.asm.wasi_fd_read(dst, fd_reg, iovs_reg, nread_reg);
    }

    /// proc_exit(code_reg) - halts execution
    pub fn wasi_proc_exit(&mut self, code_reg: u8) {
        self.asm.wasi_proc_exit(code_reg);
    }

    /// environ_sizes_get -> returns 0,0
    pub fn wasi_environ_sizes_get(&mut self, dst: u8, count_reg: u8, size_reg: u8) {
        self.asm.wasi_environ_sizes_get(dst, count_reg, size_reg);
    }

    /// environ_get -> returns success
    pub fn wasi_environ_get(&mut self, dst: u8, environ_reg: u8, buf_reg: u8) {
        self.asm.wasi_environ_get(dst, environ_reg, buf_reg);
    }

    /// args_sizes_get -> returns 0,0
    pub fn wasi_args_sizes_get(&mut self, dst: u8, count_reg: u8, size_reg: u8) {
        self.asm.wasi_args_sizes_get(dst, count_reg, size_reg);
    }

    /// args_get -> returns success
    pub fn wasi_args_get(&mut self, dst: u8, argv_reg: u8, buf_reg: u8) {
        self.asm.wasi_args_get(dst, argv_reg, buf_reg);
    }

    /// clock_time_get -> returns frame count as time
    pub fn wasi_clock_time_get(&mut self, dst: u8, clock_reg: u8, time_reg: u8) {
        self.asm.wasi_clock_time_get(dst, clock_reg, time_reg);
    }

    /// random_get -> pseudo-random from thread ID
    pub fn wasi_random_get(&mut self, dst: u8, buf_reg: u8, len_reg: u8) {
        self.asm.wasi_random_get(dst, buf_reg, len_reg);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PANIC HANDLING (Issue #209 - GPU-Native Panic)
    // THE GPU IS THE COMPUTER - panic handling on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// panic(msg_ptr_reg, msg_len_reg) - write message to debug buffer and halt
    pub fn panic(&mut self, msg_ptr_reg: u8, msg_len_reg: u8) {
        self.asm.panic(msg_ptr_reg, msg_len_reg);
    }

    /// unreachable() - halt with unreachable trap
    pub fn unreachable(&mut self) {
        self.asm.unreachable();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RECURSION SUPPORT (Issue #208 - GPU-Native Recursion)
    // THE GPU IS THE COMPUTER - function calls via GPU call stack
    // ═══════════════════════════════════════════════════════════════════════════

    /// call_func_label(label) - push return address, jump to function at label
    pub fn call_func_label(&mut self, label: usize) {
        let pc = self.asm.call_func(0);  // Placeholder
        self.patches.push((pc, label));
    }

    /// return_func() - pop return address, return to caller
    pub fn return_func(&mut self) {
        self.asm.return_func();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TABLE OPERATIONS (Issue #212 - GPU-Native Table Operations)
    // THE GPU IS THE COMPUTER - tables are GPU-resident arrays with O(1) lookup
    // ═══════════════════════════════════════════════════════════════════════════

    /// table_get(dst, idx_reg, table_idx) - get funcref from table
    pub fn table_get(&mut self, dst: u8, idx_reg: u8, table_idx: u32) {
        self.asm.table_get(dst, idx_reg, table_idx);
    }

    /// table_set(idx_reg, val_reg, table_idx) - set funcref in table
    pub fn table_set(&mut self, idx_reg: u8, val_reg: u8, table_idx: u32) {
        self.asm.table_set(idx_reg, val_reg, table_idx);
    }

    /// table_size(dst, table_idx) - get current table size
    pub fn table_size(&mut self, dst: u8, table_idx: u32) {
        self.asm.table_size(dst, table_idx);
    }

    /// table_grow(dst, delta_reg, init_reg, table_idx) - grow table
    pub fn table_grow(&mut self, dst: u8, delta_reg: u8, init_reg: u8, table_idx: u32) {
        self.asm.table_grow(dst, delta_reg, init_reg, table_idx);
    }

    /// table_fill(dst_reg, val_reg, count_reg, table_idx) - fill table
    pub fn table_fill(&mut self, dst_reg: u8, val_reg: u8, count_reg: u8, table_idx: u32) {
        self.asm.table_fill(dst_reg, val_reg, count_reg, table_idx);
    }

    /// table_copy(dst_reg, src_reg, count_reg, dst_table, src_table) - copy between tables
    pub fn table_copy(&mut self, dst_reg: u8, src_reg: u8, count_reg: u8, dst_table: u32, src_table: u32) {
        self.asm.table_copy(dst_reg, src_reg, count_reg, dst_table, src_table);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONTROL FLOW (with label support)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Jump to label (will be patched)
    pub fn jmp_label(&mut self, label: usize) {
        let pc = self.asm.jmp(0);  // Placeholder
        self.patches.push((pc, label));
    }

    /// Jump if zero to label
    pub fn jz_label(&mut self, cond: u8, label: usize) {
        let pc = self.asm.jz(cond, 0);
        self.patches.push((pc, label));
    }

    /// Jump if not zero to label
    pub fn jnz_label(&mut self, cond: u8, label: usize) {
        let pc = self.asm.jnz(cond, 0);
        self.patches.push((pc, label));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FINALIZE
    // ═══════════════════════════════════════════════════════════════════════════

    /// Finish and resolve all labels
    pub fn finish(mut self, vertex_budget: u32) -> Vec<u8> {
        // Patch all jump targets
        for (pc, label) in &self.patches {
            if let Some(&target) = self.labels.get(label) {
                self.asm.patch_jump(*pc, target);
            }
        }

        self.asm.build(vertex_budget)
    }
}

impl Default for Emitter {
    fn default() -> Self {
        Self::new()
    }
}
