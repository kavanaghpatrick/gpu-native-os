# PRD Phase 1: Integer Operations for GPU Bytecode VM

## THE GPU IS THE COMPUTER

**Issue**: Bytecode VM Integer Operations
**Phase**: 1 of 5
**Duration**: 2 weeks
**Depends On**: Existing bytecode VM (PRD_GPU_BYTECODE_VM.md)
**Enables**: WASM translation, Rust compilation

---

## Problem Statement

The current bytecode VM is float4-centric. Real programs (especially compiled from Rust/WASM) use integers heavily:
- Array indexing
- Loop counters
- Bit manipulation
- Enum discriminants
- Pointer arithmetic (as indices)

Without integer operations, we cannot compile Rust or translate WASM to our bytecode.

---

## Goals

1. Add integer arithmetic opcodes (add, sub, mul, div, rem)
2. Add bitwise opcodes (and, or, xor, not, shifts)
3. Add integer comparison opcodes
4. Maintain GPU efficiency (no SIMD divergence from type checking)
5. Enable WASM i32/i64 operations to map to our bytecode

---

## Non-Goals

- Floating-point changes (already complete)
- 128-bit integers (not needed for WASM MVP)
- Overflow trapping (use wrapping semantics like WASM)

---

## Technical Design

### Register Interpretation

Registers remain `float4`. We reinterpret `.x` component as integer when needed:

```metal
// Register is always float4
thread float4 regs[32];

// Integer operations reinterpret bits
uint as_uint(float f) { return as_type<uint>(f); }
int as_int(float f) { return as_type<int>(f); }
float from_uint(uint u) { return as_type<float>(u); }
float from_int(int i) { return as_type<float>(i); }
```

**Why**: Avoids adding separate integer registers. GPU hardware treats float4 and int4 similarly. Bit reinterpretation is free (no conversion).

### New Opcodes (0xC0-0xDF range)

```
Integer Arithmetic (0xC0-0xC9):
  0xC0  INT_ADD   dst, src1, src2    ; dst.x = src1.x + src2.x (as int32)
  0xC1  INT_SUB   dst, src1, src2    ; dst.x = src1.x - src2.x
  0xC2  INT_MUL   dst, src1, src2    ; dst.x = src1.x * src2.x
  0xC3  INT_DIV_S dst, src1, src2    ; dst.x = src1.x / src2.x (signed)
  0xC4  INT_DIV_U dst, src1, src2    ; dst.x = src1.x / src2.x (unsigned)
  0xC5  INT_REM_S dst, src1, src2    ; dst.x = src1.x % src2.x (signed)
  0xC6  INT_REM_U dst, src1, src2    ; dst.x = src1.x % src2.x (unsigned)
  0xC7  INT_NEG   dst, src1          ; dst.x = -src1.x

Bitwise (0xCA-0xCF):
  0xCA  BIT_AND   dst, src1, src2    ; dst.x = src1.x & src2.x
  0xCB  BIT_OR    dst, src1, src2    ; dst.x = src1.x | src2.x
  0xCC  BIT_XOR   dst, src1, src2    ; dst.x = src1.x ^ src2.x
  0xCD  BIT_NOT   dst, src1          ; dst.x = ~src1.x
  0xCE  SHL       dst, src1, src2    ; dst.x = src1.x << src2.x
  0xCF  SHR_U     dst, src1, src2    ; dst.x = src1.x >> src2.x (logical)

Shifts & Rotates (0xD0-0xD3):
  0xD0  SHR_S     dst, src1, src2    ; dst.x = src1.x >> src2.x (arithmetic)
  0xD1  ROTL      dst, src1, src2    ; dst.x = rotate_left(src1.x, src2.x)
  0xD2  ROTR      dst, src1, src2    ; dst.x = rotate_right(src1.x, src2.x)
  0xD3  CLZ       dst, src1          ; dst.x = count_leading_zeros(src1.x)

Integer Comparison (0xD4-0xD9):
  0xD4  INT_EQ    dst, src1, src2    ; dst.x = (src1.x == src2.x) ? 1 : 0
  0xD5  INT_NE    dst, src1, src2    ; dst.x = (src1.x != src2.x) ? 1 : 0
  0xD6  INT_LT_S  dst, src1, src2    ; dst.x = (src1.x < src2.x) ? 1 : 0 (signed)
  0xD7  INT_LT_U  dst, src1, src2    ; dst.x = (src1.x < src2.x) ? 1 : 0 (unsigned)
  0xD8  INT_LE_S  dst, src1, src2    ; dst.x = (src1.x <= src2.x) ? 1 : 0 (signed)
  0xD9  INT_LE_U  dst, src1, src2    ; dst.x = (src1.x <= src2.x) ? 1 : 0 (unsigned)

Conversion (0xDA-0xDD):
  0xDA  INT_TO_F  dst, src1          ; dst.x = float(src1.x as int)
  0xDB  UINT_TO_F dst, src1          ; dst.x = float(src1.x as uint)
  0xDC  F_TO_INT  dst, src1          ; dst.x = int(src1.x)
  0xDD  F_TO_UINT dst, src1          ; dst.x = uint(src1.x)

Load Immediate Integer (0xDE-0xDF):
  0xDE  LOADI_INT dst, imm32         ; dst.x = imm32 (as int bits)
  0xDF  LOADI_UINT dst, imm32        ; dst.x = imm32 (as uint bits)
```

### Instruction Encoding

Same 64-bit format as existing opcodes:

```
┌─────────┬───────┬───────┬───────┬───────┬─────────────────────────────────┐
│ opcode  │  dst  │ src1  │ src2  │ flags │           immediate             │
│ 8 bits  │ 5 bits│ 5 bits│ 5 bits│ 5 bits│           32 bits               │
└─────────┴───────┴───────┴───────┴───────┴─────────────────────────────────┘
```

For `LOADI_INT`/`LOADI_UINT`, the immediate holds the integer value.

### Metal Shader Implementation

```metal
// In bytecode_vm.metal, add to the switch statement:

case 0xC0: { // INT_ADD
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a + b);
    break;
}

case 0xC1: { // INT_SUB
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a - b);
    break;
}

case 0xC2: { // INT_MUL
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a * b);
    break;
}

case 0xC3: { // INT_DIV_S (signed)
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(b != 0 ? a / b : 0);
    break;
}

case 0xC4: { // INT_DIV_U (unsigned)
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(b != 0 ? a / b : 0u);
    break;
}

case 0xC5: { // INT_REM_S (signed)
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(b != 0 ? a % b : 0);
    break;
}

case 0xC6: { // INT_REM_U (unsigned)
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(b != 0 ? a % b : 0u);
    break;
}

case 0xC7: { // INT_NEG
    int a = as_type<int>(regs[src1].x);
    regs[dst].x = as_type<float>(-a);
    break;
}

case 0xCA: { // BIT_AND
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(a & b);
    break;
}

case 0xCB: { // BIT_OR
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(a | b);
    break;
}

case 0xCC: { // BIT_XOR
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(a ^ b);
    break;
}

case 0xCD: { // BIT_NOT
    uint a = as_type<uint>(regs[src1].x);
    regs[dst].x = as_type<float>(~a);
    break;
}

case 0xCE: { // SHL
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x) & 31u; // Mask to valid range
    regs[dst].x = as_type<float>(a << b);
    break;
}

case 0xCF: { // SHR_U (logical)
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x) & 31u;
    regs[dst].x = as_type<float>(a >> b);
    break;
}

case 0xD0: { // SHR_S (arithmetic)
    int a = as_type<int>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x) & 31u;
    regs[dst].x = as_type<float>(a >> b);
    break;
}

case 0xD1: { // ROTL
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x) & 31u;
    regs[dst].x = as_type<float>((a << b) | (a >> (32u - b)));
    break;
}

case 0xD2: { // ROTR
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x) & 31u;
    regs[dst].x = as_type<float>((a >> b) | (a << (32u - b)));
    break;
}

case 0xD3: { // CLZ
    uint a = as_type<uint>(regs[src1].x);
    regs[dst].x = as_type<float>(clz(a));
    break;
}

case 0xD4: { // INT_EQ
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a == b ? 1u : 0u);
    break;
}

case 0xD5: { // INT_NE
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a != b ? 1u : 0u);
    break;
}

case 0xD6: { // INT_LT_S
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a < b ? 1u : 0u);
    break;
}

case 0xD7: { // INT_LT_U
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(a < b ? 1u : 0u);
    break;
}

case 0xD8: { // INT_LE_S
    int a = as_type<int>(regs[src1].x);
    int b = as_type<int>(regs[src2].x);
    regs[dst].x = as_type<float>(a <= b ? 1u : 0u);
    break;
}

case 0xD9: { // INT_LE_U
    uint a = as_type<uint>(regs[src1].x);
    uint b = as_type<uint>(regs[src2].x);
    regs[dst].x = as_type<float>(a <= b ? 1u : 0u);
    break;
}

case 0xDA: { // INT_TO_F
    int a = as_type<int>(regs[src1].x);
    regs[dst].x = float(a);
    break;
}

case 0xDB: { // UINT_TO_F
    uint a = as_type<uint>(regs[src1].x);
    regs[dst].x = float(a);
    break;
}

case 0xDC: { // F_TO_INT
    float a = regs[src1].x;
    regs[dst].x = as_type<float>(int(a));
    break;
}

case 0xDD: { // F_TO_UINT
    float a = regs[src1].x;
    regs[dst].x = as_type<float>(uint(a));
    break;
}

case 0xDE: { // LOADI_INT
    regs[dst].x = as_type<float>(int(imm));
    break;
}

case 0xDF: { // LOADI_UINT
    regs[dst].x = as_type<float>(imm);
    break;
}
```

### Rust Assembler Updates

```rust
// In bytecode_assembler.rs

// Opcode constants
pub const INT_ADD: u8 = 0xC0;
pub const INT_SUB: u8 = 0xC1;
pub const INT_MUL: u8 = 0xC2;
pub const INT_DIV_S: u8 = 0xC3;
pub const INT_DIV_U: u8 = 0xC4;
pub const INT_REM_S: u8 = 0xC5;
pub const INT_REM_U: u8 = 0xC6;
pub const INT_NEG: u8 = 0xC7;
pub const BIT_AND: u8 = 0xCA;
pub const BIT_OR: u8 = 0xCB;
pub const BIT_XOR: u8 = 0xCC;
pub const BIT_NOT: u8 = 0xCD;
pub const SHL: u8 = 0xCE;
pub const SHR_U: u8 = 0xCF;
pub const SHR_S: u8 = 0xD0;
pub const ROTL: u8 = 0xD1;
pub const ROTR: u8 = 0xD2;
pub const CLZ: u8 = 0xD3;
pub const INT_EQ: u8 = 0xD4;
pub const INT_NE: u8 = 0xD5;
pub const INT_LT_S: u8 = 0xD6;
pub const INT_LT_U: u8 = 0xD7;
pub const INT_LE_S: u8 = 0xD8;
pub const INT_LE_U: u8 = 0xD9;
pub const INT_TO_F: u8 = 0xDA;
pub const UINT_TO_F: u8 = 0xDB;
pub const F_TO_INT: u8 = 0xDC;
pub const F_TO_UINT: u8 = 0xDD;
pub const LOADI_INT: u8 = 0xDE;
pub const LOADI_UINT: u8 = 0xDF;

impl BytecodeAssembler {
    // Integer arithmetic
    pub fn int_add(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_ADD, dst, src1, src2, 0, 0);
    }

    pub fn int_sub(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_SUB, dst, src1, src2, 0, 0);
    }

    pub fn int_mul(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_MUL, dst, src1, src2, 0, 0);
    }

    pub fn int_div_s(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_DIV_S, dst, src1, src2, 0, 0);
    }

    pub fn int_div_u(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_DIV_U, dst, src1, src2, 0, 0);
    }

    pub fn int_rem_s(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_REM_S, dst, src1, src2, 0, 0);
    }

    pub fn int_rem_u(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_REM_U, dst, src1, src2, 0, 0);
    }

    pub fn int_neg(&mut self, dst: u8, src1: u8) {
        self.emit(INT_NEG, dst, src1, 0, 0, 0);
    }

    // Bitwise
    pub fn bit_and(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(BIT_AND, dst, src1, src2, 0, 0);
    }

    pub fn bit_or(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(BIT_OR, dst, src1, src2, 0, 0);
    }

    pub fn bit_xor(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(BIT_XOR, dst, src1, src2, 0, 0);
    }

    pub fn bit_not(&mut self, dst: u8, src1: u8) {
        self.emit(BIT_NOT, dst, src1, 0, 0, 0);
    }

    pub fn shl(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(SHL, dst, src1, src2, 0, 0);
    }

    pub fn shr_u(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(SHR_U, dst, src1, src2, 0, 0);
    }

    pub fn shr_s(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(SHR_S, dst, src1, src2, 0, 0);
    }

    // Load immediate integer
    pub fn loadi_int(&mut self, dst: u8, value: i32) {
        self.emit(LOADI_INT, dst, 0, 0, 0, value as u32);
    }

    pub fn loadi_uint(&mut self, dst: u8, value: u32) {
        self.emit(LOADI_UINT, dst, 0, 0, 0, value);
    }

    // Comparison
    pub fn int_eq(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_EQ, dst, src1, src2, 0, 0);
    }

    pub fn int_lt_s(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_LT_S, dst, src1, src2, 0, 0);
    }

    pub fn int_lt_u(&mut self, dst: u8, src1: u8, src2: u8) {
        self.emit(INT_LT_U, dst, src1, src2, 0, 0);
    }

    // Conversion
    pub fn int_to_f(&mut self, dst: u8, src1: u8) {
        self.emit(INT_TO_F, dst, src1, 0, 0, 0);
    }

    pub fn f_to_int(&mut self, dst: u8, src1: u8) {
        self.emit(F_TO_INT, dst, src1, 0, 0, 0);
    }
}
```

---

## Test Cases

### Test File: `tests/test_phase1_integer_ops.rs`

```rust
//! Phase 1: Integer Operations Tests
//!
//! THE GPU IS THE COMPUTER.
//! All computation happens on GPU, including integer math.

use metal::Device;
use rust_experiment::gpu_os::bytecode_vm::{BytecodeVM, BytecodeAssembler};

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGER ARITHMETIC TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_int_add_positive() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 100);      // r8 = 100
    asm.loadi_int(9, 200);      // r9 = 200
    asm.int_add(10, 8, 9);      // r10 = r8 + r9
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, 300);
}

#[test]
fn test_int_add_negative() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, -50);
    asm.loadi_int(9, 30);
    asm.int_add(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, -20);
}

#[test]
fn test_int_add_overflow_wraps() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, i32::MAX);
    asm.loadi_int(9, 1);
    asm.int_add(10, 8, 9);  // Should wrap to i32::MIN
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, i32::MIN);  // Wrapping behavior
}

#[test]
fn test_int_sub() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 100);
    asm.loadi_int(9, 30);
    asm.int_sub(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, 70);
}

#[test]
fn test_int_mul() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 7);
    asm.loadi_int(9, 6);
    asm.int_mul(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, 42);
}

#[test]
fn test_int_div_signed() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, -100);
    asm.loadi_int(9, 7);
    asm.int_div_s(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, -14);  // Truncates toward zero
}

#[test]
fn test_int_div_unsigned() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0xFFFFFFFF);  // Max u32
    asm.loadi_uint(9, 2);
    asm.int_div_u(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0x7FFFFFFF);
}

#[test]
fn test_int_div_by_zero_returns_zero() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 100);
    asm.loadi_int(9, 0);
    asm.int_div_s(10, 8, 9);  // Division by zero
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, 0);  // Returns 0 instead of crashing
}

#[test]
fn test_int_rem() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 17);
    asm.loadi_int(9, 5);
    asm.int_rem_s(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, 2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BITWISE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bit_and() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0b11110000);
    asm.loadi_uint(9, 0b10101010);
    asm.bit_and(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0b10100000);
}

#[test]
fn test_bit_or() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0b11110000);
    asm.loadi_uint(9, 0b00001111);
    asm.bit_or(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0b11111111);
}

#[test]
fn test_bit_xor() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0b11110000);
    asm.loadi_uint(9, 0b11001100);
    asm.bit_xor(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0b00111100);
}

#[test]
fn test_bit_not() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0x00000000);
    asm.bit_not(10, 8);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0xFFFFFFFF);
}

#[test]
fn test_shl() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 1);
    asm.loadi_uint(9, 4);
    asm.shl(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 16);  // 1 << 4 = 16
}

#[test]
fn test_shr_logical() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0x80000000);  // High bit set
    asm.loadi_uint(9, 4);
    asm.shr_u(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 0x08000000);  // Logical: zeros shifted in
}

#[test]
fn test_shr_arithmetic() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, -16);  // 0xFFFFFFF0
    asm.loadi_uint(9, 2);
    asm.shr_s(10, 8, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_int(10);
    assert_eq!(r10, -4);  // Arithmetic: sign bit preserved
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPARISON TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_int_eq() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 42);
    asm.loadi_int(9, 42);
    asm.int_eq(10, 8, 9);
    asm.loadi_int(11, 43);
    asm.int_eq(12, 8, 11);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(10), 1);  // Equal
    assert_eq!(vm.read_register_uint(12), 0);  // Not equal
}

#[test]
fn test_int_lt_signed() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, -5);
    asm.loadi_int(9, 5);
    asm.int_lt_s(10, 8, 9);  // -5 < 5 = true
    asm.int_lt_s(11, 9, 8);  // 5 < -5 = false
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(10), 1);
    assert_eq!(vm.read_register_uint(11), 0);
}

#[test]
fn test_int_lt_unsigned() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0xFFFFFFFF);  // Max u32
    asm.loadi_uint(9, 5);
    asm.int_lt_u(10, 8, 9);  // 0xFFFFFFFF < 5 = false (unsigned)
    asm.int_lt_u(11, 9, 8);  // 5 < 0xFFFFFFFF = true
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(10), 0);
    assert_eq!(vm.read_register_uint(11), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVERSION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_int_to_float() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi_int(8, 42);
    asm.int_to_f(10, 8);
    asm.loadi_int(9, -100);
    asm.int_to_f(11, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_float(10);
    let r11 = vm.read_register_float(11);
    assert!((r10 - 42.0).abs() < 0.001);
    assert!((r11 - (-100.0)).abs() < 0.001);
}

#[test]
fn test_float_to_int() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();
    asm.loadi(8, 42.7);
    asm.f_to_int(10, 8);
    asm.loadi(9, -100.9);
    asm.f_to_int(11, 9);
    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_int(10), 42);   // Truncates toward zero
    assert_eq!(vm.read_register_int(11), -100); // Truncates toward zero
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOP WITH INTEGERS TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_integer_loop_sum() {
    // Sum integers 1 to 10 using a loop
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // r8 = counter (1 to 10)
    // r9 = sum
    // r10 = limit (10)
    // r11 = increment (1)
    // r12 = comparison result

    asm.loadi_int(8, 1);       // counter = 1
    asm.loadi_int(9, 0);       // sum = 0
    asm.loadi_int(10, 10);     // limit = 10
    asm.loadi_int(11, 1);      // increment = 1

    let loop_start = asm.current_offset();

    asm.int_add(9, 9, 8);      // sum += counter
    asm.int_add(8, 8, 11);     // counter += 1
    asm.int_le_s(12, 8, 10);   // r12 = (counter <= limit)
    asm.jnz(12, loop_start);   // if r12, goto loop_start

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let sum = vm.read_register_int(9);
    assert_eq!(sum, 55);  // 1+2+3+4+5+6+7+8+9+10 = 55
}
```

---

## Benchmarks

### Benchmark: Integer Throughput

```rust
// benches/bench_phase1_integer_ops.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use metal::Device;
use rust_experiment::gpu_os::bytecode_vm::{BytecodeVM, BytecodeAssembler};

fn bench_integer_ops(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device");

    let mut group = c.benchmark_group("integer_ops");

    // Benchmark: 1000 integer additions
    group.bench_function("int_add_1000", |b| {
        let vm = BytecodeVM::new(&device).expect("Failed to create VM");

        let mut asm = BytecodeAssembler::new();
        asm.loadi_int(8, 0);
        asm.loadi_int(9, 1);
        for _ in 0..1000 {
            asm.int_add(8, 8, 9);
        }
        asm.halt();

        vm.load_program(&asm.finish());

        b.iter(|| {
            vm.execute(&device);
        });
    });

    // Benchmark: 1000 shifts
    group.bench_function("shl_1000", |b| {
        let vm = BytecodeVM::new(&device).expect("Failed to create VM");

        let mut asm = BytecodeAssembler::new();
        asm.loadi_uint(8, 1);
        asm.loadi_uint(9, 1);
        for _ in 0..1000 {
            asm.shl(8, 8, 9);
            asm.bit_and(8, 8, 10); // Mask to prevent overflow
        }
        asm.halt();

        vm.load_program(&asm.finish());

        b.iter(|| {
            vm.execute(&device);
        });
    });

    // Benchmark: Mixed int/float computation
    group.bench_function("mixed_int_float_1000", |b| {
        let vm = BytecodeVM::new(&device).expect("Failed to create VM");

        let mut asm = BytecodeAssembler::new();
        asm.loadi_int(8, 0);
        asm.loadi(16, 0.0);
        for _ in 0..500 {
            asm.loadi_int(9, 1);
            asm.int_add(8, 8, 9);
            asm.int_to_f(10, 8);
            asm.add(16, 16, 10);
        }
        asm.halt();

        vm.load_program(&asm.finish());

        b.iter(|| {
            vm.execute(&device);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_integer_ops);
criterion_main!(benches);
```

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| INT_ADD | < 1 cycle | Same as float add |
| INT_MUL | < 4 cycles | May be slower than float |
| INT_DIV | < 20 cycles | Division is expensive |
| SHL/SHR | < 1 cycle | Native GPU op |
| BIT_AND/OR/XOR | < 1 cycle | Native GPU op |

---

## Success Criteria

1. **All tests pass** - Integer operations produce correct results
2. **No SIMD divergence** - Type reinterpretation doesn't cause branching
3. **WASM i32 coverage** - All WASM i32 ops have bytecode equivalents
4. **Performance parity** - Integer ops ≤ 2x slower than float ops

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| Separate integer registers | Doubles register file | Reinterpret float4 bits |
| Runtime type checking | SIMD divergence | Typed opcodes |
| Trap on overflow | GPU can't trap | Wrapping semantics |
| Trap on div-by-zero | GPU can't trap | Return 0 |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/gpu_os/shaders/bytecode_vm.metal` | Add integer opcode cases |
| `src/gpu_os/bytecode_assembler.rs` | Add assembler methods |
| `tests/test_phase1_integer_ops.rs` | Create test file |
| `benches/bench_phase1_integer_ops.rs` | Create benchmark file |

---

## Next Phase

**Phase 2: Atomic Operations** - Queue management primitives for non-blocking I/O.
