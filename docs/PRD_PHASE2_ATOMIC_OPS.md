# PRD Phase 2: Atomic Operations for GPU Command Queues

## THE GPU IS THE COMPUTER

**Issue**: Bytecode VM Atomic Operations
**Phase**: 2 of 5
**Duration**: 2 weeks
**Depends On**: Phase 1 (Integer Operations)
**Enables**: Non-blocking I/O, lock-free data structures

---

## Problem Statement

The GPU needs to manage command queues for non-blocking I/O:
- Multiple GPU threads may submit I/O requests concurrently
- Queue head/tail pointers must be updated atomically
- Completion flags must be read/written atomically

Without atomics in the bytecode VM, apps cannot safely interact with I/O queues.

---

## Architecture Context

**THE GPU NEVER WAITS. THE GPU NEVER BLOCKS.**

```
GPU App Thread                    I/O Queue (GPU-visible memory)
     │                                    │
     │ ATOMIC_ADD(&queue_tail, 1)        │
     │──────────────────────────────────>│ (claim slot)
     │                                    │
     │ STORE command to slot             │
     │──────────────────────────────────>│ (fire and forget)
     │                                    │
     │ (GPU continues other work)         │
     │                                    │
     │ ATOMIC_LOAD(&completion_flag)     │
     │<──────────────────────────────────│ (poll when convenient)
     │                                    │
```

Atomics are **not for locks**. Atomics are for **lock-free coordination**.

---

## Goals

1. Add atomic load/store opcodes
2. Add atomic read-modify-write opcodes (add, sub, max, min, and, or, xor)
3. Add atomic compare-and-swap for complex coordination
4. Support memory ordering semantics (relaxed only - GPU limitation)
5. Enable apps to safely manage I/O queues

---

## Non-Goals

- Mutex/lock primitives (GPU never blocks)
- Strong memory ordering (Metal only supports relaxed)
- 64-bit atomics (32-bit sufficient for queue indices)

---

## Technical Design

### Atomic Memory Model

Metal provides `atomic_uint` and `atomic_int` types with relaxed ordering:

```metal
// Only relaxed ordering available on Metal
atomic_fetch_add_explicit(&var, 1, memory_order_relaxed);
atomic_load_explicit(&var, memory_order_relaxed);
atomic_store_explicit(&var, val, memory_order_relaxed);
```

**Key insight**: Relaxed ordering is sufficient for queue management because:
- Each queue slot is written once, read once
- Slot index claims are atomic
- No ordering between different slots needed

### Atomic State Buffer

Atomics operate on a dedicated region of state memory:

```
State Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Regular State (float4 array)                                    │
│ [0..N]: App state                                               │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Atomic Region (uint32 array, starting at ATOMIC_BASE)           │
│ [ATOMIC_BASE + 0]: io_request_head                              │
│ [ATOMIC_BASE + 1]: io_request_tail                              │
│ [ATOMIC_BASE + 2]: io_completion_count                          │
│ [ATOMIC_BASE + 3..N]: completion flags per request              │
└─────────────────────────────────────────────────────────────────┘
```

### New Opcodes (0xE0-0xEF range)

```
Atomic Load/Store (0xE0-0xE1):
  0xE0  ATOMIC_LOAD   dst, addr_reg       ; dst.x = atomic_load(state[addr_reg.x])
  0xE1  ATOMIC_STORE  val_reg, addr_reg   ; atomic_store(state[addr_reg.x], val_reg.x)

Atomic Read-Modify-Write (0xE2-0xE9):
  0xE2  ATOMIC_ADD    dst, addr_reg, val_reg  ; dst.x = atomic_fetch_add(&state[addr], val)
  0xE3  ATOMIC_SUB    dst, addr_reg, val_reg  ; dst.x = atomic_fetch_sub(&state[addr], val)
  0xE4  ATOMIC_MAX_U  dst, addr_reg, val_reg  ; dst.x = atomic_fetch_max(&state[addr], val) (unsigned)
  0xE5  ATOMIC_MIN_U  dst, addr_reg, val_reg  ; dst.x = atomic_fetch_min(&state[addr], val) (unsigned)
  0xE6  ATOMIC_MAX_S  dst, addr_reg, val_reg  ; dst.x = atomic_fetch_max(&state[addr], val) (signed)
  0xE7  ATOMIC_MIN_S  dst, addr_reg, val_reg  ; dst.x = atomic_fetch_min(&state[addr], val) (signed)
  0xE8  ATOMIC_AND    dst, addr_reg, val_reg  ; dst.x = atomic_fetch_and(&state[addr], val)
  0xE9  ATOMIC_OR     dst, addr_reg, val_reg  ; dst.x = atomic_fetch_or(&state[addr], val)
  0xEA  ATOMIC_XOR    dst, addr_reg, val_reg  ; dst.x = atomic_fetch_xor(&state[addr], val)

Atomic Compare-and-Swap (0xEB):
  0xEB  ATOMIC_CAS    dst, addr_reg, expected_reg, desired_reg
        ; if state[addr] == expected: state[addr] = desired, dst.x = 1
        ; else: dst.x = 0

Atomic Increment/Decrement (0xEC-0xED) - convenience:
  0xEC  ATOMIC_INC    dst, addr_reg       ; dst.x = atomic_fetch_add(&state[addr], 1)
  0xED  ATOMIC_DEC    dst, addr_reg       ; dst.x = atomic_fetch_sub(&state[addr], 1)

Memory Fence (0xEE) - for multi-operation consistency:
  0xEE  MEM_FENCE                         ; threadgroup_barrier(mem_flags::mem_device)
```

### Metal Shader Implementation

```metal
// Atomic state buffer - reinterpret regular state as atomic
// This works because Metal's device memory supports atomics
#define ATOMIC_BASE 65536  // Start of atomic region in state buffer

// Helper to get atomic pointer
device atomic_uint* get_atomic_ptr(device float4* state, uint addr) {
    // Reinterpret state memory as atomic uint array
    device uint* uint_state = (device uint*)state;
    return (device atomic_uint*)&uint_state[addr];
}

// In bytecode interpreter switch:

case 0xE0: { // ATOMIC_LOAD
    uint addr = as_type<uint>(regs[src1].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint val = atomic_load_explicit(ptr, memory_order_relaxed);
    regs[dst].x = as_type<float>(val);
    break;
}

case 0xE1: { // ATOMIC_STORE
    uint addr = as_type<uint>(regs[src2].x);  // addr in src2 for store
    uint val = as_type<uint>(regs[src1].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    atomic_store_explicit(ptr, val, memory_order_relaxed);
    break;
}

case 0xE2: { // ATOMIC_ADD
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_add_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE3: { // ATOMIC_SUB
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_sub_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE4: { // ATOMIC_MAX_U
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_max_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE5: { // ATOMIC_MIN_U
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_min_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE6: { // ATOMIC_MAX_S (signed)
    uint addr = as_type<uint>(regs[src1].x);
    int val = as_type<int>(regs[src2].x);
    device atomic_int* ptr = (device atomic_int*)get_atomic_ptr(state, addr);
    int old = atomic_fetch_max_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE7: { // ATOMIC_MIN_S (signed)
    uint addr = as_type<uint>(regs[src1].x);
    int val = as_type<int>(regs[src2].x);
    device atomic_int* ptr = (device atomic_int*)get_atomic_ptr(state, addr);
    int old = atomic_fetch_min_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE8: { // ATOMIC_AND
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_and_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xE9: { // ATOMIC_OR
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_or_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xEA: { // ATOMIC_XOR
    uint addr = as_type<uint>(regs[src1].x);
    uint val = as_type<uint>(regs[src2].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_xor_explicit(ptr, val, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xEB: { // ATOMIC_CAS
    // dst = result (1 if swapped, 0 if not)
    // src1 = addr
    // src2 = expected (via flags encoding or separate instruction)
    // imm = desired (or another register)
    uint addr = as_type<uint>(regs[src1].x);
    uint expected = as_type<uint>(regs[src2].x);
    uint desired = imm;
    device atomic_uint* ptr = get_atomic_ptr(state, addr);

    bool success = atomic_compare_exchange_weak_explicit(
        ptr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    );
    regs[dst].x = as_type<float>(success ? 1u : 0u);
    break;
}

case 0xEC: { // ATOMIC_INC
    uint addr = as_type<uint>(regs[src1].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_add_explicit(ptr, 1, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xED: { // ATOMIC_DEC
    uint addr = as_type<uint>(regs[src1].x);
    device atomic_uint* ptr = get_atomic_ptr(state, addr);
    uint old = atomic_fetch_sub_explicit(ptr, 1, memory_order_relaxed);
    regs[dst].x = as_type<float>(old);
    break;
}

case 0xEE: { // MEM_FENCE
    threadgroup_barrier(mem_flags::mem_device);
    break;
}
```

### Rust Assembler Updates

```rust
// In bytecode_assembler.rs

// Atomic opcode constants
pub const ATOMIC_LOAD: u8 = 0xE0;
pub const ATOMIC_STORE: u8 = 0xE1;
pub const ATOMIC_ADD: u8 = 0xE2;
pub const ATOMIC_SUB: u8 = 0xE3;
pub const ATOMIC_MAX_U: u8 = 0xE4;
pub const ATOMIC_MIN_U: u8 = 0xE5;
pub const ATOMIC_MAX_S: u8 = 0xE6;
pub const ATOMIC_MIN_S: u8 = 0xE7;
pub const ATOMIC_AND: u8 = 0xE8;
pub const ATOMIC_OR: u8 = 0xE9;
pub const ATOMIC_XOR: u8 = 0xEA;
pub const ATOMIC_CAS: u8 = 0xEB;
pub const ATOMIC_INC: u8 = 0xEC;
pub const ATOMIC_DEC: u8 = 0xED;
pub const MEM_FENCE: u8 = 0xEE;

// Well-known atomic addresses for I/O
pub const ATOMIC_IO_REQUEST_HEAD: u32 = 0;
pub const ATOMIC_IO_REQUEST_TAIL: u32 = 1;
pub const ATOMIC_IO_COMPLETION_COUNT: u32 = 2;
pub const ATOMIC_COMPLETION_FLAGS_BASE: u32 = 16;  // Flags for slots 0-N

impl BytecodeAssembler {
    // Load/Store
    pub fn atomic_load(&mut self, dst: u8, addr_reg: u8) {
        self.emit(ATOMIC_LOAD, dst, addr_reg, 0, 0, 0);
    }

    pub fn atomic_store(&mut self, val_reg: u8, addr_reg: u8) {
        self.emit(ATOMIC_STORE, 0, val_reg, addr_reg, 0, 0);
    }

    // Read-Modify-Write
    pub fn atomic_add(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_ADD, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_sub(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_SUB, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_max_u(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_MAX_U, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_min_u(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_MIN_U, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_and(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_AND, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_or(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_OR, dst, addr_reg, val_reg, 0, 0);
    }

    pub fn atomic_xor(&mut self, dst: u8, addr_reg: u8, val_reg: u8) {
        self.emit(ATOMIC_XOR, dst, addr_reg, val_reg, 0, 0);
    }

    // Compare-and-Swap
    pub fn atomic_cas(&mut self, dst: u8, addr_reg: u8, expected_reg: u8, desired: u32) {
        self.emit(ATOMIC_CAS, dst, addr_reg, expected_reg, 0, desired);
    }

    // Convenience
    pub fn atomic_inc(&mut self, dst: u8, addr_reg: u8) {
        self.emit(ATOMIC_INC, dst, addr_reg, 0, 0, 0);
    }

    pub fn atomic_dec(&mut self, dst: u8, addr_reg: u8) {
        self.emit(ATOMIC_DEC, dst, addr_reg, 0, 0, 0);
    }

    pub fn mem_fence(&mut self) {
        self.emit(MEM_FENCE, 0, 0, 0, 0, 0);
    }

    // High-level helpers for I/O queue management

    /// Claim a slot in the I/O request queue
    /// Returns slot index in dst register
    pub fn io_claim_slot(&mut self, dst: u8) {
        // r30 = address of tail counter
        // r31 = increment value (1)
        self.loadi_uint(30, ATOMIC_IO_REQUEST_TAIL);
        self.loadi_uint(31, 1);
        self.atomic_add(dst, 30, 31);  // dst = old tail (our slot)
    }

    /// Check if an I/O request has completed
    /// Sets dst to 1 if complete, 0 otherwise
    pub fn io_poll_completion(&mut self, dst: u8, slot_reg: u8) {
        // Completion flag address = ATOMIC_COMPLETION_FLAGS_BASE + slot
        self.loadi_uint(30, ATOMIC_COMPLETION_FLAGS_BASE);
        self.int_add(30, 30, slot_reg);  // r30 = flag address
        self.atomic_load(dst, 30);       // dst = completion flag
    }
}
```

---

## Example: Non-Blocking File Load

```rust
// Bytecode for non-blocking file load
// THE GPU NEVER WAITS

fn emit_file_load_request(asm: &mut BytecodeAssembler, path_idx: u32) {
    // 1. Claim slot in request queue (atomic)
    asm.io_claim_slot(8);  // r8 = our slot index

    // 2. Write request to slot (non-atomic, we own the slot)
    // Request format: [op, path_idx, buffer_offset, size]
    asm.loadi_uint(9, IO_OP_READ);
    asm.loadi_uint(10, path_idx);
    // ... compute buffer offset and size ...

    // Calculate request address: REQUEST_BASE + slot * REQUEST_SIZE
    asm.loadi_uint(20, REQUEST_BASE);
    asm.loadi_uint(21, REQUEST_SIZE);
    asm.int_mul(22, 8, 21);       // r22 = slot * REQUEST_SIZE
    asm.int_add(20, 20, 22);      // r20 = request address

    // Store request fields
    asm.st(9, 20, 0);             // request.op = READ
    asm.st(10, 20, 1);            // request.path_idx = path_idx

    // 3. GPU continues doing other work (NO WAIT!)
    // ...

    // 4. Later, poll for completion (when convenient)
    let poll_loop = asm.current_offset();
    asm.io_poll_completion(12, 8);  // r12 = completion flag for our slot
    // If not complete, do other work or continue next frame
    // NO SPIN LOOP - just check and move on
}

fn emit_check_completion(asm: &mut BytecodeAssembler, slot_reg: u8) {
    // Non-blocking check
    asm.io_poll_completion(12, slot_reg);
    asm.jz(12, /* skip_use_data */);

    // Data is ready, use it
    // ...
}
```

---

## Test Cases

### Test File: `tests/test_phase2_atomic_ops.rs`

```rust
//! Phase 2: Atomic Operations Tests
//!
//! THE GPU IS THE COMPUTER.
//! Atomics enable lock-free queue management.

use metal::Device;
use rust_experiment::gpu_os::bytecode_vm::{BytecodeVM, BytecodeAssembler};

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC LOAD/STORE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_load_store() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Store 42 to atomic location 0
    asm.loadi_uint(8, 0);    // r8 = address (0)
    asm.loadi_uint(9, 42);   // r9 = value (42)
    asm.atomic_store(9, 8);

    // Load it back
    asm.atomic_load(10, 8);  // r10 = atomic_load(&state[0])

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let r10 = vm.read_register_uint(10);
    assert_eq!(r10, 42);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC ADD TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_add_returns_old_value() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize location 0 to 100
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 100);
    asm.atomic_store(9, 8);

    // Atomic add 5, should return old value (100)
    asm.loadi_uint(10, 5);
    asm.atomic_add(11, 8, 10);  // r11 = old value

    // Load new value
    asm.atomic_load(12, 8);    // r12 = new value

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 100);  // Old value
    assert_eq!(vm.read_register_uint(12), 105);  // New value
}

#[test]
fn test_atomic_inc() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize location 0 to 0
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 0);
    asm.atomic_store(9, 8);

    // Increment 3 times
    asm.atomic_inc(10, 8);  // r10 = 0 (old), state[0] = 1
    asm.atomic_inc(11, 8);  // r11 = 1 (old), state[0] = 2
    asm.atomic_inc(12, 8);  // r12 = 2 (old), state[0] = 3

    // Load final value
    asm.atomic_load(13, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(10), 0);
    assert_eq!(vm.read_register_uint(11), 1);
    assert_eq!(vm.read_register_uint(12), 2);
    assert_eq!(vm.read_register_uint(13), 3);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC MAX/MIN TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_max_unsigned() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize to 50
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 50);
    asm.atomic_store(9, 8);

    // Max with 30 (no change)
    asm.loadi_uint(10, 30);
    asm.atomic_max_u(11, 8, 10);

    // Max with 70 (updates to 70)
    asm.loadi_uint(10, 70);
    asm.atomic_max_u(12, 8, 10);

    // Load final value
    asm.atomic_load(13, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 50);  // Old value (unchanged)
    assert_eq!(vm.read_register_uint(12), 50);  // Old value before update
    assert_eq!(vm.read_register_uint(13), 70);  // Final value
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC BITWISE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_or_set_bits() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize to 0
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 0);
    asm.atomic_store(9, 8);

    // Set bit 0
    asm.loadi_uint(10, 0b0001);
    asm.atomic_or(11, 8, 10);

    // Set bit 2
    asm.loadi_uint(10, 0b0100);
    asm.atomic_or(12, 8, 10);

    // Load final value
    asm.atomic_load(13, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 0b0000);  // Old value
    assert_eq!(vm.read_register_uint(12), 0b0001);  // After first OR
    assert_eq!(vm.read_register_uint(13), 0b0101);  // Final (bits 0 and 2 set)
}

#[test]
fn test_atomic_and_clear_bits() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize to 0xFF
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 0xFF);
    asm.atomic_store(9, 8);

    // Clear bits 0-3
    asm.loadi_uint(10, 0xF0);  // Mask: keep bits 4-7
    asm.atomic_and(11, 8, 10);

    // Load final value
    asm.atomic_load(12, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 0xFF);  // Old value
    assert_eq!(vm.read_register_uint(12), 0xF0);  // Bits 0-3 cleared
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC CAS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_atomic_cas_success() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize to 42
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 42);
    asm.atomic_store(9, 8);

    // CAS: if value == 42, set to 100
    asm.loadi_uint(10, 42);   // expected
    asm.atomic_cas(11, 8, 10, 100);  // r11 = success (1 or 0)

    // Load final value
    asm.atomic_load(12, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 1);    // Success
    assert_eq!(vm.read_register_uint(12), 100);  // Updated
}

#[test]
fn test_atomic_cas_failure() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize to 42
    asm.loadi_uint(8, 0);
    asm.loadi_uint(9, 42);
    asm.atomic_store(9, 8);

    // CAS: if value == 99, set to 100 (will fail, value is 42)
    asm.loadi_uint(10, 99);   // expected (wrong!)
    asm.atomic_cas(11, 8, 10, 100);

    // Load final value
    asm.atomic_load(12, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 0);   // Failure
    assert_eq!(vm.read_register_uint(12), 42);  // Unchanged
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUEUE SIMULATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_queue_slot_claim() {
    // Simulate multiple "threads" claiming queue slots
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Initialize tail to 0
    asm.loadi_uint(8, ATOMIC_IO_REQUEST_TAIL);
    asm.loadi_uint(9, 0);
    asm.atomic_store(9, 8);

    // Claim 3 slots sequentially (simulating 3 threads)
    asm.io_claim_slot(10);  // r10 = slot 0
    asm.io_claim_slot(11);  // r11 = slot 1
    asm.io_claim_slot(12);  // r12 = slot 2

    // Load final tail
    asm.atomic_load(13, 8);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(10), 0);  // First got slot 0
    assert_eq!(vm.read_register_uint(11), 1);  // Second got slot 1
    assert_eq!(vm.read_register_uint(12), 2);  // Third got slot 2
    assert_eq!(vm.read_register_uint(13), 3);  // Tail advanced to 3
}

#[test]
fn test_completion_flag_poll() {
    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Set completion flag for slot 5
    asm.loadi_uint(8, ATOMIC_COMPLETION_FLAGS_BASE + 5);
    asm.loadi_uint(9, 1);  // Complete
    asm.atomic_store(9, 8);

    // Poll completion for slot 5 (should be complete)
    asm.loadi_uint(10, 5);
    asm.io_poll_completion(11, 10);

    // Poll completion for slot 6 (should NOT be complete)
    asm.loadi_uint(10, 6);
    asm.io_poll_completion(12, 10);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    assert_eq!(vm.read_register_uint(11), 1);  // Slot 5 complete
    assert_eq!(vm.read_register_uint(12), 0);  // Slot 6 not complete
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const ATOMIC_IO_REQUEST_TAIL: u32 = 1;
const ATOMIC_COMPLETION_FLAGS_BASE: u32 = 16;
```

---

## Benchmarks

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| ATOMIC_LOAD | < 50 cycles | Device memory read |
| ATOMIC_STORE | < 50 cycles | Device memory write |
| ATOMIC_ADD | < 100 cycles | Read-modify-write |
| ATOMIC_CAS | < 150 cycles | Compare + conditional write |

### Throughput Test

```rust
// benches/bench_phase2_atomic_ops.rs

fn bench_atomic_inc_throughput(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device");

    c.bench_function("atomic_inc_10000", |b| {
        let vm = BytecodeVM::new(&device).expect("Failed to create VM");

        let mut asm = BytecodeAssembler::new();
        asm.loadi_uint(8, 0);  // address
        for _ in 0..10000 {
            asm.atomic_inc(9, 8);
        }
        asm.halt();

        vm.load_program(&asm.finish());

        b.iter(|| {
            vm.execute(&device);
        });
    });
}
```

---

## Success Criteria

1. **All tests pass** - Atomics produce correct results
2. **No race conditions** - Sequential consistency within single thread
3. **Queue pattern works** - Slot claiming returns unique indices
4. **Completion polling works** - Non-blocking status checks

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| Spin locks | GPU threads can't block | Lock-free with atomics |
| Mutex acquire/release | No blocking primitives | CAS for complex state |
| Waiting for completion | GPU never waits | Poll + continue |
| Strong ordering | Metal doesn't support | Relaxed + fence if needed |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/gpu_os/shaders/bytecode_vm.metal` | Add atomic opcode cases |
| `src/gpu_os/bytecode_assembler.rs` | Add assembler methods |
| `tests/test_phase2_atomic_ops.rs` | Create test file |
| `benches/bench_phase2_atomic_ops.rs` | Create benchmark file |

---

## Next Phase

**Phase 3: GPU Kernel DSL Macro** - Write GPU apps in Rust syntax.
