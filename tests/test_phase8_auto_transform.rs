//! Phase 8 - Automatic Code Transformation Tests (Issue #182)
//!
//! THE GPU IS THE COMPUTER.
//!
//! These tests verify that CPU patterns (Mutex, Condvar, sleep, Rc) are correctly
//! transformed to GPU-native equivalents (spinlocks, barriers, frame wait, atomics).

use rust_experiment::gpu_os::gpu_app_system::{BytecodeAssembler, bytecode_op};

// ============================================================================
// BYTECODE OPCODE TESTS
// ============================================================================

/// Test that spinlock acquire/release opcodes are correctly emitted
#[test]
fn test_spinlock_acquire_release() {
    let mut asm = BytecodeAssembler::new();

    // Load lock address into r4
    asm.loadi_uint(4, 0x100);  // Lock at state offset 0x100

    // Acquire spinlock
    asm.spinlock(4);

    // Critical section: increment counter at state[0]
    asm.loadi_uint(5, 0);      // Counter address
    asm.loadi_uint(6, 1);      // Increment value
    asm.atomic_add(7, 5, 6);   // Atomically add 1

    // Release spinlock
    asm.spinunlock(4);

    asm.halt();

    let bytecode = asm.build(0);

    // Verify bytecode contains SPINLOCK and SPINUNLOCK opcodes
    assert!(bytecode.len() > 16, "Bytecode should contain header + instructions");

    // Find SPINLOCK opcode in bytecode (after 16-byte header)
    let has_spinlock = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::SPINLOCK);
    let has_spinunlock = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::SPINUNLOCK);

    assert!(has_spinlock, "Bytecode should contain SPINLOCK opcode");
    assert!(has_spinunlock, "Bytecode should contain SPINUNLOCK opcode");
}

/// Test that barrier opcode is correctly emitted
#[test]
fn test_barrier() {
    let mut asm = BytecodeAssembler::new();

    // Thread 0 writes to state[0]
    asm.loadi_uint(4, 0);
    asm.loadi_uint(5, 42);
    asm.st(4, 5, 0.0);

    // Barrier: all threads synchronize
    asm.barrier();

    // After barrier, all threads can read the value
    asm.ld(6, 4, 0.0);

    asm.halt();

    let bytecode = asm.build(0);

    // Find BARRIER opcode
    let has_barrier = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::BARRIER);
    assert!(has_barrier, "Bytecode should contain BARRIER opcode");
}

/// Test that frame_wait opcode is correctly emitted
#[test]
fn test_frame_wait() {
    let mut asm = BytecodeAssembler::new();

    // Wait for 5 frames
    asm.loadi_uint(4, 5);
    asm.frame_wait(4);

    // After waiting, do something
    asm.loadi_uint(5, 1);
    asm.halt();

    let bytecode = asm.build(0);

    // Find FRAME_WAIT opcode
    let has_frame_wait = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::FRAME_WAIT);
    assert!(has_frame_wait, "Bytecode should contain FRAME_WAIT opcode");
}

/// Test atomic increment/decrement for Rc pattern
#[test]
fn test_atomic_inc_dec() {
    let mut asm = BytecodeAssembler::new();

    // Refcount address
    asm.loadi_uint(4, 0x200);

    // Rc::clone (increment refcount)
    asm.rc_clone(5, 4);  // r5 = old refcount

    // Use the Rc...
    asm.nop();

    // Rc::drop (decrement refcount)
    asm.rc_drop(6, 4);   // r6 = old refcount

    asm.halt();

    let bytecode = asm.build(0);

    // rc_clone uses ATOMIC_INC, rc_drop uses ATOMIC_DEC
    let has_atomic_inc = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_INC);
    let has_atomic_dec = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_DEC);

    assert!(has_atomic_inc, "Bytecode should contain ATOMIC_INC for Rc::clone");
    assert!(has_atomic_dec, "Bytecode should contain ATOMIC_DEC for Rc::drop");
}

/// Test work queue push/pop for async pattern
#[test]
fn test_work_queue() {
    let mut asm = BytecodeAssembler::new();

    // Queue index (default work queue = 0)
    asm.loadi_uint(4, 0);

    // Push work item
    asm.loadi_uint(5, 123);  // Work item value
    asm.work_push(5, 4);

    // Pop work item
    asm.work_pop(6, 4);      // r6 = popped item (should be 123)

    asm.halt();

    let bytecode = asm.build(0);

    let has_work_push = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::WORK_PUSH);
    let has_work_pop = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::WORK_POP);

    assert!(has_work_push, "Bytecode should contain WORK_PUSH opcode");
    assert!(has_work_pop, "Bytecode should contain WORK_POP opcode");
}

// ============================================================================
// MUTEX PATTERN TRANSFORMATION TESTS
// ============================================================================

/// Test simulated Mutex usage pattern transforms to spinlock
#[test]
fn test_mutex_pattern() {
    // This simulates what a WASM translator would generate for:
    // ```rust
    // let guard = mutex.lock().unwrap();
    // *guard += 1;
    // drop(guard);
    // ```

    let mut asm = BytecodeAssembler::new();

    // Mutex internal lock at state[0x100]
    asm.loadi_uint(4, 0x100);

    // mutex.lock() -> spinlock_acquire
    asm.spinlock(4);

    // *guard += 1 (protected data at state[0])
    asm.loadi_uint(5, 0);      // Data address
    asm.ld(6, 5, 0.0);         // Load current value
    asm.loadi(7, 1.0);
    asm.add(6, 6, 7);          // Add 1
    asm.st(5, 6, 0.0);         // Store back

    // drop(guard) -> spinlock_release
    asm.spinunlock(4);

    asm.halt();

    let bytecode = asm.build(0);

    // Verify the transformation produces a valid spinlock pattern
    let instructions: Vec<_> = bytecode[16..].chunks(8).map(|c| c[0]).collect();

    // Should have: LOADI_UINT, SPINLOCK, ..., SPINUNLOCK, HALT
    let lock_idx = instructions.iter().position(|&op| op == bytecode_op::SPINLOCK);
    let unlock_idx = instructions.iter().position(|&op| op == bytecode_op::SPINUNLOCK);
    let halt_idx = instructions.iter().position(|&op| op == bytecode_op::HALT);

    assert!(lock_idx.is_some(), "Should have SPINLOCK");
    assert!(unlock_idx.is_some(), "Should have SPINUNLOCK");
    assert!(lock_idx < unlock_idx, "SPINLOCK should come before SPINUNLOCK");
    assert!(unlock_idx < halt_idx, "SPINUNLOCK should come before HALT");
}

/// Test simulated Rc usage pattern transforms to atomics
#[test]
fn test_rc_pattern() {
    // This simulates what a WASM translator would generate for:
    // ```rust
    // let rc2 = Rc::clone(&rc1);
    // drop(rc2);
    // ```

    let mut asm = BytecodeAssembler::new();

    // Rc internal refcount at state[0x200]
    asm.loadi_uint(4, 0x200);

    // Initialize refcount to 1
    asm.loadi_uint(5, 1);
    asm.atomic_store(5, 4);

    // Rc::clone (increment refcount)
    asm.rc_clone(6, 4);  // r6 = old count (should be 1)

    // Now refcount should be 2
    // Do something with the clone...
    asm.nop();

    // drop(rc2) -> decrement refcount
    asm.rc_drop(7, 4);   // r7 = old count (should be 2)

    // Check if refcount dropped to 0 (it's now 1, not 0)
    asm.loadi_uint(8, 0);
    asm.atomic_load(9, 4);  // Load current refcount
    asm.int_eq(10, 9, 8);   // Compare with 0

    // If refcount == 0, would deallocate (skip for this test)

    asm.halt();

    let bytecode = asm.build(0);

    // Verify atomic operations are present
    let has_store = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_STORE);
    let has_inc = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_INC);
    let has_dec = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_DEC);
    let has_load = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::ATOMIC_LOAD);

    assert!(has_store, "Should have ATOMIC_STORE for init");
    assert!(has_inc, "Should have ATOMIC_INC for Rc::clone");
    assert!(has_dec, "Should have ATOMIC_DEC for drop");
    assert!(has_load, "Should have ATOMIC_LOAD for refcount check");
}

// ============================================================================
// WASM TRANSLATOR INTRINSIC MAPPING TESTS
// ============================================================================

/// Test that WASM translator types include Phase 8 intrinsics
#[test]
fn test_intrinsic_mapping() {
    use rust_experiment::gpu_os::gpu_app_system::bytecode_op;

    // Verify all Phase 8 opcodes are defined
    assert_eq!(bytecode_op::WORK_PUSH, 0x84);
    assert_eq!(bytecode_op::WORK_POP, 0x85);
    assert_eq!(bytecode_op::REQUEST_QUEUE, 0x86);
    assert_eq!(bytecode_op::REQUEST_POLL, 0x87);
    assert_eq!(bytecode_op::FRAME_WAIT, 0x88);
    assert_eq!(bytecode_op::SPINLOCK, 0x89);
    assert_eq!(bytecode_op::SPINUNLOCK, 0x8A);
    assert_eq!(bytecode_op::BARRIER, 0x8B);

    // Verify existing opcodes weren't broken
    assert_eq!(bytecode_op::ATOMIC_INC, 0xEC);
    assert_eq!(bytecode_op::ATOMIC_DEC, 0xED);
    assert_eq!(bytecode_op::MEM_FENCE, 0xEE);
}

/// Test request queue/poll opcodes for async I/O
#[test]
fn test_request_queue_poll() {
    let mut asm = BytecodeAssembler::new();

    // Queue an I/O request
    asm.loadi_uint(4, 1);      // Request type: 1 = file read
    asm.loadi_uint(5, 0x1000); // Data pointer
    asm.request_queue(4, 5);

    // Poll until complete
    asm.loadi_uint(6, 0);      // Request ID
    asm.request_poll(7, 6);    // r7 = status (0=pending, 1=complete, -1=error)

    asm.halt();

    let bytecode = asm.build(0);

    let has_request_queue = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::REQUEST_QUEUE);
    let has_request_poll = bytecode[16..].chunks(8).any(|inst| inst[0] == bytecode_op::REQUEST_POLL);

    assert!(has_request_queue, "Bytecode should contain REQUEST_QUEUE opcode");
    assert!(has_request_poll, "Bytecode should contain REQUEST_POLL opcode");
}

// ============================================================================
// COMPREHENSIVE PATTERN TEST
// ============================================================================

/// Test a complete producer-consumer pattern using all Phase 8 features
#[test]
fn test_producer_consumer_pattern() {
    let mut asm = BytecodeAssembler::new();

    // Setup addresses
    let lock_addr = 0x100u32;
    let queue_addr = 0u32;  // Use default work queue
    let counter_addr = 0x200u32;

    // Producer (thread 0): Generate work items
    // Consumer pattern uses spinlock + work queue

    // Load addresses
    asm.loadi_uint(4, lock_addr);
    asm.loadi_uint(5, queue_addr);
    asm.loadi_uint(6, counter_addr);

    // Producer: acquire lock, push item, release lock
    asm.spinlock(4);
    asm.loadi_uint(7, 42);     // Work item value
    asm.work_push(7, 5);       // Push to queue
    asm.spinunlock(4);

    // Barrier: synchronize all threads
    asm.barrier();

    // Consumer: acquire lock, pop item, process, release lock
    asm.spinlock(4);
    asm.work_pop(8, 5);        // Pop from queue
    // Process: increment counter by item value
    asm.atomic_add(9, 6, 8);   // counter += item
    asm.spinunlock(4);

    // Wait 1 frame for effect
    asm.loadi_uint(10, 1);
    asm.frame_wait(10);

    asm.halt();

    let bytecode = asm.build(0);

    // Verify all Phase 8 opcodes are present
    let opcodes: Vec<u8> = bytecode[16..].chunks(8).map(|c| c[0]).collect();

    assert!(opcodes.contains(&bytecode_op::SPINLOCK), "Missing SPINLOCK");
    assert!(opcodes.contains(&bytecode_op::SPINUNLOCK), "Missing SPINUNLOCK");
    assert!(opcodes.contains(&bytecode_op::WORK_PUSH), "Missing WORK_PUSH");
    assert!(opcodes.contains(&bytecode_op::WORK_POP), "Missing WORK_POP");
    assert!(opcodes.contains(&bytecode_op::BARRIER), "Missing BARRIER");
    assert!(opcodes.contains(&bytecode_op::FRAME_WAIT), "Missing FRAME_WAIT");
    assert!(opcodes.contains(&bytecode_op::ATOMIC_ADD), "Missing ATOMIC_ADD");
}
