// Persistent Runtime Tests (Issue #280)
//
// THE GPU IS THE COMPUTER. These tests verify that the persistent runtime:
// 1. Runs indefinitely without crashing (all-threads-participate pattern)
// 2. Executes multiple WASM programs concurrently
// 3. Maintains process isolation
// 4. Handles edge cases gracefully
//
// CRITICAL: This runtime uses the all-threads-participate pattern to avoid
// the ~5M iteration crash that occurred with single-thread patterns.

use metal::Device;
use std::time::Duration;

use rust_experiment::gpu_os::persistent_runtime::PersistentRuntime;

// ============================================================================
// Opcode Constants (must match Metal shader)
// ============================================================================

// Opcodes are defined for completeness; not all are used in current tests
#[allow(dead_code)] const OP_NOP: u8 = 0x00;
#[allow(dead_code)] const OP_CONST: u8 = 0x01;
#[allow(dead_code)] const OP_ADD: u8 = 0x02;
#[allow(dead_code)] const OP_SUB: u8 = 0x03;
#[allow(dead_code)] const OP_MUL: u8 = 0x04;
#[allow(dead_code)] const OP_DIV: u8 = 0x05;
#[allow(dead_code)] const OP_MOD: u8 = 0x06;
#[allow(dead_code)] const OP_AND: u8 = 0x07;
#[allow(dead_code)] const OP_OR: u8 = 0x08;
#[allow(dead_code)] const OP_XOR: u8 = 0x09;
#[allow(dead_code)] const OP_JUMP: u8 = 0x0A;
#[allow(dead_code)] const OP_JUMP_IF: u8 = 0x0B;
#[allow(dead_code)] const OP_SHL: u8 = 0x0C;
#[allow(dead_code)] const OP_SHR: u8 = 0x0D;
#[allow(dead_code)] const OP_LOAD: u8 = 0x10;
#[allow(dead_code)] const OP_STORE: u8 = 0x11;
#[allow(dead_code)] const OP_CALL: u8 = 0x20;
#[allow(dead_code)] const OP_RET: u8 = 0x21;
#[allow(dead_code)] const OP_YIELD: u8 = 0x30;
#[allow(dead_code)] const OP_EMIT_QUAD: u8 = 0x40;
#[allow(dead_code)] const OP_HALT: u8 = 0xFF;

// ============================================================================
// Helper Functions
// ============================================================================

/// Build an 8-byte instruction from components
fn instr(opcode: u8, dst: u8, src1: u8, src2: u8, imm: i32) -> [u8; 8] {
    let imm_bytes = imm.to_le_bytes();
    [opcode, dst, src1, src2, imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3]]
}

/// Build bytecode from a slice of instructions
fn build_bytecode(instrs: &[[u8; 8]]) -> Vec<u8> {
    let mut bytecode = Vec::with_capacity(instrs.len() * 8);
    for instr in instrs {
        bytecode.extend_from_slice(instr);
    }
    bytecode
}

// ============================================================================
// Core Tests
// ============================================================================

/// Test: Runtime starts with zero processes and doesn't crash
#[test]
fn test_zero_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Start with no processes
    runtime.start();

    // Let it run for a bit
    std::thread::sleep(Duration::from_millis(500));

    // Frame counter should still advance
    let frames = runtime.frame_count();
    assert!(frames > 0, "Kernel should run even with no processes");

    std::thread::sleep(Duration::from_millis(500));
    let more_frames = runtime.frame_count();
    assert!(more_frames > frames, "Kernel should keep running");

    runtime.stop();
}

/// Test: Single process spawns, executes, and halts
#[test]
fn test_single_process() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Simple bytecode: CONST r0, 42; HALT
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 42),
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).expect("Load failed");
    runtime.spawn(offset, len, 0).expect("Spawn failed");

    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0), "Process should have halted");
    assert_eq!(runtime.read_register(0, 0), Some(42), "r0 should be 42");
}

/// Test: 64 processes spawn and run concurrently
#[test]
fn test_64_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Spawn 64 processes (the maximum), each stores its index in r0
    for i in 0..64u32 {
        let bytecode = build_bytecode(&[
            instr(OP_CONST, 0, 0, 0, i as i32),
            instr(OP_HALT, 0, 0, 0, 0),
        ]);
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        runtime.spawn(offset, len, 0).unwrap();
    }

    runtime.start();
    std::thread::sleep(Duration::from_millis(1000));
    runtime.stop();

    // All 64 should complete with their index value
    for i in 0..64 {
        assert!(runtime.is_dead(i), "Process {} should have halted", i);
        assert_eq!(runtime.read_register(i, 0), Some(i as i32), "Process {} value wrong", i);
    }
}

/// Test: Process isolation - one process cannot corrupt another's memory
#[test]
fn test_process_isolation() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Process 0: Write 0xDEAD to heap[0], read back to r0
    let bytecode0 = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0xDEAD_i32),      // r0 = 0xDEAD
        instr(OP_CONST, 1, 0, 0, 0),               // r1 = 0 (address)
        instr(OP_STORE, 1, 0, 0, 0),               // heap[r1] = r0
        instr(OP_CONST, 0, 0, 0, 0),               // r0 = 0 (clear)
        instr(OP_LOAD, 0, 1, 0, 0),                // r0 = heap[r1]
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    // Process 1: Write 0xBEEF to heap[0], read back to r0
    let bytecode1 = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0xBEEF_i32),      // r0 = 0xBEEF
        instr(OP_CONST, 1, 0, 0, 0),               // r1 = 0 (address)
        instr(OP_STORE, 1, 0, 0, 0),               // heap[r1] = r0
        instr(OP_CONST, 0, 0, 0, 0),               // r0 = 0 (clear)
        instr(OP_LOAD, 0, 1, 0, 0),                // r0 = heap[r1]
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (off0, len0) = runtime.load_bytecode(&bytecode0).unwrap();
    let (off1, len1) = runtime.load_bytecode(&bytecode1).unwrap();

    runtime.spawn(off0, len0, 0).unwrap();
    runtime.spawn(off1, len1, 0).unwrap();

    runtime.start();
    std::thread::sleep(Duration::from_millis(500));
    runtime.stop();

    // Each process should see only its own value
    assert_eq!(runtime.read_register(0, 0), Some(0xDEAD), "Process 0 should see 0xDEAD");
    assert_eq!(runtime.read_register(1, 0), Some(0xBEEF), "Process 1 should see 0xBEEF");
}

/// Test: Spawn queue full error is handled gracefully
#[test]
fn test_spawn_queue_full() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0),
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    // Don't start the kernel - let spawn queue fill up
    for i in 0..16 {
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        let result = runtime.spawn(offset, len, 0);
        assert!(result.is_ok(), "Spawn {} should succeed", i);
    }

    // 17th should fail with queue full error
    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    let result = runtime.spawn(offset, len, 0);
    assert!(result.is_err(), "Spawn 17 should fail - queue full");
    assert!(result.unwrap_err().contains("full"), "Error should mention queue full");
}

/// Test: Clean shutdown
#[test]
fn test_shutdown() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Spawn an infinite loop process
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0),
        instr(OP_ADD, 0, 0, 0, 1),  // r0++
        instr(OP_JUMP, 0, 0, 0, 1), // goto instruction 1
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();
    std::thread::sleep(Duration::from_millis(500));

    let frames_before = runtime.frame_count();
    assert!(frames_before > 0, "Kernel should have run");

    runtime.stop();

    std::thread::sleep(Duration::from_millis(100));
    let frames_after = runtime.frame_count();

    // Frame counter should be stable after stop
    assert_eq!(frames_before, frames_after, "Kernel should have stopped cleanly");
}

/// REGRESSION TEST: Run 5+ million iterations without crash
/// This is the specific failure mode that crashed the computer with the old design
#[test]
fn test_5m_iterations_no_crash() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Count to 6 million (beyond the ~5M crash threshold)
    // Use SUB + JUMP_IF pattern
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0),           // r0 = 0 (counter)
        instr(OP_CONST, 1, 0, 0, 6_000_000),   // r1 = 6000000 (limit)
        instr(OP_CONST, 2, 0, 0, 1),           // r2 = 1
        instr(OP_ADD, 0, 0, 2, 0),             // r0 += r2 (counter++)
        instr(OP_SUB, 3, 1, 0, 0),             // r3 = r1 - r0 (remaining)
        instr(OP_JUMP_IF, 0, 3, 0, 3),         // if r3 != 0, goto instruction 3
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    let start = std::time::Instant::now();
    while !runtime.is_dead(0) && start.elapsed() < Duration::from_secs(120) {
        std::thread::sleep(Duration::from_millis(500));
        println!("Still running at {:?}, frames: {}", start.elapsed(), runtime.frame_count());
    }

    runtime.stop();

    assert!(runtime.is_dead(0), "Process should complete 6M iterations without crash");
    assert_eq!(runtime.read_register(0, 0), Some(6_000_000), "Counter should reach 6M");
    println!("REGRESSION TEST PASSED: 6M iterations completed in {:?}", start.elapsed());
}

// ============================================================================
// Additional Tests
// ============================================================================

/// Test: Kernel starts and stops cleanly
#[test]
fn test_start_stop() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    runtime.start();
    std::thread::sleep(Duration::from_millis(100));

    let frames = runtime.frame_count();
    assert!(frames > 0, "Kernel should have run some frames");

    runtime.stop();

    let final_frames = runtime.frame_count();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(runtime.frame_count(), final_frames, "Kernel should have stopped");
}

/// Test: Multiple processes run concurrently and produce correct results
#[test]
fn test_multiple_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Process 0: CONST r0, 100; HALT
    let bytecode0 = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 100),
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    // Process 1: CONST r0, 200; HALT
    let bytecode1 = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 200),
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (off0, len0) = runtime.load_bytecode(&bytecode0).unwrap();
    let (off1, len1) = runtime.load_bytecode(&bytecode1).unwrap();

    runtime.spawn(off0, len0, 0).unwrap();
    runtime.spawn(off1, len1, 0).unwrap();

    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0), "Process 0 should have halted");
    assert!(runtime.is_dead(1), "Process 1 should have halted");
    assert_eq!(runtime.read_register(0, 0), Some(100), "Process 0: r0 should be 100");
    assert_eq!(runtime.read_register(1, 0), Some(200), "Process 1: r0 should be 200");
}

/// Test: 32 processes run in parallel (full SIMD utilization)
#[test]
fn test_32_processes_parallel() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Each process: CONST r0, <its_index>; HALT
    for i in 0..32u32 {
        let bytecode = build_bytecode(&[
            instr(OP_CONST, 0, 0, 0, i as i32),
            instr(OP_HALT, 0, 0, 0, 0),
        ]);
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        runtime.spawn(offset, len, 0).unwrap();
    }

    runtime.start();
    std::thread::sleep(Duration::from_millis(500));
    runtime.stop();

    for i in 0..32 {
        assert!(runtime.is_dead(i), "Process {} should have halted", i);
        assert_eq!(runtime.read_register(i, 0), Some(i as i32), "Process {} r0 should be {}", i, i);
    }
}

/// Test: Bytecode pool full error
#[test]
fn test_bytecode_pool_full() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Try to fill the 16MB pool
    let large_bytecode = vec![0u8; 1024 * 1024];  // 1MB chunks

    for i in 0..16 {
        let result = runtime.load_bytecode(&large_bytecode);
        assert!(result.is_ok(), "Load {} should succeed", i);
    }

    // 17th should fail (pool is 16MB)
    let result = runtime.load_bytecode(&large_bytecode);
    assert!(result.is_err(), "Load 17 should fail - pool full");
}

// ============================================================================
// Opcode Tests
// ============================================================================

/// Test: NOP opcode does nothing
#[test]
fn test_opcode_nop() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // NOP should do nothing, r0 should remain 0
    let bytecode = build_bytecode(&[
        instr(OP_NOP, 0, 0, 0, 0),
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(0));
}

/// Test: CONST opcode sets register values
#[test]
fn test_opcode_const() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 42),      // r0 = 42
        instr(OP_CONST, 1, 0, 0, -100),    // r1 = -100
        instr(OP_CONST, 63, 0, 0, 999),    // r63 = 999 (max register)
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
    assert_eq!(runtime.read_register(0, 1), Some(-100));
    assert_eq!(runtime.read_register(0, 63), Some(999));
}

/// Test: ADD opcode
#[test]
fn test_opcode_add() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 10),     // r0 = 10
        instr(OP_CONST, 1, 0, 0, 20),     // r1 = 20
        instr(OP_ADD, 2, 0, 1, 0),        // r2 = r0 + r1
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 2), Some(30));
}

/// Test: SUB opcode
#[test]
fn test_opcode_sub() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 50),     // r0 = 50
        instr(OP_CONST, 1, 0, 0, 20),     // r1 = 20
        instr(OP_SUB, 2, 0, 1, 0),        // r2 = r0 - r1
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 2), Some(30));
}

/// Test: MUL opcode
#[test]
fn test_opcode_mul() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 7),      // r0 = 7
        instr(OP_CONST, 1, 0, 0, 6),      // r1 = 6
        instr(OP_MUL, 2, 0, 1, 0),        // r2 = r0 * r1
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 2), Some(42));
}

/// Test: DIV opcode with divide by zero protection
#[test]
fn test_opcode_div() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 100),    // r0 = 100
        instr(OP_CONST, 1, 0, 0, 10),     // r1 = 10
        instr(OP_DIV, 2, 0, 1, 0),        // r2 = r0 / r1
        instr(OP_CONST, 3, 0, 0, 0),      // r3 = 0
        instr(OP_DIV, 4, 0, 3, 0),        // r4 = r0 / r3 (divide by zero, should be 0)
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 2), Some(10), "100 / 10 = 10");
    assert_eq!(runtime.read_register(0, 4), Some(0), "Divide by zero should return 0");
}

/// Test: LOAD/STORE heap operations
#[test]
fn test_opcode_load_store() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 12345),  // r0 = 12345 (value)
        instr(OP_CONST, 1, 0, 0, 100),    // r1 = 100 (address)
        instr(OP_STORE, 1, 0, 0, 0),      // heap[r1] = r0
        instr(OP_CONST, 0, 0, 0, 0),      // r0 = 0 (clear)
        instr(OP_LOAD, 0, 1, 0, 0),       // r0 = heap[r1]
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(12345), "Should read back stored value");
}

/// Test: JUMP and JUMP_IF control flow
#[test]
fn test_opcode_jump() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Test unconditional JUMP: skip setting r0=999, go directly to r0=42
    let bytecode = build_bytecode(&[
        instr(OP_JUMP, 0, 0, 0, 2),        // 0: goto instruction 2
        instr(OP_CONST, 0, 0, 0, 999),     // 1: r0 = 999 (skipped)
        instr(OP_CONST, 0, 0, 0, 42),      // 2: r0 = 42
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42), "Should have jumped past 999");
}

/// Test: JUMP_IF conditional control flow
#[test]
fn test_opcode_jump_if() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Test JUMP_IF: if r1 != 0, jump; else fall through
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 1, 0, 0, 1),       // 0: r1 = 1 (true)
        instr(OP_JUMP_IF, 0, 1, 0, 3),     // 1: if r1 != 0, goto 3
        instr(OP_CONST, 0, 0, 0, 999),     // 2: r0 = 999 (skipped)
        instr(OP_CONST, 0, 0, 0, 42),      // 3: r0 = 42
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42), "Should have jumped past 999");
}

/// Test: Bitwise operations (AND, OR, XOR, SHL, SHR)
#[test]
fn test_opcode_bitwise() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0b1100),  // r0 = 12
        instr(OP_CONST, 1, 0, 0, 0b1010),  // r1 = 10
        instr(OP_AND, 2, 0, 1, 0),         // r2 = r0 & r1 = 0b1000 = 8
        instr(OP_OR, 3, 0, 1, 0),          // r3 = r0 | r1 = 0b1110 = 14
        instr(OP_XOR, 4, 0, 1, 0),         // r4 = r0 ^ r1 = 0b0110 = 6
        instr(OP_CONST, 5, 0, 0, 1),       // r5 = 1
        instr(OP_CONST, 6, 0, 0, 2),       // r6 = 2
        instr(OP_SHL, 7, 5, 6, 0),         // r7 = 1 << 2 = 4
        instr(OP_CONST, 8, 0, 0, 16),      // r8 = 16
        instr(OP_SHR, 9, 8, 6, 0),         // r9 = 16 >> 2 = 4
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 2), Some(8), "12 & 10 = 8");
    assert_eq!(runtime.read_register(0, 3), Some(14), "12 | 10 = 14");
    assert_eq!(runtime.read_register(0, 4), Some(6), "12 ^ 10 = 6");
    assert_eq!(runtime.read_register(0, 7), Some(4), "1 << 2 = 4");
    assert_eq!(runtime.read_register(0, 9), Some(4), "16 >> 2 = 4");
}

// ============================================================================
// Stress Tests (run with --ignored)
// ============================================================================

/// Test: Run for 60 seconds without crash
#[test]
#[ignore]  // Long test - run with `cargo test -- --ignored`
fn test_60_second_stability() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Spawn an infinite loop process
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0),
        instr(OP_ADD, 0, 0, 0, 1),    // r0++
        instr(OP_JUMP, 0, 0, 0, 1),   // goto instruction 1
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    let start = std::time::Instant::now();
    let mut last_frame = 0;

    while start.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_secs(1));

        let frame = runtime.frame_count();
        assert!(frame > last_frame, "Kernel should still be running at {:?}", start.elapsed());
        last_frame = frame;

        println!("Elapsed: {:?}, Frames: {}", start.elapsed(), frame);
    }

    runtime.stop();
    println!("Completed 60 seconds without crash");
}

/// Test: Long-running process doesn't crash
#[test]
fn test_long_running_no_crash() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Bytecode: Loop 10 million times, then halt
    let bytecode = build_bytecode(&[
        instr(OP_CONST, 0, 0, 0, 0),           // r0 = 0 (counter)
        instr(OP_CONST, 1, 0, 0, 10_000_000),  // r1 = 10M (limit)
        instr(OP_CONST, 2, 0, 0, 1),           // r2 = 1
        instr(OP_ADD, 0, 0, 2, 0),             // r0 += r2 (counter++)
        instr(OP_SUB, 3, 1, 0, 0),             // r3 = r1 - r0 (remaining)
        instr(OP_JUMP_IF, 0, 3, 0, 3),         // if r3 != 0, goto instruction 3
        instr(OP_HALT, 0, 0, 0, 0),
    ]);

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    // Wait up to 30 seconds for completion
    for _ in 0..300 {
        std::thread::sleep(Duration::from_millis(100));
        if runtime.is_dead(0) {
            break;
        }
    }

    runtime.stop();

    assert!(runtime.is_dead(0), "Process should have completed");
    assert_eq!(runtime.read_register(0, 0), Some(10_000_000), "Counter should be 10M");
}
