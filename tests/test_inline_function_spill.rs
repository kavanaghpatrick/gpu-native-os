//! Test inline function with multiple locals that require spilling
//!
//! This test mimics the Mandelbrot pattern: an inline function with local
//! variables and a while loop that iterates.

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler};

#[test]
fn test_simple_store_load_104() {
    // Simplified test - just store to state[104] and copy to state[0]
    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let mut asm = BytecodeAssembler::new();

    // Store 12345 to state[104]
    asm.loadi_int(8, 12345);
    asm.loadi_uint(30, 104);
    asm.st(30, 8, 0.0);         // state[104] = 12345

    // Load from state[104] to r8
    asm.loadi_uint(30, 104);
    asm.ld(8, 30, 0.0);         // r8 = state[104]

    // Store r8 to state[0]
    asm.loadi_uint(30, 0);
    asm.st(30, 8, 0.0);         // state[0] = r8

    asm.halt();

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 12345)", result);

    assert_eq!(result, 12345, "Store/load to state[104] should work");
}

#[test]
fn test_multiple_spill_slots_adjacent() {
    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Test storing to adjacent spill slots and verify no corruption
    // This directly tests Grok's hypothesis about float4 overwrites

    let mut asm = BytecodeAssembler::new();

    // Store different values to adjacent slots
    asm.loadi_int(8, 111);
    asm.loadi_uint(30, 104);
    asm.st(30, 8, 0.0);         // state[104] = 111

    asm.loadi_int(8, 222);
    asm.loadi_uint(30, 105);
    asm.st(30, 8, 0.0);         // state[105] = 222

    asm.loadi_int(8, 333);
    asm.loadi_uint(30, 106);
    asm.st(30, 8, 0.0);         // state[106] = 333

    asm.loadi_int(8, 444);
    asm.loadi_uint(30, 107);
    asm.st(30, 8, 0.0);         // state[107] = 444

    // Now load them all back - each stored to a different register
    // to avoid any register reuse issues
    asm.loadi_uint(30, 104);
    asm.ld(4, 30, 0.0);         // r4 = state[104] (should be 111) -> goes to state[0]

    asm.loadi_uint(30, 105);
    asm.ld(5, 30, 0.0);         // r5 = state[105] (should be 222)

    asm.loadi_uint(30, 106);
    asm.ld(6, 30, 0.0);         // r6 = state[106] (should be 333)

    asm.loadi_uint(30, 107);
    asm.ld(7, 30, 0.0);         // r7 = state[107] (should be 444)

    // Store r4 (which should be 111) to state[0] (return value)
    asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);         // state[0] = r4 = 111

    asm.halt();

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    // Use read_bytecode_result which correctly finds state[0]
    let v0 = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("State[0] = {} (expected 111 from state[104])", v0);

    assert_eq!(v0, 111, "state[104] should be 111 after storing 111 to it");
}
