//! Test denormal float behavior in GPU bytecode VM

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler};

#[test]
fn test_denormal_addresses() {
    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Test: Load address 104 into r30, store a value to state[104], load it back
    let mut asm = BytecodeAssembler::new();
    
    // Load test value 12345 into r8
    asm.loadi_int(8, 12345);
    
    // Load address 104 into r30 (104 = 0x68, which is denormal as float bits)
    asm.loadi_uint(30, 104);
    
    // Store r8 to state[r30] = state[104]
    asm.st(30, 8, 0.0);
    
    // Load back from state[104] into r9
    asm.ld(9, 30, 0.0);
    
    // Move r9 to r4 (return register)
    asm.mov(4, 9);
    
    // Store r4 to state[0] as return value
    asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);
    
    asm.halt();

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 12345)", result);
    
    if result != 12345 {
        println!("DENORMAL FLUSH DETECTED! Address 104 (0x68) was flushed to 0");
    }
    
    assert_eq!(result, 12345, "Value should survive LD/ST through address 104");
}
