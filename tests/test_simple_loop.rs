//! Test simple loop in GPU bytecode VM

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler};

#[test]
fn test_simple_loop() {
    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Simple loop: count from 0 to 10
    let mut asm = BytecodeAssembler::new();
    
    // r4 = counter (starts at 0)
    asm.loadi_int(4, 0);
    
    // r5 = limit (10)
    asm.loadi_int(5, 10);
    
    // Loop start (instruction 2)
    let loop_start = 2;
    
    // r6 = counter < limit
    asm.int_lt_s(6, 4, 5);
    
    // if r6 == 0 (counter >= limit), jump to end
    let jz_idx = asm.jz(6, 0);  // placeholder target
    
    // counter += 1
    asm.loadi_int(7, 1);
    asm.int_add(4, 4, 7);
    
    // jump back to loop start
    asm.jmp(loop_start);
    
    // End: store counter to state[0]
    let end = asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);
    asm.halt();
    
    // Patch the JZ target
    asm.patch_jump(jz_idx, end);
    
    let bytecode = asm.build(1000);
    
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 10)", result);
    
    assert_eq!(result, 10, "Loop should count to 10");
}
