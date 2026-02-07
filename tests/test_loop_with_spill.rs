//! Test loop with memory spill in GPU bytecode VM

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler};

#[test]
fn test_loop_with_spill() {
    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Loop: count from 0 to 10, storing counter in state[104] (spill slot)
    let mut asm = BytecodeAssembler::new();
    
    // Initialize counter to 0 and store to state[104]
    asm.loadi_int(8, 0);
    asm.loadi_uint(30, 104);  // spill address
    asm.st(30, 8, 0.0);       // state[104] = 0
    
    // r5 = limit (10)
    asm.loadi_int(5, 10);
    
    // Loop start (instruction 4)
    let loop_start = 4;
    
    // Load counter from state[104]
    asm.loadi_uint(30, 104);
    asm.ld(8, 30, 0.0);       // r8 = state[104]
    
    // r6 = counter < limit
    asm.int_lt_s(6, 8, 5);
    
    // if r6 == 0 (counter >= limit), jump to end
    let jz_idx = asm.jz(6, 0);  // placeholder target
    
    // counter += 1 and store back
    asm.loadi_int(7, 1);
    asm.int_add(8, 8, 7);
    asm.loadi_uint(30, 104);
    asm.st(30, 8, 0.0);       // state[104] = counter
    
    // jump back to loop start
    asm.jmp(loop_start);
    
    // End: load counter and store to state[0]
    let end = asm.loadi_uint(30, 104);
    asm.ld(4, 30, 0.0);       // r4 = state[104]
    asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);       // state[0] = counter
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
    
    assert_eq!(result, 10, "Loop with spill should count to 10");
}
