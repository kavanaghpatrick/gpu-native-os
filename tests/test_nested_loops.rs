//! Test nested loops with inline function
//!
//! This mimics the Mandelbrot pattern to isolate what's failing.

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type, BytecodeAssembler, bytecode_op};

#[test]
fn test_simple_nested_loops() {
    // Two nested loops: outer 0-3, inner 0-4 = 12 iterations
    // counter increments each inner loop iteration
    // Expected result: 12

    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let mut asm = BytecodeAssembler::new();

    // r4 = counter = 0
    asm.loadi_int(4, 0);                  // inst 0

    // r5 = outer_limit = 3
    asm.loadi_int(5, 3);                  // inst 1

    // r6 = outer_i = 0
    asm.loadi_int(6, 0);                  // inst 2

    // Outer loop start (instruction 3)
    let outer_start = 3;

    // Check outer_i < outer_limit
    asm.int_lt_s(8, 6, 5);                // inst 3

    // JZ to outer_end if condition is false
    let outer_jz = asm.jz(8, 0);          // inst 4

    // r7 = inner_limit = 4
    asm.loadi_int(7, 4);                  // inst 5

    // r9 = inner_i = 0
    asm.loadi_int(9, 0);                  // inst 6

    // Inner loop start (instruction 7)
    let inner_start = 7;

    // Check inner_i < inner_limit
    asm.int_lt_s(10, 9, 7);               // inst 7

    // JZ to inner_end if condition is false
    let inner_jz = asm.jz(10, 0);         // inst 8

    // counter += 1
    asm.loadi_int(11, 1);                 // inst 9
    asm.int_add(4, 4, 11);                // inst 10

    // inner_i += 1
    asm.int_add(9, 9, 11);                // inst 11

    // JMP back to inner_start
    asm.jmp(inner_start);                 // inst 12

    // Inner end (instruction 13)
    let inner_end = 13;

    // outer_i += 1
    asm.loadi_int(11, 1);                 // inst 13
    asm.int_add(6, 6, 11);                // inst 14

    // JMP back to outer_start
    asm.jmp(outer_start);                 // inst 15

    // Outer end (instruction 16)
    let outer_end = 16;

    // Store counter to state[0]
    asm.loadi_uint(30, 0);                // inst 16
    asm.st(30, 4, 0.0);                   // inst 17

    asm.halt();                           // inst 18

    // Patch jumps
    asm.patch_jump(outer_jz, outer_end);
    asm.patch_jump(inner_jz, inner_end);

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes, expected {} bytes",
             bytecode.len(), 16 + 19 * 8);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 12 = 3*4)", result);

    assert_eq!(result, 12, "Nested loops should count to 3*4 = 12");
}

#[test]
fn test_loop_with_inline_function_pattern() {
    // Simulates the Mandelbrot pattern:
    // - Outer loop (py 0 to 2)
    //   - Inner loop (px 0 to 2)
    //     - Call inline function that does work
    //     - increment counter
    // Expected result: 4 (2*2)

    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let mut asm = BytecodeAssembler::new();

    // r4 = counter = 0
    asm.loadi_int(4, 0);

    // r5 = py_limit = 2
    asm.loadi_int(5, 2);

    // r6 = py = 0
    asm.loadi_int(6, 0);

    // Outer loop start (instruction 3)
    let outer_start = 3;

    // Check py < py_limit
    asm.int_lt_s(8, 6, 5);  // r8 = (py < py_limit)
    let outer_jz = asm.jz(8, 0);  // JZ to outer_end

    // r7 = px_limit = 2
    asm.loadi_int(7, 2);

    // r9 = px = 0
    asm.loadi_int(9, 0);

    // Inner loop start (instruction 7)
    let inner_start = 7;

    // Check px < px_limit
    asm.int_lt_s(10, 9, 7);  // r10 = (px < px_limit)
    let inner_jz = asm.jz(10, 0);  // JZ to inner_end

    // Simulate inline function: compute x, y from px, py (just add them)
    // This is where the mandelbrot() function would be called
    // r11 = px + py (dummy computation)
    asm.int_add(11, 9, 6);

    // counter += 1
    asm.loadi_int(12, 1);
    asm.int_add(4, 4, 12);

    // px += 1
    asm.int_add(9, 9, 12);

    // JMP back to inner_start
    asm.jmp(inner_start);

    // Inner end (instruction 14)
    let inner_end = 14;

    // py += 1
    asm.loadi_int(12, 1);
    asm.int_add(6, 6, 12);

    // JMP back to outer_start
    asm.jmp(outer_start);

    // Outer end (instruction 17)
    let outer_end = 17;

    // Store counter to state[0]
    asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);

    asm.halt();

    // Patch jumps
    asm.patch_jump(outer_jz, outer_end);
    asm.patch_jump(inner_jz, inner_end);

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 4 = 2*2)", result);

    assert_eq!(result, 4, "Nested loops with inline pattern should count to 2*2 = 4");
}

#[test]
fn test_loop_with_third_level() {
    // Three-level nested loops like Mandelbrot's outer + inner + mandelbrot_loop
    // outer: 0-2, inner: 0-2, deepest: 0-3
    // Expected: 2 * 2 * 3 = 12

    let device = Device::system_default().expect("No Metal device found");
    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let mut asm = BytecodeAssembler::new();

    // r4 = counter = 0
    asm.loadi_int(4, 0);

    // r5 = limit1 = 2
    asm.loadi_int(5, 2);
    // r6 = limit2 = 2
    asm.loadi_int(6, 2);
    // r7 = limit3 = 3
    asm.loadi_int(7, 3);

    // r8 = i1 = 0
    asm.loadi_int(8, 0);

    // Loop 1 start (instruction 5)
    let loop1_start = 5;
    asm.int_lt_s(12, 8, 5);  // r12 = (i1 < limit1)
    let loop1_jz = asm.jz(12, 0);

    // r9 = i2 = 0
    asm.loadi_int(9, 0);

    // Loop 2 start (instruction 8)
    let loop2_start = 8;
    asm.int_lt_s(13, 9, 6);  // r13 = (i2 < limit2)
    let loop2_jz = asm.jz(13, 0);

    // r10 = i3 = 0
    asm.loadi_int(10, 0);

    // Loop 3 start (instruction 11)
    let loop3_start = 11;
    asm.int_lt_s(14, 10, 7);  // r14 = (i3 < limit3)
    let loop3_jz = asm.jz(14, 0);

    // counter += 1
    asm.loadi_int(15, 1);
    asm.int_add(4, 4, 15);

    // i3 += 1
    asm.int_add(10, 10, 15);
    asm.jmp(loop3_start);

    // Loop 3 end (instruction 17)
    let loop3_end = 17;
    // i2 += 1
    asm.loadi_int(15, 1);
    asm.int_add(9, 9, 15);
    asm.jmp(loop2_start);

    // Loop 2 end (instruction 20)
    let loop2_end = 20;
    // i1 += 1
    asm.loadi_int(15, 1);
    asm.int_add(8, 8, 15);
    asm.jmp(loop1_start);

    // Loop 1 end (instruction 23)
    let loop1_end = 23;

    // Store counter to state[0]
    asm.loadi_uint(30, 0);
    asm.st(30, 4, 0.0);
    asm.halt();

    // Patch jumps
    asm.patch_jump(loop1_jz, loop1_end);
    asm.patch_jump(loop2_jz, loop2_end);
    asm.patch_jump(loop3_jz, loop3_end);

    let bytecode = asm.build(1000);
    println!("Bytecode size: {} bytes", bytecode.len());

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 12 = 2*2*3)", result);

    assert_eq!(result, 12, "Three-level nested loops should count to 2*2*3 = 12");
}
