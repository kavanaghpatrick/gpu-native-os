//! Test GPU-Native Recursion Support (Issue #208)
//!
//! THE GPU IS THE COMPUTER - function calls via GPU call stack

use rust_experiment::gpu_os::gpu_app_system::{
    bytecode_op, BytecodeAssembler, GpuAppSystem, app_type,
};

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_call_func_opcode_value() {
    assert_eq!(bytecode_op::CALL_FUNC, 0x78);
}

#[test]
fn test_return_func_opcode_value() {
    assert_eq!(bytecode_op::RETURN_FUNC, 0x79);
}

#[test]
fn test_recursion_opcodes_contiguous() {
    // Recursion opcodes should be contiguous
    assert_eq!(bytecode_op::RETURN_FUNC, bytecode_op::CALL_FUNC + 1);
}

#[test]
fn test_recursion_opcodes_dont_overlap_with_panic() {
    // Should be after panic opcodes (0x76-0x77)
    assert!(bytecode_op::CALL_FUNC > bytecode_op::UNREACHABLE);
    assert!(bytecode_op::RETURN_FUNC > bytecode_op::UNREACHABLE);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE ASSEMBLER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_call_func_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.call_func(42);  // Jump to PC 42

    let bytecode = asm.build(0);

    // Check instruction
    let inst_offset = 16;  // After header
    assert_eq!(bytecode[inst_offset], bytecode_op::CALL_FUNC);

    // Check target PC in immediate (stored as uint bits in f32)
    let imm_bytes = &bytecode[inst_offset + 4..inst_offset + 8];
    let imm_bits = u32::from_le_bytes([imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3]]);
    assert_eq!(imm_bits, 42);  // Target PC stored as bits
}

#[test]
fn test_return_func_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.return_func();
    
    let bytecode = asm.build(0);
    
    // Check instruction
    let inst_offset = 16;  // After header
    assert_eq!(bytecode[inst_offset], bytecode_op::RETURN_FUNC);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASIC CALL/RETURN BYTECODE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_simple_call_return_bytecode() {
    // Create bytecode that:
    // 0: loadi r4, 10        ; initial value
    // 1: call_func 4         ; call subroutine at PC 4
    // 2: loadi r30, 3        ; store to state[3]
    // 3: st r30, r4          ; save result
    // 4: halt                ; end main
    // 
    // Subroutine (PC 4):
    // 4: loadi r4, 42        ; set result
    // 5: return_func         ; return to caller
    
    let mut asm = BytecodeAssembler::new();
    
    // Main code
    asm.loadi_uint(4, 10);     // 0: initial value
    asm.call_func(5);          // 1: call subroutine at PC 5
    asm.loadi_uint(30, 3);     // 2: store index
    asm.st(30, 4, 0.0);        // 3: store result to state[3]
    asm.halt();                // 4: done
    
    // Subroutine starts at PC 5
    asm.loadi_uint(4, 42);     // 5: set result to 42
    asm.return_func();         // 6: return
    
    let bytecode = asm.build(0);
    
    // Verify we have 7 instructions
    let header = &bytecode[0..16];
    let code_size = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    assert_eq!(code_size, 7);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU EXECUTION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod gpu_tests {
    use super::*;
    use metal::Device;

    fn create_gpu_system() -> GpuAppSystem {
        let device = Device::system_default().expect("No Metal device");
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        system.set_use_parallel_megakernel(true);
        system
    }

    #[test]
    fn test_simple_call_return_gpu() {
        let mut system = create_gpu_system();
        
        // Bytecode:
        // 0: loadi r4, 10        ; initial value (will be overwritten)
        // 1: call_func 5         ; call subroutine at PC 5
        // 2: loadi r30, 3        ; store to state[3]
        // 3: st r30, r4          ; save result
        // 4: halt                ; end main
        // 
        // Subroutine (PC 5):
        // 5: loadi r4, 42        ; set result to 42
        // 6: return_func         ; return to caller
        
        let mut asm = BytecodeAssembler::new();
        asm.loadi_uint(4, 10);     // 0: initial value
        asm.call_func(5);          // 1: call subroutine at PC 5
        asm.loadi_uint(30, 3);     // 2: store index
        asm.st(30, 4, 0.0);        // 3: store result to state[3]
        asm.halt();                // 4: done
        asm.loadi_uint(4, 42);     // 5: set result to 42
        asm.return_func();         // 6: return
        
        let bytecode = asm.build(0);
        
        // Launch and execute
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        // Read result - should be 42 (from subroutine)
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 42, "Call/return should work: got {}, expected 42", result);
        
        system.close_app(slot);
    }

    #[test]
    fn test_nested_calls_gpu() {
        let mut system = create_gpu_system();
        
        // Bytecode with nested calls:
        // 0: loadi r4, 1         ; initial value
        // 1: call_func 5         ; call first level
        // 2: loadi r30, 3        ; store to state[3]
        // 3: st r30, r4          ; save result
        // 4: halt                ; end main
        // 
        // First level (PC 5):
        // 5: loadi r5, 10        ; add 10
        // 6: int_add r4, r4, r5
        // 7: call_func 10        ; call second level
        // 8: int_add r4, r4, r5  ; add another 10 after return
        // 9: return_func         ; return
        // 
        // Second level (PC 10):
        // 10: loadi r6, 100      ; add 100
        // 11: int_add r4, r4, r6
        // 12: return_func        ; return
        
        let mut asm = BytecodeAssembler::new();
        
        // Main
        asm.loadi_uint(4, 1);      // 0: r4 = 1
        asm.call_func(5);          // 1: call first level
        asm.loadi_uint(30, 3);     // 2: store index
        asm.st(30, 4, 0.0);        // 3: store result to state[3]
        asm.halt();                // 4: done
        
        // First level (PC 5)
        asm.loadi_uint(5, 10);     // 5: r5 = 10
        asm.int_add(4, 4, 5);      // 6: r4 = r4 + r5 (1 + 10 = 11)
        asm.call_func(10);         // 7: call second level
        asm.int_add(4, 4, 5);      // 8: r4 = r4 + r5 (111 + 10 = 121)
        asm.return_func();         // 9: return
        
        // Second level (PC 10)
        asm.loadi_uint(6, 100);    // 10: r6 = 100
        asm.int_add(4, 4, 6);      // 11: r4 = r4 + r6 (11 + 100 = 111)
        asm.return_func();         // 12: return
        
        let bytecode = asm.build(0);
        
        // Launch and execute
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        // Read result
        // Expected: 1 + 10 + 100 + 10 = 121
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 121, "Nested calls should work: got {}, expected 121", result);
        
        system.close_app(slot);
    }

    #[test]
    fn test_recursive_sum_gpu() {
        let mut system = create_gpu_system();

        // Tail-recursive sum using accumulator:
        // sum(n, acc) = n == 0 ? acc : sum(n-1, acc + n)
        // Call with sum(5, 0) → returns 15
        //
        // This pattern works because we don't need to save registers across calls -
        // each recursive call just updates the parameters and the return value
        // is the final accumulated result.
        //
        // Bytecode:
        // Main:
        // 0: loadi r4, 5         ; n = 5
        // 1: loadi r5, 0         ; acc = 0
        // 2: call_func 6         ; call sum(n, acc)
        // 3: loadi r30, 3        ; store to state[3]
        // 4: st r30, r4          ; save result
        // 5: halt                ; end
        //
        // sum(n, acc) at PC 6:
        // r4 = n, r5 = acc, result in r4
        // 6: loadi r6, 0         ; constant 0
        // 7: int_eq r7, r4, r6   ; r7 = (n == 0)
        // 8: jnz r7, 14          ; if n == 0, go to return acc
        // 9: int_add r5, r5, r4  ; acc = acc + n
        // 10: loadi r6, 1
        // 11: int_sub r4, r4, r6 ; n = n - 1
        // 12: call_func 6        ; tail call sum(n-1, acc+n)
        // 13: return_func        ; (unreachable with tail call, but needed for return)
        // 14: mov r4, r5         ; result = acc
        // 15: return_func        ; return

        let mut asm = BytecodeAssembler::new();

        // Main
        asm.loadi_uint(4, 5);      // 0: n = 5
        asm.loadi_uint(5, 0);      // 1: acc = 0
        asm.call_func(6);          // 2: call sum(n, acc)
        asm.loadi_uint(30, 3);     // 3: store index
        asm.st(30, 4, 0.0);        // 4: store result to state[3]
        asm.halt();                // 5: done

        // sum(n, acc) at PC 6
        asm.loadi_uint(6, 0);      // 6: constant 0
        asm.int_eq(7, 4, 6);       // 7: r7 = (n == 0)
        asm.jnz(7, 14);            // 8: if n == 0, jump to return acc
        asm.int_add(5, 5, 4);      // 9: acc = acc + n
        asm.loadi_uint(6, 1);      // 10: constant 1
        asm.int_sub(4, 4, 6);      // 11: n = n - 1
        asm.call_func(6);          // 12: tail call (will return through us)
        asm.return_func();         // 13: return (after tail call returns)
        asm.mov(4, 5);             // 14: result = acc (base case)
        asm.return_func();         // 15: return

        let bytecode = asm.build(0);

        // Launch and execute
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        // Read result
        // sum(5, 0) = 5+4+3+2+1+0 = 15
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 15, "Recursive sum(5) should be 15: got {}", result);

        system.close_app(slot);
    }
}
