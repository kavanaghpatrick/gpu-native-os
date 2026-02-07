//! Test GPU-Native SIMD Operations (Issue #211)
//!
//! THE GPU IS THE COMPUTER - float4 is native SIMD on GPU

use rust_experiment::gpu_os::gpu_app_system::{
    bytecode_op, BytecodeAssembler, GpuAppSystem, app_type,
};

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_simd_opcodes_values() {
    assert_eq!(bytecode_op::V4_ADD, 0x90);
    assert_eq!(bytecode_op::V4_SUB, 0x91);
    assert_eq!(bytecode_op::V4_MUL, 0x92);
    assert_eq!(bytecode_op::V4_DIV, 0x93);
    assert_eq!(bytecode_op::V4_MIN, 0x94);
    assert_eq!(bytecode_op::V4_MAX, 0x95);
    assert_eq!(bytecode_op::V4_ABS, 0x96);
    assert_eq!(bytecode_op::V4_NEG, 0x97);
    assert_eq!(bytecode_op::V4_SQRT, 0x98);
    assert_eq!(bytecode_op::V4_DOT, 0x99);
    assert_eq!(bytecode_op::V4_SHUFFLE, 0x9A);
    assert_eq!(bytecode_op::V4_EXTRACT, 0x9B);
    assert_eq!(bytecode_op::V4_REPLACE, 0x9C);
    assert_eq!(bytecode_op::V4_SPLAT, 0x9D);
    assert_eq!(bytecode_op::V4_EQ, 0x9E);
    assert_eq!(bytecode_op::V4_LT, 0x9F);
}

#[test]
fn test_simd_opcodes_contiguous() {
    assert_eq!(bytecode_op::V4_SUB, bytecode_op::V4_ADD + 1);
    assert_eq!(bytecode_op::V4_MUL, bytecode_op::V4_SUB + 1);
    assert_eq!(bytecode_op::V4_DIV, bytecode_op::V4_MUL + 1);
    assert_eq!(bytecode_op::V4_MIN, bytecode_op::V4_DIV + 1);
    assert_eq!(bytecode_op::V4_MAX, bytecode_op::V4_MIN + 1);
    assert_eq!(bytecode_op::V4_ABS, bytecode_op::V4_MAX + 1);
    assert_eq!(bytecode_op::V4_NEG, bytecode_op::V4_ABS + 1);
    assert_eq!(bytecode_op::V4_SQRT, bytecode_op::V4_NEG + 1);
    assert_eq!(bytecode_op::V4_DOT, bytecode_op::V4_SQRT + 1);
    assert_eq!(bytecode_op::V4_SHUFFLE, bytecode_op::V4_DOT + 1);
    assert_eq!(bytecode_op::V4_EXTRACT, bytecode_op::V4_SHUFFLE + 1);
    assert_eq!(bytecode_op::V4_REPLACE, bytecode_op::V4_EXTRACT + 1);
    assert_eq!(bytecode_op::V4_SPLAT, bytecode_op::V4_REPLACE + 1);
    assert_eq!(bytecode_op::V4_EQ, bytecode_op::V4_SPLAT + 1);
    assert_eq!(bytecode_op::V4_LT, bytecode_op::V4_EQ + 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE ASSEMBLER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_v4_add_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.v4_add(4, 5, 6);

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::V4_ADD);
    assert_eq!(bytecode[inst_offset + 1], 4);  // dst
    assert_eq!(bytecode[inst_offset + 2], 5);  // s1
    assert_eq!(bytecode[inst_offset + 3], 6);  // s2
}

#[test]
fn test_v4_extract_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.v4_extract(4, 5, 2);  // Extract lane 2

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::V4_EXTRACT);
    assert_eq!(bytecode[inst_offset + 1], 4);  // dst
    assert_eq!(bytecode[inst_offset + 2], 5);  // s1

    // Check lane in immediate
    let imm_bytes = &bytecode[inst_offset + 4..inst_offset + 8];
    let imm_bits = u32::from_le_bytes([imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3]]);
    assert_eq!(imm_bits, 2);
}

#[test]
fn test_v4_shuffle_encoding() {
    let mut asm = BytecodeAssembler::new();
    // mask: x=1, y=2, z=0, w=3 => 0b11_00_10_01 = 0xC9
    asm.v4_shuffle(4, 5, 6, 0b11_00_10_01);

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::V4_SHUFFLE);

    // Check mask in immediate
    let imm_bytes = &bytecode[inst_offset + 4..inst_offset + 8];
    let imm_bits = u32::from_le_bytes([imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3]]);
    assert_eq!(imm_bits, 0b11_00_10_01);
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
    fn test_v4_add_gpu() {
        let mut system = create_gpu_system();

        // Set up r5 = (1.0, 2.0, 3.0, 4.0) and r6 = (10.0, 20.0, 30.0, 40.0)
        // v4_add r4 = r5 + r6 = (11.0, 22.0, 33.0, 44.0)
        // Extract lane 0 (x) and store to result

        let mut asm = BytecodeAssembler::new();
        // Set r5 = (1, 2, 3, 4)
        asm.loadi(5, 1.0);
        asm.sety(5, 2.0);
        asm.setz(5, 3.0);
        asm.setw(5, 4.0);
        // Set r6 = (10, 20, 30, 40)
        asm.loadi(6, 10.0);
        asm.sety(6, 20.0);
        asm.setz(6, 30.0);
        asm.setw(6, 40.0);
        // v4_add
        asm.v4_add(4, 5, 6);
        // Extract x (lane 0) and convert to int for result
        asm.f_to_int(4, 4);
        // Store to state[3]
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 11, "v4_add: 1+10=11, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_mul_gpu() {
        let mut system = create_gpu_system();

        // r5 = (2, 3, 4, 5), r6 = (10, 10, 10, 10)
        // v4_mul = (20, 30, 40, 50)
        // Extract lane 2 (z) = 40

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 2.0);
        asm.sety(5, 3.0);
        asm.setz(5, 4.0);
        asm.setw(5, 5.0);
        asm.loadi(6, 10.0);
        asm.sety(6, 10.0);
        asm.setz(6, 10.0);
        asm.setw(6, 10.0);
        asm.v4_mul(4, 5, 6);
        asm.v4_extract(7, 4, 2);  // Extract z (lane 2)
        asm.f_to_int(4, 7);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 40, "v4_mul lane 2: 4*10=40, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_dot_gpu() {
        let mut system = create_gpu_system();

        // r5 = (1, 2, 3, 4), r6 = (2, 2, 2, 2)
        // dot = 1*2 + 2*2 + 3*2 + 4*2 = 2 + 4 + 6 + 8 = 20

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 1.0);
        asm.sety(5, 2.0);
        asm.setz(5, 3.0);
        asm.setw(5, 4.0);
        asm.loadi(6, 2.0);
        asm.sety(6, 2.0);
        asm.setz(6, 2.0);
        asm.setw(6, 2.0);
        asm.v4_dot(4, 5, 6);
        asm.f_to_int(4, 4);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 20, "v4_dot: 1*2+2*2+3*2+4*2=20, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_splat_gpu() {
        let mut system = create_gpu_system();

        // r5.x = 7, splat to (7, 7, 7, 7)
        // Add all lanes: 7*4 = 28

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 7.0);
        asm.v4_splat(4, 5);
        // Sum all lanes using dot with (1,1,1,1)
        asm.loadi(6, 1.0);
        asm.sety(6, 1.0);
        asm.setz(6, 1.0);
        asm.setw(6, 1.0);
        asm.v4_dot(7, 4, 6);  // 7+7+7+7 = 28
        asm.f_to_int(4, 7);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 28, "v4_splat: 7*4=28, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_shuffle_gpu() {
        let mut system = create_gpu_system();

        // r5 = (10, 20, 30, 40)
        // shuffle with mask 0b11_10_01_00 = (x=0->10, y=1->20, z=2->30, w=3->40) = identity
        // shuffle with mask 0b00_00_00_00 = (x=0, y=0, z=0, w=0) = (10, 10, 10, 10)
        // dot with (1,1,1,1) = 40

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 10.0);
        asm.sety(5, 20.0);
        asm.setz(5, 30.0);
        asm.setw(5, 40.0);
        // Shuffle: all lanes get lane 0 (mask = 0b00_00_00_00 = 0)
        asm.v4_shuffle(4, 5, 0, 0b00_00_00_00);
        // Sum all lanes
        asm.loadi(6, 1.0);
        asm.sety(6, 1.0);
        asm.setz(6, 1.0);
        asm.setw(6, 1.0);
        asm.v4_dot(7, 4, 6);  // 10+10+10+10 = 40
        asm.f_to_int(4, 7);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 40, "v4_shuffle: 10*4=40, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_replace_gpu() {
        let mut system = create_gpu_system();

        // r5 = (1, 2, 3, 4), r6.x = 100
        // replace lane 1 -> (1, 100, 3, 4)
        // dot with (0, 1, 0, 0) = 100

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 1.0);
        asm.sety(5, 2.0);
        asm.setz(5, 3.0);
        asm.setw(5, 4.0);
        asm.loadi(6, 100.0);
        asm.v4_replace(4, 5, 6, 1);  // Replace lane 1 with 100
        // Extract lane 1
        asm.v4_extract(7, 4, 1);
        asm.f_to_int(4, 7);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 100, "v4_replace lane 1: got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_min_max_gpu() {
        let mut system = create_gpu_system();

        // r5 = (5, 15, 25, 35), r6 = (10, 10, 10, 10)
        // min = (5, 10, 10, 10), sum = 35
        // max = (10, 15, 25, 35), sum = 85

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 5.0);
        asm.sety(5, 15.0);
        asm.setz(5, 25.0);
        asm.setw(5, 35.0);
        asm.loadi(6, 10.0);
        asm.sety(6, 10.0);
        asm.setz(6, 10.0);
        asm.setw(6, 10.0);
        asm.v4_min(4, 5, 6);
        // Sum all lanes
        asm.loadi(7, 1.0);
        asm.sety(7, 1.0);
        asm.setz(7, 1.0);
        asm.setw(7, 1.0);
        asm.v4_dot(8, 4, 7);  // 5+10+10+10 = 35
        asm.f_to_int(4, 8);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 35, "v4_min sum: 5+10+10+10=35, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_abs_neg_gpu() {
        let mut system = create_gpu_system();

        // r5 = (-5, -10, 15, -20)
        // abs = (5, 10, 15, 20), sum = 50

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, -5.0);
        asm.sety(5, -10.0);
        asm.setz(5, 15.0);
        asm.setw(5, -20.0);
        asm.v4_abs(4, 5);
        // Sum all lanes
        asm.loadi(6, 1.0);
        asm.sety(6, 1.0);
        asm.setz(6, 1.0);
        asm.setw(6, 1.0);
        asm.v4_dot(7, 4, 6);  // 5+10+15+20 = 50
        asm.f_to_int(4, 7);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 50, "v4_abs sum: 5+10+15+20=50, got {}", result);

        system.close_app(slot);
    }

    #[test]
    fn test_v4_eq_lt_gpu() {
        let mut system = create_gpu_system();

        // r5 = (5, 10, 15, 20), r6 = (5, 5, 20, 20)
        // eq = (1, 0, 0, 1), sum = 2
        // lt = (0, 0, 1, 0), sum = 1

        let mut asm = BytecodeAssembler::new();
        asm.loadi(5, 5.0);
        asm.sety(5, 10.0);
        asm.setz(5, 15.0);
        asm.setw(5, 20.0);
        asm.loadi(6, 5.0);
        asm.sety(6, 5.0);
        asm.setz(6, 20.0);
        asm.setw(6, 20.0);
        asm.v4_eq(4, 5, 6);  // (1, 0, 0, 1)
        // Sum all lanes
        asm.loadi(7, 1.0);
        asm.sety(7, 1.0);
        asm.setz(7, 1.0);
        asm.setw(7, 1.0);
        asm.v4_dot(8, 4, 7);  // 1+0+0+1 = 2
        asm.f_to_int(4, 8);
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 2, "v4_eq sum: 1+0+0+1=2, got {}", result);

        system.close_app(slot);
    }
}
