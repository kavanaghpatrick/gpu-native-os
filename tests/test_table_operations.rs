//! Test GPU-Native Table Operations (Issue #212)
//!
//! THE GPU IS THE COMPUTER - tables are GPU-resident arrays with O(1) lookup

use rust_experiment::gpu_os::gpu_app_system::{
    bytecode_op, BytecodeAssembler, GpuAppSystem, app_type,
};

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_table_opcodes_values() {
    assert_eq!(bytecode_op::TABLE_GET, 0x50);
    assert_eq!(bytecode_op::TABLE_SET, 0x51);
    assert_eq!(bytecode_op::TABLE_SIZE, 0x52);
    assert_eq!(bytecode_op::TABLE_GROW, 0x53);
    assert_eq!(bytecode_op::TABLE_INIT, 0x54);
    assert_eq!(bytecode_op::TABLE_COPY, 0x55);
    assert_eq!(bytecode_op::TABLE_FILL, 0x56);
}

#[test]
fn test_table_opcodes_contiguous() {
    // Table opcodes should be contiguous
    assert_eq!(bytecode_op::TABLE_SET, bytecode_op::TABLE_GET + 1);
    assert_eq!(bytecode_op::TABLE_SIZE, bytecode_op::TABLE_SET + 1);
    assert_eq!(bytecode_op::TABLE_GROW, bytecode_op::TABLE_SIZE + 1);
    assert_eq!(bytecode_op::TABLE_INIT, bytecode_op::TABLE_GROW + 1);
    assert_eq!(bytecode_op::TABLE_COPY, bytecode_op::TABLE_INIT + 1);
    assert_eq!(bytecode_op::TABLE_FILL, bytecode_op::TABLE_COPY + 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE ASSEMBLER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_table_get_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.table_get(4, 5, 0);  // Get from table 0, index in r5, result in r4

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::TABLE_GET);
    assert_eq!(bytecode[inst_offset + 1], 4);  // dst
    assert_eq!(bytecode[inst_offset + 2], 5);  // src1 (idx_reg)
}

#[test]
fn test_table_set_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.table_set(5, 6, 0);  // Set table 0[r5] = r6

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::TABLE_SET);
    assert_eq!(bytecode[inst_offset + 2], 5);  // src1 (idx_reg)
    assert_eq!(bytecode[inst_offset + 3], 6);  // src2 (val_reg)
}

#[test]
fn test_table_size_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.table_size(4, 0);  // Get size of table 0 into r4

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::TABLE_SIZE);
    assert_eq!(bytecode[inst_offset + 1], 4);  // dst
}

#[test]
fn test_table_grow_encoding() {
    let mut asm = BytecodeAssembler::new();
    asm.table_grow(4, 5, 6, 0);  // Grow table 0 by r5, init with r6, old size in r4

    let bytecode = asm.build(0);
    let inst_offset = 16;
    assert_eq!(bytecode[inst_offset], bytecode_op::TABLE_GROW);
    assert_eq!(bytecode[inst_offset + 1], 4);  // dst
    assert_eq!(bytecode[inst_offset + 2], 5);  // src1 (delta_reg)
    assert_eq!(bytecode[inst_offset + 3], 6);  // src2 (init_reg)
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

    /// Helper to initialize a table in heap memory
    /// Table layout: [size (4 bytes), max_size (4 bytes), entries...]
    /// Heap starts at state[4] (byte offset 64) - see gpu_app_system.rs:
    ///   device uchar* heap = (device uchar*)(state + 4);
    fn init_table_in_heap(asm: &mut BytecodeAssembler, table_idx: u32, size: u32, max_size: u32) {
        // Heap offset in state buffer = 64 bytes (4 float4s)
        const HEAP_OFFSET: u32 = 64;
        // Table base in heap = table_idx * 1024
        let base = HEAP_OFFSET + table_idx * 1024;

        // Store size at offset 0
        asm.loadi_uint(8, base);      // r8 = base offset in state
        asm.loadi_uint(9, size);      // r9 = size
        asm.st4(8, 9, 0.0);           // state[base] = size

        // Store max_size at offset 4
        asm.loadi_uint(9, max_size);  // r9 = max_size
        asm.st4(8, 9, 4.0);           // state[base+4] = max_size
    }

    #[test]
    fn test_table_size_gpu() {
        let mut system = create_gpu_system();

        // Bytecode:
        // 1. Initialize table 0 with size=5, max_size=10
        // 2. Get table size
        // 3. Store result to state[3]
        
        let mut asm = BytecodeAssembler::new();
        
        // Initialize table
        init_table_in_heap(&mut asm, 0, 5, 10);
        
        // Get table size
        asm.table_size(4, 0);  // r4 = table.size(0)
        
        // Store result to state[3]
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);
        
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 5, "table.size(0) should return 5: got {}", result);
        
        system.close_app(slot);
    }

    #[test]
    fn test_table_get_set_gpu() {
        let mut system = create_gpu_system();

        // Bytecode:
        // 1. Initialize table 0 with size=5, max_size=10
        // 2. Set table[2] = 42
        // 3. Get table[2]
        // 4. Store result to state[3]
        
        let mut asm = BytecodeAssembler::new();
        
        // Initialize table
        init_table_in_heap(&mut asm, 0, 5, 10);
        
        // Set table[2] = 42
        asm.loadi_uint(5, 2);   // r5 = index 2
        asm.loadi_uint(6, 42);  // r6 = value 42
        asm.table_set(5, 6, 0); // table.set(0, 2, 42)
        
        // Get table[2]
        asm.table_get(4, 5, 0); // r4 = table.get(0, 2)
        
        // Store result to state[3]
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);
        
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 42, "table[2] should be 42: got {}", result);
        
        system.close_app(slot);
    }

    #[test]
    fn test_table_grow_gpu() {
        let mut system = create_gpu_system();

        // Bytecode:
        // 1. Initialize table 0 with size=2, max_size=10
        // 2. Grow by 3 (with null init)
        // 3. Old size should be 2
        // 4. New size should be 5
        
        let mut asm = BytecodeAssembler::new();
        
        // Initialize table
        init_table_in_heap(&mut asm, 0, 2, 10);
        
        // Grow table by 3
        asm.loadi_uint(5, 3);           // r5 = delta 3
        asm.loadi_uint(6, 0xFFFFFFFF);  // r6 = null (init value)
        asm.table_grow(4, 5, 6, 0);     // r4 = table.grow(0, 3, null)
        
        // Store old size to state[3]
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);
        
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 2, "table.grow should return old size 2: got {}", result);
        
        system.close_app(slot);
    }

    #[test]
    fn test_table_fill_gpu() {
        let mut system = create_gpu_system();

        // Bytecode:
        // 1. Initialize table 0 with size=5, max_size=10
        // 2. Fill entries 1-3 with value 99
        // 3. Get table[2] (should be 99)
        
        let mut asm = BytecodeAssembler::new();
        
        // Initialize table
        init_table_in_heap(&mut asm, 0, 5, 10);
        
        // Fill entries 1-3 with 99: table.fill(0, dst=1, val=99, count=3)
        asm.loadi_uint(4, 3);   // r4 = count (passed as dst reg)
        asm.loadi_uint(5, 1);   // r5 = dst index
        asm.loadi_uint(6, 99);  // r6 = value
        asm.table_fill(5, 6, 4, 0);
        
        // Get table[2]
        asm.loadi_uint(5, 2);
        asm.table_get(4, 5, 0);
        
        // Store result to state[3]
        asm.loadi_uint(30, 3);
        asm.st(30, 4, 0.0);
        asm.halt();

        let bytecode = asm.build(0);
        
        let slot = system.launch_app(app_type::BYTECODE, 128 * 1024, 1024)
            .expect("Failed to launch app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();
        
        let result = system.read_bytecode_result(slot).expect("Failed to read result");
        assert_eq!(result, 99, "table[2] should be 99 after fill: got {}", result);
        
        system.close_app(slot);
    }
}
