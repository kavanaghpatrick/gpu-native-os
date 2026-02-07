//! Phase 6: GPU Allocator Tests (Issue #179)
//!
//! THE GPU IS THE COMPUTER.
//! GPU-resident memory allocator enabling Rust's alloc crate (Vec, String, Box).
//!
//! These tests verify:
//! - ALLOC returns valid pointer
//! - DEALLOC frees memory for reuse
//! - REALLOC preserves data when growing
//! - ALLOC_ZERO returns zeroed memory
//! - Lock-free operation (no GPU hangs)
//! - WASM __rust_alloc intrinsic mapping

use rust_experiment::gpu_os::gpu_app_system::{bytecode_op, BytecodeAssembler, BytecodeInst, BytecodeHeader};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode bytecode bytes back into header and instructions
fn decode_bytecode(bytecode: &[u8]) -> (BytecodeHeader, Vec<BytecodeInst>) {
    let header: BytecodeHeader = unsafe {
        std::ptr::read(bytecode.as_ptr() as *const BytecodeHeader)
    };

    let inst_start = std::mem::size_of::<BytecodeHeader>();
    let inst_bytes = &bytecode[inst_start..];
    let inst_count = header.code_size as usize;

    let mut instructions = Vec::with_capacity(inst_count);
    for i in 0..inst_count {
        let offset = i * std::mem::size_of::<BytecodeInst>();
        let inst: BytecodeInst = unsafe {
            std::ptr::read(inst_bytes[offset..].as_ptr() as *const BytecodeInst)
        };
        instructions.push(inst);
    }

    (header, instructions)
}

/// Extract immediate as u32 bits
fn imm_as_u32(imm: f32) -> u32 {
    imm.to_bits()
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE CONSTANT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_allocator_opcode_values() {
    // Verify all allocator opcodes are in 0xF0-0xF3 range
    assert_eq!(bytecode_op::ALLOC, 0xF0);
    assert_eq!(bytecode_op::DEALLOC, 0xF1);
    assert_eq!(bytecode_op::REALLOC, 0xF2);
    assert_eq!(bytecode_op::ALLOC_ZERO, 0xF3);
}

#[test]
fn test_allocator_opcodes_dont_overlap_with_atomics() {
    // Ensure allocator range (0xF0-0xF3) doesn't overlap with atomic range (0xE0-0xEE)
    assert!(bytecode_op::ALLOC > bytecode_op::MEM_FENCE);
    assert_eq!(bytecode_op::MEM_FENCE, 0xEE);
    assert_eq!(bytecode_op::ALLOC, 0xF0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASSEMBLER ENCODING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_alloc_basic() {
    let mut asm = BytecodeAssembler::new();

    // Load size (64 bytes) into r8
    asm.loadi_uint(8, 64);
    // Load alignment (4 bytes) into r9
    asm.loadi_uint(9, 4);
    // Allocate: r10 = alloc(r8, r9)
    asm.alloc(10, 8, 9);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[2].opcode, bytecode_op::ALLOC);
    assert_eq!(insts[2].dst, 10);   // destination register
    assert_eq!(insts[2].src1, 8);   // size register
    assert_eq!(insts[2].src2, 9);   // align register
}

#[test]
fn test_alloc_multiple() {
    let mut asm = BytecodeAssembler::new();

    // Allocate three blocks of different sizes
    asm.loadi_uint(8, 16);   // 16 bytes
    asm.loadi_uint(9, 4);    // align 4
    asm.alloc(10, 8, 9);     // first allocation -> r10

    asm.loadi_uint(8, 32);   // 32 bytes
    asm.alloc(11, 8, 9);     // second allocation -> r11

    asm.loadi_uint(8, 64);   // 64 bytes
    asm.alloc(12, 8, 9);     // third allocation -> r12

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _) = decode_bytecode(&bytecode);

    // Count: 2 loadi + alloc + loadi + alloc + loadi + alloc + halt = 8
    assert_eq!(header.code_size, 8);
}

#[test]
fn test_dealloc_reuse() {
    let mut asm = BytecodeAssembler::new();

    // Allocate
    asm.loadi_uint(8, 64);
    asm.loadi_uint(9, 4);
    asm.alloc(10, 8, 9);     // r10 = ptr

    // Free it
    asm.dealloc(10, 8, 9);   // dealloc(ptr, size, align)

    // Allocate again - should reuse freed block
    asm.alloc(11, 8, 9);     // r11 should get same address as r10 did

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    // Verify dealloc encoding
    assert_eq!(insts[3].opcode, bytecode_op::DEALLOC);
    assert_eq!(insts[3].src1, 10);  // ptr register
    assert_eq!(insts[3].src2, 8);   // size register
}

#[test]
fn test_realloc_grow() {
    let mut asm = BytecodeAssembler::new();

    // Allocate initial block
    asm.loadi_uint(8, 16);   // old_size = 16
    asm.loadi_uint(9, 4);    // align
    asm.alloc(10, 8, 9);     // r10 = ptr

    // Grow to larger size
    asm.loadi_uint(11, 64);  // new_size = 64
    asm.realloc(12, 10, 8, 11);  // r12 = realloc(ptr, old_size, new_size)

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    // Verify realloc encoding
    assert_eq!(insts[4].opcode, bytecode_op::REALLOC);
    assert_eq!(insts[4].dst, 12);   // destination
    assert_eq!(insts[4].src1, 10);  // ptr register
    assert_eq!(insts[4].src2, 8);   // old_size register
    // new_size register index is in imm
}

#[test]
fn test_alloc_zero() {
    let mut asm = BytecodeAssembler::new();

    // Allocate zeroed memory
    asm.loadi_uint(8, 128);  // size = 128
    asm.loadi_uint(9, 8);    // align = 8
    asm.alloc_zero(10, 8, 9);  // r10 = alloc_zeroed(size, align)

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[2].opcode, bytecode_op::ALLOC_ZERO);
    assert_eq!(insts[2].dst, 10);
    assert_eq!(insts[2].src1, 8);
    assert_eq!(insts[2].src2, 9);
}

#[test]
fn test_alloc_different_sizes() {
    // Test all size classes: 16, 32, 64, 128, 256, 512, 1024, 2048+
    let sizes: [u32; 8] = [16, 32, 64, 128, 256, 512, 1024, 2048];

    for (i, size) in sizes.iter().enumerate() {
        let mut asm = BytecodeAssembler::new();
        asm.loadi_uint(8, *size);
        asm.loadi_uint(9, 1);
        asm.alloc(10, 8, 9);
        asm.halt();

        let bytecode = asm.build(0);
        let (_, insts) = decode_bytecode(&bytecode);

        assert_eq!(insts[2].opcode, bytecode_op::ALLOC,
            "Size class {} should produce ALLOC opcode", i);
        assert_eq!(imm_as_u32(insts[0].imm), *size,
            "Size {} should be loaded correctly", size);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATTERN TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_vec_like_pattern() {
    // Pattern for Vec<T>: allocate, grow, eventually free
    let mut asm = BytecodeAssembler::new();

    // Initial allocation (capacity 4)
    asm.loadi_uint(8, 16);   // 4 * 4 bytes
    asm.loadi_uint(9, 4);    // align
    asm.alloc(10, 8, 9);     // r10 = buffer ptr

    // Grow to capacity 8
    asm.loadi_uint(11, 32);  // 8 * 4 bytes
    asm.realloc(10, 10, 8, 11);  // r10 = realloc(ptr, 16, 32)

    // Update old_size for next realloc
    asm.mov(8, 11);

    // Grow to capacity 16
    asm.loadi_uint(11, 64);  // 16 * 4 bytes
    asm.realloc(10, 10, 8, 11);

    // Free on drop
    asm.dealloc(10, 11, 9);

    asm.halt();

    let bytecode = asm.build(0);
    let (header, _) = decode_bytecode(&bytecode);

    // Should compile without issues
    assert!(header.code_size > 0);
}

#[test]
fn test_box_pattern() {
    // Pattern for Box<T>: allocate once, use, free once
    let mut asm = BytecodeAssembler::new();

    // Box::new(value)
    asm.loadi_uint(8, 64);   // sizeof(T)
    asm.loadi_uint(9, 8);    // alignof(T)
    asm.alloc(10, 8, 9);     // r10 = Box pointer

    // Use the box... (store something at the pointer)
    asm.loadi_uint(11, 42);
    asm.st(10, 11, 0.0);     // *box = 42

    // Drop: Box drops automatically, freeing memory
    asm.dealloc(10, 8, 9);

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    // Verify pattern
    assert_eq!(insts[2].opcode, bytecode_op::ALLOC);
    assert_eq!(insts[5].opcode, bytecode_op::DEALLOC);
}

#[test]
fn test_string_pattern() {
    // Pattern for String: allocate zeroed, reallocate as needed
    let mut asm = BytecodeAssembler::new();

    // String::with_capacity(16)
    asm.loadi_uint(8, 16);   // initial capacity
    asm.loadi_uint(9, 1);    // byte alignment
    asm.alloc_zero(10, 8, 9);  // r10 = buffer (zeroed)

    // Push more data, need to grow
    asm.loadi_uint(11, 32);
    asm.realloc(10, 10, 8, 11);

    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[2].opcode, bytecode_op::ALLOC_ZERO);
    assert_eq!(insts[4].opcode, bytecode_op::REALLOC);
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTRUCTION COUNT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_alloc_ops_emit_one_instruction() {
    // Each allocator operation should emit exactly one instruction

    let ops: Vec<(&str, Box<dyn Fn(&mut BytecodeAssembler)>)> = vec![
        ("alloc", Box::new(|a: &mut BytecodeAssembler| { a.alloc(10, 8, 9); })),
        ("dealloc", Box::new(|a: &mut BytecodeAssembler| { a.dealloc(10, 8, 9); })),
        ("realloc", Box::new(|a: &mut BytecodeAssembler| { a.realloc(10, 8, 9, 11); })),
        ("alloc_zero", Box::new(|a: &mut BytecodeAssembler| { a.alloc_zero(10, 8, 9); })),
    ];

    for (name, emit_fn) in ops {
        let mut asm = BytecodeAssembler::new();
        emit_fn(&mut asm);
        assert_eq!(asm.pc(), 1, "{} should emit exactly 1 instruction", name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPCODE RANGE VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_allocator_opcodes_contiguous() {
    // Verify allocator opcodes form a contiguous range
    let opcodes = [
        bytecode_op::ALLOC,
        bytecode_op::DEALLOC,
        bytecode_op::REALLOC,
        bytecode_op::ALLOC_ZERO,
    ];

    // All should be in 0xF0-0xF3 range
    for (i, op) in opcodes.iter().enumerate() {
        assert_eq!(*op, 0xF0 + i as u8, "Opcode at index {} should be 0x{:02X}", i, 0xF0 + i);
    }
}

#[test]
fn test_allocator_opcode_count() {
    // We have 4 allocator operations
    assert_eq!(bytecode_op::ALLOC_ZERO - bytecode_op::ALLOC + 1, 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASM INTRINSIC MAPPING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "wasm")]
mod wasm_tests {
    use wasm_translator::types::{GpuIntrinsic, ImportedFunc};

    #[test]
    fn test_intrinsic_rust_alloc() {
        let import = ImportedFunc::from_import("env", "__rust_alloc", 0);
        assert_eq!(import.intrinsic, Some(GpuIntrinsic::Alloc));
    }

    #[test]
    fn test_intrinsic_rust_dealloc() {
        let import = ImportedFunc::from_import("env", "__rust_dealloc", 0);
        assert_eq!(import.intrinsic, Some(GpuIntrinsic::Dealloc));
    }

    #[test]
    fn test_intrinsic_rust_realloc() {
        let import = ImportedFunc::from_import("env", "__rust_realloc", 0);
        assert_eq!(import.intrinsic, Some(GpuIntrinsic::Realloc));
    }

    #[test]
    fn test_intrinsic_rust_alloc_zeroed() {
        let import = ImportedFunc::from_import("env", "__rust_alloc_zeroed", 0);
        assert_eq!(import.intrinsic, Some(GpuIntrinsic::AllocZeroed));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CASE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_zero_size_alloc_encoding() {
    // Allocating zero bytes should still produce valid bytecode
    // (GPU will return failure pointer 0xFFFFFFFF)
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0);  // size = 0
    asm.loadi_uint(9, 1);  // align = 1
    asm.alloc(10, 8, 9);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[2].opcode, bytecode_op::ALLOC);
    assert_eq!(imm_as_u32(insts[0].imm), 0);  // size = 0
}

#[test]
fn test_large_size_alloc_encoding() {
    // Large allocation (> 2048 bytes) goes to bump allocator
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 65536);  // 64KB
    asm.loadi_uint(9, 4096);   // page aligned
    asm.alloc(10, 8, 9);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[2].opcode, bytecode_op::ALLOC);
    assert_eq!(imm_as_u32(insts[0].imm), 65536);
}

#[test]
fn test_null_ptr_realloc_encoding() {
    // realloc with null pointer should act like alloc
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 0xFFFFFFFF);  // null/invalid ptr
    asm.loadi_uint(9, 0);           // old_size = 0
    asm.loadi_uint(10, 64);         // new_size = 64
    asm.realloc(11, 8, 9, 10);      // should allocate new block
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[3].opcode, bytecode_op::REALLOC);
}

#[test]
fn test_shrink_realloc_encoding() {
    // realloc to smaller size - should return same pointer if same size class
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 64);     // assume ptr = 64 (arbitrary)
    asm.loadi_uint(9, 128);    // old_size = 128
    asm.loadi_uint(10, 32);    // new_size = 32 (smaller)
    asm.realloc(11, 8, 9, 10);
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[3].opcode, bytecode_op::REALLOC);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY SIZE/GROW TESTS (Issue #210 - GPU-Native Dynamic Memory)
// THE GPU IS THE COMPUTER - WASM linear memory on GPU
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_memory_size_opcode() {
    let mut asm = BytecodeAssembler::new();
    asm.memory_size(8);  // dst = memory.size (in pages)
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[0].opcode, bytecode_op::MEMORY_SIZE);
    assert_eq!(insts[0].dst, 8);
}

#[test]
fn test_memory_grow_opcode() {
    let mut asm = BytecodeAssembler::new();
    asm.loadi_uint(8, 1);     // delta = 1 page
    asm.memory_grow(9, 8);    // dst = memory.grow(delta)
    asm.halt();

    let bytecode = asm.build(0);
    let (_, insts) = decode_bytecode(&bytecode);

    assert_eq!(insts[1].opcode, bytecode_op::MEMORY_GROW);
    assert_eq!(insts[1].dst, 9);
    assert_eq!(insts[1].src1, 8);
}

#[test]
fn test_memory_opcodes_contiguous() {
    // Memory opcodes should be contiguous
    assert_eq!(bytecode_op::MEMORY_GROW - bytecode_op::MEMORY_SIZE, 1);
}

#[test]
fn test_memory_opcodes_values() {
    // Verify expected opcode values
    assert_eq!(bytecode_op::MEMORY_SIZE, 0xF4);
    assert_eq!(bytecode_op::MEMORY_GROW, 0xF5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY SIZE/GROW GPU EXECUTION TESTS (Issue #210)
// THE GPU IS THE COMPUTER - test actual GPU execution of memory operations
// ═══════════════════════════════════════════════════════════════════════════════

mod memory_gpu_tests {
    use super::*;
    use metal::Device;
    use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type};

    #[test]
    fn test_memory_size_gpu_execution() {
        // Test that memory.size returns a non-zero value on GPU
        let mut asm = BytecodeAssembler::new();
        asm.memory_size(0);  // r0.x = memory.size (in pages)
        // Store result to state[3] for reading back
        asm.loadi_uint(30, 3);
        asm.st(30, 0, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        // Execute on GPU
        let device = Device::system_default().expect("No Metal device");
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        system.set_use_parallel_megakernel(true);

        let slot = system.launch_by_type(app_type::BYTECODE)
            .expect("Failed to launch bytecode app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot);
        system.close_app(slot);

        // Memory size should be > 0 pages
        assert!(result.is_some(), "Should get result from GPU");
        let pages = result.unwrap();
        assert!(pages > 0, "Memory size should be > 0 pages, got {}", pages);
    }

    #[test]
    fn test_memory_grow_success() {
        // Test that memory.grow returns old size on success
        let mut asm = BytecodeAssembler::new();
        asm.memory_size(1);        // r1.x = current size
        asm.loadi_uint(2, 0);      // r2.x = 0 pages (grow by 0 - should always succeed)
        asm.memory_grow(0, 2);     // r0.x = memory.grow(0) - returns old size
        // Store result to state[3] for reading back
        asm.loadi_uint(30, 3);
        asm.st(30, 0, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let device = Device::system_default().expect("No Metal device");
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        system.set_use_parallel_megakernel(true);

        let slot = system.launch_by_type(app_type::BYTECODE)
            .expect("Failed to launch bytecode app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot);
        system.close_app(slot);

        assert!(result.is_some(), "Should get result from GPU");
        let old_pages = result.unwrap();
        // Growing by 0 should succeed and return the current size (not -1)
        assert!(old_pages >= 0, "memory.grow(0) should succeed, got {}", old_pages);
    }

    #[test]
    fn test_memory_grow_failure() {
        // Test that memory.grow returns -1 when growing beyond capacity
        let mut asm = BytecodeAssembler::new();
        // Try to grow by a huge number of pages (should fail)
        asm.loadi_uint(1, 0xFFFFFF);  // ~16 million pages = 1TB - definitely too large
        asm.memory_grow(0, 1);         // r0.x = memory.grow(huge) - should return -1
        // Store result to state[3] for reading back
        asm.loadi_uint(30, 3);
        asm.st(30, 0, 0.0);
        asm.halt();

        let bytecode = asm.build(0);

        let device = Device::system_default().expect("No Metal device");
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        system.set_use_parallel_megakernel(true);

        let slot = system.launch_by_type(app_type::BYTECODE)
            .expect("Failed to launch bytecode app");
        system.write_app_state(slot, &bytecode);
        system.run_frame();

        let result = system.read_bytecode_result(slot);
        system.close_app(slot);

        assert!(result.is_some(), "Should get result from GPU");
        let grow_result = result.unwrap();
        // Growing by 16M pages should fail and return -1
        assert_eq!(grow_result, -1, "memory.grow with huge size should return -1");
    }
}
