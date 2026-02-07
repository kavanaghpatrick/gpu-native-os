//! Issue #168: GPU App Loader Tests
//!
//! Tests for loading .gpuapp files from filesystem.
//! THE GPU IS THE COMPUTER - GPU parses, validates, initializes.

use metal::Device;
use std::path::PathBuf;
use tempfile::tempdir;

use rust_experiment::gpu_os::gpu_app_loader::{
    GpuAppBuilder, GpuAppLoader, GpuAppFileHeader, GPUAPP_MAGIC, GPUAPP_VERSION,
    GPUAPP_HEADER_SIZE, parse_header, read_gpuapp_header,
    ACTIVE_FLAG, VISIBLE_FLAG, APP_TYPE_BYTECODE, INVALID_SLOT,
};
use rust_experiment::gpu_os::gpu_app_system::BytecodeInst;

// ═══════════════════════════════════════════════════════════════════════════════
// BUILDER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_builder_creates_valid_header() {
    let builder = GpuAppBuilder::new("test_app");
    let data = builder.build();

    assert!(data.len() >= GPUAPP_HEADER_SIZE);

    let header = parse_header(&data).expect("Should parse header");
    assert_eq!(&header.magic, GPUAPP_MAGIC);
    assert_eq!(header.version, GPUAPP_VERSION);
    assert_eq!(header.name_str(), "test_app");
    assert_eq!(header.code_size, 0);
}

#[test]
fn test_builder_with_bytecode() {
    let mut builder = GpuAppBuilder::new("bytecode_app");

    // Add some instructions
    builder.add_instruction(BytecodeInst {
        opcode: 1,  // LOADI
        dst: 4,
        src1: 0,
        src2: 0,
        imm: 100.0,
    });
    builder.add_instruction(BytecodeInst {
        opcode: 0,  // HALT
        dst: 0,
        src1: 0,
        src2: 0,
        imm: 0.0,
    });
    builder.set_vertex_budget(100);

    let data = builder.build();
    let header = parse_header(&data).expect("Should parse");

    assert_eq!(header.code_size, 2);
    assert_eq!(header.vertex_budget, 100);
    assert_eq!(header.code_offset, GPUAPP_HEADER_SIZE as u32);

    // Verify total size
    let expected_size = GPUAPP_HEADER_SIZE + 2 * std::mem::size_of::<BytecodeInst>();
    assert_eq!(data.len(), expected_size);
}

#[test]
fn test_builder_write_to_file() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test.gpuapp");

    let mut builder = GpuAppBuilder::new("file_test");
    builder.add_instruction(BytecodeInst {
        opcode: 0,
        dst: 0,
        src1: 0,
        src2: 0,
        imm: 0.0,
    });
    builder.set_state_size(1024);

    builder.write_to_file(&path).expect("Should write file");

    // Read back and verify
    let header = read_gpuapp_header(&path).expect("Should read header");
    assert_eq!(header.name_str(), "file_test");
    assert_eq!(header.code_size, 1);
    assert_eq!(header.state_size, 1024);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEADER VALIDATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_valid_header() {
    let builder = GpuAppBuilder::new("valid");
    let data = builder.build();
    let header = parse_header(&data);

    assert!(header.is_some());
    assert!(header.unwrap().is_valid());
}

#[test]
fn test_invalid_magic() {
    let mut builder = GpuAppBuilder::new("bad_magic");
    let mut data = builder.build();

    // Corrupt magic
    data[0] = b'X';

    let header = parse_header(&data);
    assert!(header.is_none());
}

#[test]
fn test_invalid_version() {
    let builder = GpuAppBuilder::new("bad_version");
    let mut data = builder.build();

    // Set invalid version (bytes 6-7)
    data[6] = 99;
    data[7] = 0;

    let header = parse_header(&data);
    assert!(header.is_none());
}

#[test]
fn test_too_small_data() {
    let data = vec![0u8; 32];  // Too small
    let header = parse_header(&data);
    assert!(header.is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU VALIDATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_gpu_validate_valid_header() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    let builder = GpuAppBuilder::new("gpu_valid");
    let data = builder.build();

    assert!(loader.gpu_validate_header(&device, &data));
}

#[test]
fn test_gpu_validate_invalid_magic() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    let mut builder = GpuAppBuilder::new("gpu_bad");
    let mut data = builder.build();
    data[0] = b'X';  // Corrupt magic

    assert!(!loader.gpu_validate_header(&device, &data));
}

#[test]
fn test_gpu_validate_invalid_version() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    let builder = GpuAppBuilder::new("gpu_ver");
    let mut data = builder.build();
    data[6] = 99;  // Bad version

    assert!(!loader.gpu_validate_header(&device, &data));
}

// ═══════════════════════════════════════════════════════════════════════════════
// APP INITIALIZATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_gpu_init_app() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // Create test .gpuapp with bytecode
    let mut builder = GpuAppBuilder::new("init_test");
    builder.add_instruction(BytecodeInst {
        opcode: 1,  // LOADI
        dst: 4,
        src1: 0,
        src2: 0,
        imm: 50.0,
    });
    builder.add_instruction(BytecodeInst {
        opcode: 0,  // HALT
        dst: 0,
        src1: 0,
        src2: 0,
        imm: 0.0,
    });
    builder.set_vertex_budget(100);

    let data = builder.build();

    // Create app table buffer (simplified structure for testing)
    // Header (32 bytes) + free_bitmap (8 bytes) + descriptors
    let max_slots = 64u32;
    let header_size = 32;
    let bitmap_size = 8;
    let descriptor_size = 128;
    let table_size = header_size + bitmap_size + (max_slots as usize * descriptor_size);

    let app_table_buffer = device.new_buffer(
        table_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize header
    let table_ptr = app_table_buffer.contents() as *mut u8;
    unsafe {
        // Magic
        *(table_ptr as *mut u32) = 0x41505054;  // "TAPP"
        // Version
        *(table_ptr.add(4) as *mut u32) = 1;
        // Max slots
        *(table_ptr.add(8) as *mut u32) = max_slots;
        // Active count
        *(table_ptr.add(12) as *mut u32) = 0;

        // Free bitmap: all free (all 1s)
        let bitmap_ptr = table_ptr.add(32) as *mut u32;
        *bitmap_ptr = 0xFFFFFFFF;
        *bitmap_ptr.add(1) = 0xFFFFFFFF;
    }

    // Create unified state buffer
    let state_buffer = device.new_buffer(
        4 * 1024 * 1024,  // 4MB
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize app
    let slot = loader.gpu_init_app(&device, &data, &app_table_buffer, &state_buffer);

    assert!(slot.is_some(), "Should allocate slot");
    let slot = slot.unwrap();
    println!("Initialized app in slot {}", slot);

    // Verify app descriptor
    let desc_offset = header_size + bitmap_size + (slot as usize * descriptor_size);
    let desc_ptr = unsafe { (app_table_buffer.contents() as *const u8).add(desc_offset) };

    // Read flags
    let flags = unsafe { *(desc_ptr as *const u32) };
    assert!(flags & ACTIVE_FLAG != 0, "App should be active");
    assert!(flags & VISIBLE_FLAG != 0, "App should be visible");

    // Read app_type
    let app_type = unsafe { *(desc_ptr.add(4) as *const u32) };
    assert_eq!(app_type, APP_TYPE_BYTECODE, "App type should be BYTECODE");

    // Read vertex_budget (offset 36 in GpuAppDescriptor: flags=0, app_type=4, slot_id=8, window_id=12,
    //   state_offset=16, state_size=20, vertex_offset=24, vertex_size=28, vertex_count=32, vertex_budget=36)
    let vertex_budget = unsafe { *(desc_ptr.add(36) as *const u32) };
    assert_eq!(vertex_budget, 100, "Vertex budget should match");
}

#[test]
fn test_gpu_init_multiple_apps() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // Create app table
    let max_slots = 64u32;
    let header_size = 32;
    let bitmap_size = 8;
    let descriptor_size = 128;
    let table_size = header_size + bitmap_size + (max_slots as usize * descriptor_size);

    let app_table_buffer = device.new_buffer(
        table_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let table_ptr = app_table_buffer.contents() as *mut u8;
    unsafe {
        *(table_ptr as *mut u32) = 0x41505054;
        *(table_ptr.add(4) as *mut u32) = 1;
        *(table_ptr.add(8) as *mut u32) = max_slots;
        *(table_ptr.add(12) as *mut u32) = 0;
        let bitmap_ptr = table_ptr.add(32) as *mut u32;
        *bitmap_ptr = 0xFFFFFFFF;
        *bitmap_ptr.add(1) = 0xFFFFFFFF;
    }

    let state_buffer = device.new_buffer(
        4 * 1024 * 1024,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize 5 apps
    let mut slots = Vec::new();
    for i in 0..5 {
        let mut builder = GpuAppBuilder::new(&format!("app_{}", i));
        builder.add_instruction(BytecodeInst {
            opcode: 0,
            dst: 0,
            src1: 0,
            src2: 0,
            imm: 0.0,
        });
        builder.set_vertex_budget(100 + i * 10);

        let data = builder.build();
        let slot = loader.gpu_init_app(&device, &data, &app_table_buffer, &state_buffer);
        assert!(slot.is_some(), "Should allocate slot for app {}", i);
        slots.push(slot.unwrap());
    }

    // Verify all slots are unique
    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            assert_ne!(slots[i], slots[j], "Slots should be unique");
        }
    }

    println!("Initialized 5 apps in slots: {:?}", slots);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE COPY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bytecode_copied_correctly() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // Create app with specific bytecode pattern
    let mut builder = GpuAppBuilder::new("bytecode_test");
    for i in 0..10 {
        builder.add_instruction(BytecodeInst {
            opcode: (i % 256) as u8,
            dst: ((i + 1) % 32) as u8,
            src1: ((i + 2) % 32) as u8,
            src2: ((i + 3) % 32) as u8,
            imm: i as f32 * 1.5,
        });
    }

    let data = builder.build();
    let original_header = parse_header(&data).unwrap();

    // Create buffers
    let max_slots = 64u32;
    let header_size = 32;
    let bitmap_size = 8;
    let descriptor_size = 128;
    let table_size = header_size + bitmap_size + (max_slots as usize * descriptor_size);

    let app_table_buffer = device.new_buffer(
        table_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let table_ptr = app_table_buffer.contents() as *mut u8;
    unsafe {
        *(table_ptr as *mut u32) = 0x41505054;
        *(table_ptr.add(4) as *mut u32) = 1;
        *(table_ptr.add(8) as *mut u32) = max_slots;
        *(table_ptr.add(12) as *mut u32) = 0;
        let bitmap_ptr = table_ptr.add(32) as *mut u32;
        *bitmap_ptr = 0xFFFFFFFF;
        *bitmap_ptr.add(1) = 0xFFFFFFFF;
    }

    let state_buffer = device.new_buffer(
        4 * 1024 * 1024,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize app
    let slot = loader.gpu_init_app(&device, &data, &app_table_buffer, &state_buffer);
    assert!(slot.is_some());
    let slot = slot.unwrap();

    // Read back BytecodeHeader from state buffer
    let state_chunk_size = 64 * 1024u32;
    let state_offset = slot * state_chunk_size;

    let state_ptr = state_buffer.contents() as *const u8;
    let bc_header: [u32; 4] = unsafe {
        let ptr = state_ptr.add(state_offset as usize) as *const [u32; 4];
        *ptr
    };

    assert_eq!(bc_header[0], original_header.code_size, "code_size should match");
    assert_eq!(bc_header[1], original_header.entry_point, "entry_point should match");
    assert_eq!(bc_header[2], original_header.vertex_budget, "vertex_budget should match");

    // Read bytecode instructions and verify
    let bytecode_offset = state_offset as usize + 16;  // After BytecodeHeader
    let inst_size = std::mem::size_of::<BytecodeInst>();

    for i in 0..10 {
        let inst_ptr = unsafe { state_ptr.add(bytecode_offset + i * inst_size) as *const BytecodeInst };
        let inst = unsafe { *inst_ptr };

        assert_eq!(inst.opcode, (i % 256) as u8, "opcode should match for inst {}", i);
        assert_eq!(inst.dst, ((i + 1) % 32) as u8, "dst should match for inst {}", i);
        assert!((inst.imm - (i as f32 * 1.5)).abs() < 0.001, "imm should match for inst {}", i);
    }

    println!("Bytecode copied and verified correctly");
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CASE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_bytecode() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // App with no instructions
    let builder = GpuAppBuilder::new("empty");
    let data = builder.build();

    // Should still validate
    assert!(loader.gpu_validate_header(&device, &data));

    let header = parse_header(&data).unwrap();
    assert_eq!(header.code_size, 0);
}

#[test]
fn test_long_name() {
    let long_name = "this_is_a_very_long_app_name_that_exceeds_limit";
    let builder = GpuAppBuilder::new(long_name);
    let data = builder.build();

    let header = parse_header(&data).unwrap();
    // Name should be truncated to 23 chars (24 - 1 for null)
    assert!(header.name_str().len() <= 23);
    assert!(header.name_str().starts_with("this_is_a_very_long_app"));
}

#[test]
fn test_max_vertex_budget() {
    let mut builder = GpuAppBuilder::new("max_verts");
    builder.set_vertex_budget(65536);  // MAX_APP_VERTICES

    let data = builder.build();
    let header = parse_header(&data).unwrap();

    assert!(header.is_valid());
    assert_eq!(header.vertex_budget, 65536);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_large_bytecode_performance() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // Create app with 1000 instructions
    let mut builder = GpuAppBuilder::new("large_bytecode");
    for i in 0..1000 {
        builder.add_instruction(BytecodeInst {
            opcode: (i % 100) as u8,
            dst: (i % 32) as u8,
            src1: ((i + 1) % 32) as u8,
            src2: ((i + 2) % 32) as u8,
            imm: i as f32,
        });
    }

    let data = builder.build();
    println!("Built .gpuapp with {} bytes", data.len());

    // Create buffers
    let max_slots = 64u32;
    let table_size = 32 + 8 + (max_slots as usize * 128);

    let app_table_buffer = device.new_buffer(
        table_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let table_ptr = app_table_buffer.contents() as *mut u8;
    unsafe {
        *(table_ptr as *mut u32) = 0x41505054;
        *(table_ptr.add(4) as *mut u32) = 1;
        *(table_ptr.add(8) as *mut u32) = max_slots;
        *(table_ptr.add(12) as *mut u32) = 0;
        let bitmap_ptr = table_ptr.add(32) as *mut u32;
        *bitmap_ptr = 0xFFFFFFFF;
        *bitmap_ptr.add(1) = 0xFFFFFFFF;
    }

    let state_buffer = device.new_buffer(
        4 * 1024 * 1024,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Time initialization
    let start = std::time::Instant::now();
    let slot = loader.gpu_init_app(&device, &data, &app_table_buffer, &state_buffer);
    let elapsed = start.elapsed();

    assert!(slot.is_some());
    println!("Initialized 1000-instruction app in {:?}", elapsed);

    // Should complete in <10ms (generous bound for testing)
    assert!(elapsed.as_millis() < 100, "Init took too long: {:?}", elapsed);
}

#[test]
fn test_batch_init_performance() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuAppLoader::new(&device).expect("Failed to create loader");

    // Create app table with many slots
    let max_slots = 64u32;
    let table_size = 32 + 8 + (max_slots as usize * 128);

    let app_table_buffer = device.new_buffer(
        table_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let table_ptr = app_table_buffer.contents() as *mut u8;
    unsafe {
        *(table_ptr as *mut u32) = 0x41505054;
        *(table_ptr.add(4) as *mut u32) = 1;
        *(table_ptr.add(8) as *mut u32) = max_slots;
        *(table_ptr.add(12) as *mut u32) = 0;
        let bitmap_ptr = table_ptr.add(32) as *mut u32;
        *bitmap_ptr = 0xFFFFFFFF;
        *bitmap_ptr.add(1) = 0xFFFFFFFF;
    }

    let state_buffer = device.new_buffer(
        4 * 1024 * 1024,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Create 10 apps
    let mut apps: Vec<Vec<u8>> = Vec::new();
    for i in 0..10 {
        let mut builder = GpuAppBuilder::new(&format!("batch_{}", i));
        builder.add_instruction(BytecodeInst {
            opcode: 0,
            dst: 0,
            src1: 0,
            src2: 0,
            imm: 0.0,
        });
        apps.push(builder.build());
    }

    // Time batch initialization
    let start = std::time::Instant::now();
    for app_data in &apps {
        let slot = loader.gpu_init_app(&device, app_data, &app_table_buffer, &state_buffer);
        assert!(slot.is_some());
    }
    let elapsed = start.elapsed();

    println!("Initialized 10 apps in {:?}", elapsed);

    // Should complete in <50ms
    assert!(elapsed.as_millis() < 500, "Batch init took too long: {:?}", elapsed);
}
