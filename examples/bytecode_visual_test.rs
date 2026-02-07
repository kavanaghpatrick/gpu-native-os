//! Bytecode Visual Test
//!
//! Creates a window and runs a bytecode program to draw colored quads.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::*;
use std::mem;

fn main() {
    println!("=== Bytecode Visual Test ===\n");

    // Get Metal device
    let device = Device::system_default().expect("No Metal device found");
    println!("Device: {}", device.name());

    // Compile the shader
    println!("Compiling shader...");
    let library = device.new_library_with_source(
        GPU_APP_SYSTEM_SHADER,
        &CompileOptions::new(),
    ).expect("Failed to compile shader");

    let megakernel = library.get_function("gpu_app_megakernel_parallel", None)
        .expect("Failed to get megakernel function");
    let pipeline = device.new_compute_pipeline_state_with_function(&megakernel)
        .expect("Failed to create pipeline");

    println!("Shader compiled successfully");

    // Create bytecode program
    println!("\nCreating bytecode program...");
    let mut asm = BytecodeAssembler::new();

    // Draw a grid of colored squares
    let grid_size = 8;
    let cell_size = 50.0;
    let margin = 20.0;

    for y in 0..grid_size {
        for x in 0..grid_size {
            let px = margin + (x as f32) * (cell_size + 10.0);
            let py = margin + (y as f32) * (cell_size + 10.0);

            // Position (r4.xy)
            asm.setx(4, px);
            asm.sety(4, py);

            // Size (r5.xy)
            asm.setx(5, cell_size);
            asm.sety(5, cell_size);

            // Color based on position (creates gradient) (r6.xyzw)
            let r = x as f32 / (grid_size as f32 - 1.0);
            let g = y as f32 / (grid_size as f32 - 1.0);
            let b = 0.5;
            asm.setx(6, r);
            asm.sety(6, g);
            asm.setz(6, b);
            asm.setw(6, 1.0);

            // quad(pos_reg, size_reg, color_reg, depth)
            asm.quad(4, 5, 6, 0.5);
        }
    }

    asm.halt();

    let vertex_budget = (grid_size * grid_size * 6) as u32;
    let bytecode = asm.build(vertex_budget);
    println!("Generated {} instructions", (bytecode.len() - 16) / 8);

    // Create buffers
    // App table header
    let header = AppTableHeader {
        max_slots: 64,
        active_count: 1,
        free_bitmap: [0xFFFFFFFE, 0xFFFFFFFF],  // Slot 0 in use
        _pad: [0; 4],
    };

    // We need multiple buffers for the megakernel:
    // buffer(0): AppTableHeader
    // buffer(1): GpuAppDescriptor[]
    // buffer(2): unified_state
    // buffer(3): frame_number
    // buffer(4): unified_vertices
    // buffer(5): GpuWindow[]
    // buffer(6): window_count

    let header_buffer = device.new_buffer_with_data(
        &header as *const _ as *const _,
        mem::size_of::<AppTableHeader>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // State buffer (bytecode goes here)
    let state_size = bytecode.len() + 4096;  // Extra for app state after bytecode
    let state_buffer = device.new_buffer(state_size as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = state_buffer.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(bytecode.as_ptr(), ptr, bytecode.len());
    }

    // App descriptor
    let app = GpuAppDescriptor {
        flags: flags::ACTIVE | flags::VISIBLE | flags::DIRTY,
        app_type: app_type::BYTECODE,
        slot_id: 0,
        window_id: 0,
        state_offset: 0,
        state_size: state_size as u32,
        vertex_offset: 0,
        vertex_size: (vertex_budget as usize * mem::size_of::<RenderVertex>()) as u32,
        vertex_count: 0,
        priority: 1,
        thread_count: 1,
        ..Default::default()
    };

    // Create app buffer with space for 64 apps
    let apps_buffer = device.new_buffer(
        (64 * mem::size_of::<GpuAppDescriptor>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let ptr = apps_buffer.contents() as *mut GpuAppDescriptor;
        *ptr = app;
    }

    // Vertex buffer
    let vertex_buffer = device.new_buffer(
        (vertex_budget as usize * mem::size_of::<RenderVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Window buffer (empty for now)
    let window_buffer = device.new_buffer(
        (64 * mem::size_of::<GpuWindow>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Frame number
    let frame_number: u32 = 1;
    let frame_buffer = device.new_buffer_with_data(
        &frame_number as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Window count
    let window_count: u32 = 0;
    let window_count_buffer = device.new_buffer_with_data(
        &window_count as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Create command queue and run
    let queue = device.new_command_queue();
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&header_buffer), 0);
    encoder.set_buffer(1, Some(&apps_buffer), 0);
    encoder.set_buffer(2, Some(&state_buffer), 0);
    encoder.set_buffer(3, Some(&frame_buffer), 0);
    encoder.set_buffer(4, Some(&vertex_buffer), 0);
    encoder.set_buffer(5, Some(&window_buffer), 0);
    encoder.set_buffer(6, Some(&window_count_buffer), 0);

    // Issue #236 fix: Buffer 7 = screen_size (width, height as float2)
    let screen_size: [f32; 2] = [800.0, 600.0];
    encoder.set_bytes(
        7,
        std::mem::size_of::<[f32; 2]>() as u64,
        &screen_size as *const _ as *const _,
    );

    // Dispatch: 1 threadgroup (for slot 0), 256 threads
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();

    println!("\nRunning bytecode on GPU...");
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Read back results
    let app_result: &GpuAppDescriptor = unsafe {
        &*(apps_buffer.contents() as *const GpuAppDescriptor)
    };

    println!("\nResults:");
    println!("  App flags: 0x{:08x}", app_result.flags);
    println!("  Vertex count: {}", app_result.vertex_count);

    if app_result.vertex_count > 0 {
        println!("\n  First few vertices:");
        let vertices: &[RenderVertex] = unsafe {
            std::slice::from_raw_parts(
                vertex_buffer.contents() as *const RenderVertex,
                std::cmp::min(app_result.vertex_count as usize, 12),
            )
        };
        for (i, v) in vertices.iter().enumerate() {
            println!("    [{}] pos=({:.1}, {:.1}, {:.1}) color=({:.2}, {:.2}, {:.2}, {:.2})",
                     i, v.position[0], v.position[1], v.position[2],
                     v.color[0], v.color[1], v.color[2], v.color[3]);
        }
    }

    // Check if dirty flag was cleared
    let dirty_cleared = (app_result.flags & flags::DIRTY) == 0;
    println!("\n  Dirty flag cleared: {}", dirty_cleared);

    if app_result.vertex_count == vertex_budget {
        println!("\n SUCCESS: Generated expected {} vertices", vertex_budget);
    } else if app_result.vertex_count > 0 {
        println!("\n PARTIAL: Generated {} of {} expected vertices", app_result.vertex_count, vertex_budget);
    } else {
        println!("\n FAILED: No vertices generated");
    }
}

// Need to access the shader source
const GPU_APP_SYSTEM_SHADER: &str = rust_experiment::gpu_os::gpu_app_system::GPU_APP_SYSTEM_SHADER;
