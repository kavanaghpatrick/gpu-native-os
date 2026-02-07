// Megakernel QA - GPU Vertex Buffer Diagnostic
//
// Reads back the GPU-generated vertex buffer and diagnoses rendering issues.
// This test reveals exactly what the GPU megakernel is producing.

use metal::*;
use rust_experiment::gpu_os::gpu_os::GpuOs;
use rust_experiment::gpu_os::gpu_app_system::RenderVertex;

fn main() {
    println!("Megakernel QA - GPU Vertex Buffer Diagnostic");
    println!("=============================================\n");

    let device = Device::system_default().expect("No Metal device");
    println!("GPU: {}\n", device.name());

    // Boot GPU OS
    let width = 1280.0;
    let height = 720.0;
    let mut os = GpuOs::boot_with_size(&device, width, height)
        .expect("Failed to boot GPU OS");

    println!("GPU OS booted: {}x{}\n", width, height);

    // Run a few frames to let the system stabilize
    for _ in 0..5 {
        os.run_frame();
    }

    // Get vertex count and buffer
    let vertex_count = os.total_vertex_count();
    let buffer = os.render_vertices_buffer();

    println!("=== VERTEX BUFFER ANALYSIS ===\n");
    println!("Total vertex count: {}", vertex_count);
    println!("Buffer length: {} bytes", buffer.length());
    println!("RenderVertex size: {} bytes", std::mem::size_of::<RenderVertex>());
    println!("Max vertices at this stride: {}\n", buffer.length() / std::mem::size_of::<RenderVertex>() as u64);

    // Read back vertices
    let vertex_ptr = buffer.contents() as *const RenderVertex;

    if vertex_count == 0 {
        println!("ERROR: No vertices generated!");
        return;
    }

    // Analyze all vertices
    let mut valid_count = 0;
    let mut visible_count = 0;
    let mut in_clip_space = 0;
    let mut has_color = 0;
    let mut zero_position = 0;
    let mut nan_count = 0;
    let mut inf_count = 0;

    println!("=== FIRST 30 VERTICES ===\n");

    for i in 0..vertex_count as usize {
        let v = unsafe { &*vertex_ptr.add(i) };

        // Check for NaN/Inf
        let has_nan = v.position.iter().any(|x| x.is_nan()) || v.color.iter().any(|x| x.is_nan());
        let has_inf = v.position.iter().any(|x| x.is_infinite()) || v.color.iter().any(|x| x.is_infinite());

        if has_nan { nan_count += 1; }
        if has_inf { inf_count += 1; }

        // Check if position is all zeros
        if v.position[0] == 0.0 && v.position[1] == 0.0 && v.position[2] == 0.0 {
            zero_position += 1;
        }

        // Check if position would be in clip space after transform [-1, 1]
        let clip_x = (v.position[0] / width) * 2.0 - 1.0;
        let clip_y = 1.0 - (v.position[1] / height) * 2.0;
        let in_clip = clip_x >= -1.0 && clip_x <= 1.0 && clip_y >= -1.0 && clip_y <= 1.0;

        if in_clip { in_clip_space += 1; }

        // Check if has visible color (alpha > 0)
        if v.color[3] > 0.0 { has_color += 1; }

        // Check if would be visible (in clip space AND has alpha)
        if in_clip && v.color[3] > 0.0 { visible_count += 1; }

        // Valid = not NaN, not Inf, not zero
        if !has_nan && !has_inf && (v.position[0] != 0.0 || v.position[1] != 0.0) {
            valid_count += 1;
        }

        // Print first 30 vertices
        if i < 30 {
            let status = if has_nan { "NaN!" }
                else if has_inf { "Inf!" }
                else if v.color[3] == 0.0 { "a=0" }
                else if !in_clip { "OOB" }
                else { "OK" };
            println!("V{:4}: pos=({:8.1}, {:8.1}, {:4.2}) rgba=({:.2},{:.2},{:.2},{:.2}) {}",
                i,
                v.position[0], v.position[1], v.position[2],
                v.color[0], v.color[1], v.color[2], v.color[3],
                status);
        }
    }

    println!("\n=== VERTEX STATISTICS ===\n");
    println!("Total vertices:     {:6}", vertex_count);
    println!("Valid (not NaN/0):  {:6} ({:.1}%)", valid_count, 100.0 * valid_count as f32 / vertex_count as f32);
    println!("Zero position:      {:6} ({:.1}%)", zero_position, 100.0 * zero_position as f32 / vertex_count as f32);
    println!("Has NaN:            {:6}", nan_count);
    println!("Has Inf:            {:6}", inf_count);
    println!("In clip space:      {:6} ({:.1}%)", in_clip_space, 100.0 * in_clip_space as f32 / vertex_count as f32);
    println!("Has color (a>0):    {:6} ({:.1}%)", has_color, 100.0 * has_color as f32 / vertex_count as f32);
    println!("Visible (clip+a):   {:6} ({:.1}%)", visible_count, 100.0 * visible_count as f32 / vertex_count as f32);

    // Analyze triangles
    println!("\n=== TRIANGLE ANALYSIS ===\n");

    let tri_count = vertex_count / 3;
    println!("Triangle count: {}", tri_count);

    let mut zero_area_tris = 0;
    let mut valid_tris = 0;
    let mut cw_tris = 0;
    let mut ccw_tris = 0;

    for t in 0..tri_count as usize {
        let i = t * 3;
        let v0 = unsafe { &*vertex_ptr.add(i) };
        let v1 = unsafe { &*vertex_ptr.add(i + 1) };
        let v2 = unsafe { &*vertex_ptr.add(i + 2) };

        // 2D signed area (cross product z-component)
        let area = (v1.position[0] - v0.position[0]) * (v2.position[1] - v0.position[1]) -
                   (v2.position[0] - v0.position[0]) * (v1.position[1] - v0.position[1]);

        if area.abs() < 0.001 {
            zero_area_tris += 1;
        } else {
            valid_tris += 1;
            if area > 0.0 { ccw_tris += 1; } else { cw_tris += 1; }
        }

        // Print first 10 triangles
        if t < 10 {
            let winding = if area.abs() < 0.001 { "DEGEN" } else if area > 0.0 { "CCW" } else { "CW" };
            println!("Tri {:3}: area={:12.1} {}", t, area.abs(), winding);
            println!("  v0: ({:8.1}, {:8.1}) rgba=({:.2},{:.2},{:.2},{:.2})",
                v0.position[0], v0.position[1], v0.color[0], v0.color[1], v0.color[2], v0.color[3]);
            println!("  v1: ({:8.1}, {:8.1})",
                v1.position[0], v1.position[1]);
            println!("  v2: ({:8.1}, {:8.1})\n",
                v2.position[0], v2.position[1]);
        }
    }

    println!("Degenerate (zero-area): {} ({:.1}%)", zero_area_tris, 100.0 * zero_area_tris as f32 / tri_count as f32);
    println!("Valid triangles:        {} ({:.1}%)", valid_tris, 100.0 * valid_tris as f32 / tri_count as f32);
    println!("  - CCW (counter-clockwise): {}", ccw_tris);
    println!("  - CW (clockwise):          {}", cw_tris);

    // Check slot boundaries
    println!("\n=== SLOT BOUNDARY ANALYSIS ===\n");
    println!("(1024 vertices per slot)\n");

    let verts_per_slot = 1024;
    for slot in 0..8 {
        let start = slot * verts_per_slot;
        if start >= vertex_count as usize { break; }

        let v = unsafe { &*vertex_ptr.add(start) };
        let slot_end = std::cmp::min(start + verts_per_slot, vertex_count as usize);

        // Count non-zero vertices in this slot
        let mut slot_non_zero = 0;
        let mut slot_visible = 0;
        for i in start..slot_end {
            let v = unsafe { &*vertex_ptr.add(i) };
            if v.position[0] != 0.0 || v.position[1] != 0.0 {
                slot_non_zero += 1;
            }
            if v.color[3] > 0.0 {
                slot_visible += 1;
            }
        }

        let app_name = match slot {
            0 => "window_chrome",
            1 => "dock",
            2 => "menubar",
            3 => "compositor",
            _ => "user_app",
        };

        println!("Slot {} ({}): vertices {}..{}", slot, app_name, start, slot_end - 1);
        println!("  First 3 vertices:");
        for i in 0..3.min(slot_end - start) {
            let vi = unsafe { &*vertex_ptr.add(start + i) };
            println!("    V{}: pos=({:8.1}, {:8.1}, {:4.2}) rgba=({:.2},{:.2},{:.2},{:.2})",
                start + i, vi.position[0], vi.position[1], vi.position[2],
                vi.color[0], vi.color[1], vi.color[2], vi.color[3]);
        }
        println!("  Non-zero pos: {} / {}", slot_non_zero, slot_end - start);
        println!("  Visible (a>0): {} / {}\n", slot_visible, slot_end - start);
    }

    // Final diagnosis
    println!("=== DIAGNOSIS ===\n");

    let mut issues = Vec::new();

    if nan_count > 0 {
        issues.push(format!("CRITICAL: {} vertices have NaN - memory corruption!", nan_count));
    }
    if inf_count > 0 {
        issues.push(format!("CRITICAL: {} vertices have Inf - overflow!", inf_count));
    }
    if visible_count == 0 {
        issues.push("CRITICAL: No vertices would be visible!".to_string());
        if in_clip_space == 0 {
            issues.push("  - All positions outside screen bounds".to_string());
        }
        if has_color == 0 {
            issues.push("  - All vertices have alpha=0".to_string());
        }
    }
    if zero_position > vertex_count as usize / 2 {
        issues.push(format!("WARNING: {}% vertices at (0,0,0) - uninitialized?",
            100 * zero_position / vertex_count as usize));
    }
    if zero_area_tris > valid_tris {
        issues.push("WARNING: More degenerate triangles than valid ones".to_string());
    }
    if cw_tris > 0 && ccw_tris > 0 {
        issues.push(format!("INFO: Mixed winding - {} CCW, {} CW", ccw_tris, cw_tris));
    }

    if issues.is_empty() {
        println!("VERTEX DATA LOOKS CORRECT!");
        println!("  - {} visible vertices", visible_count);
        println!("  - {} valid triangles", valid_tris);
        println!("\nIf still not rendering, check:");
        println!("  1. Vertex shader clip space transform");
        println!("  2. Face culling (try disabling)");
        println!("  3. Blend mode for alpha < 1");
        println!("  4. Draw call parameters");
    } else {
        for issue in &issues {
            println!("{}", issue);
        }
    }

    println!();
}
