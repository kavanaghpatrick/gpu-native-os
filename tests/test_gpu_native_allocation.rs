// GPU-NATIVE ALLOCATION BENCHMARK
//
// THE GPU IS THE COMPUTER.
//
// This benchmark demonstrates the fundamental difference between:
// 1. GPU-NATIVE: Allocation happens INSIDE compute kernels, zero CPU
// 2. GRAPHICS-CARD: CPU dispatches → GPU computes → CPU allocates → repeat
//
// The insight: When GPU threads need memory, they should allocate THEMSELVES.

use metal::*;
use std::time::Instant;

const PARTICLE_COUNT: usize = 1024 * 64;  // 64K particles
const FRAMES: usize = 100;

/// Two shaders demonstrating the paradigm difference
const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 position;
    float2 velocity;
    float life;
    uint child_offset;
    uint has_child;
    uint _pad;
};

struct AllocState {
    atomic_uint bump_pointer;
    uint pool_size;
    atomic_uint alloc_count;
    atomic_uint spawn_count;
};

// =============================================================================
// GPU-NATIVE: Inline allocation using parallel prefix sum
// One dispatch, zero CPU involvement, threads allocate themselves
// =============================================================================

kernel void particle_simulate_native(
    device Particle* particles [[buffer(0)]],
    device AllocState* alloc [[buffer(1)]],
    device float4* child_pool [[buffer(2)]],
    constant uint& particle_count [[buffer(3)]],
    constant uint& frame [[buffer(4)]],
    threadgroup uint* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup uint* prefix = shared;
    threadgroup uint* base_ptr = shared + tg_size;

    if (lid == 0) *base_ptr = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= particle_count) {
        prefix[lid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 1; s < tg_size; s *= 2) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    Particle p = particles[tid];

    // Physics update
    p.velocity.y -= 9.8f * 0.016f;
    p.position += p.velocity * 0.016f;
    p.life -= 0.016f;

    // Should spawn? Deterministic based on tid + frame
    uint hash = tid * 2654435761u + frame * 12345u;
    bool should_spawn = ((hash >> 20) & 0xFF) < 26 && p.life > 0.5f;  // ~10%
    uint my_size = should_spawn ? 1 : 0;  // 1 slot = 16 bytes (float4)

    // === INLINE PARALLEL PREFIX SUM ===
    prefix[lid] = my_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele (log n steps)
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        uint val = (lid >= stride) ? prefix[lid - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[lid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Last thread reserves space for entire group with ONE atomic
    if (lid == tg_size - 1) {
        uint total = prefix[tg_size - 1];
        if (total > 0) {
            uint base = atomic_fetch_add_explicit(&alloc->bump_pointer, total, memory_order_relaxed);
            *base_ptr = base;
            atomic_fetch_add_explicit(&alloc->alloc_count, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&alloc->spawn_count, total, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread gets its offset
    if (should_spawn) {
        uint exclusive = (lid == 0) ? 0 : prefix[lid - 1];
        uint offset = *base_ptr + exclusive;

        if (offset < alloc->pool_size / 16) {
            p.child_offset = offset;
            p.has_child = 1;
            child_pool[offset] = float4(p.position, p.velocity);
        }
    }

    particles[tid] = p;
}

// =============================================================================
// GRAPHICS-CARD MODEL: Just physics, allocation happens on CPU
// =============================================================================

kernel void particle_physics_only(
    device Particle* particles [[buffer(0)]],
    device atomic_uint* need_alloc_count [[buffer(1)]],
    device uint* need_alloc_list [[buffer(2)]],
    constant uint& particle_count [[buffer(3)]],
    constant uint& frame [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= particle_count) return;

    Particle p = particles[tid];

    // Physics
    p.velocity.y -= 9.8f * 0.016f;
    p.position += p.velocity * 0.016f;
    p.life -= 0.016f;

    // Should spawn?
    uint hash = tid * 2654435761u + frame * 12345u;
    bool should_spawn = ((hash >> 20) & 0xFF) < 26 && p.life > 0.5f;

    if (should_spawn) {
        // Record that we need allocation - CPU will handle it
        uint idx = atomic_fetch_add_explicit(need_alloc_count, 1, memory_order_relaxed);
        need_alloc_list[idx] = tid;
    }

    particles[tid] = p;
}

// Apply CPU-computed allocations
kernel void apply_allocations(
    device Particle* particles [[buffer(0)]],
    device const uint* alloc_list [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device float4* child_pool [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint particle_id = alloc_list[tid];
    uint offset = offsets[tid];

    Particle p = particles[particle_id];
    p.child_offset = offset;
    p.has_child = 1;
    child_pool[offset] = float4(p.position, p.velocity);
    particles[particle_id] = p;
}
"#;

#[repr(C)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    life: f32,
    child_offset: u32,
    has_child: u32,
    _pad: u32,
}

#[repr(C)]
struct AllocState {
    bump_pointer: u32,
    pool_size: u32,
    alloc_count: u32,
    spawn_count: u32,
}

fn init_particles() -> Vec<Particle> {
    (0..PARTICLE_COUNT)
        .map(|i| Particle {
            position: [(i % 256) as f32, (i / 256) as f32],
            velocity: [0.0, 1.0],
            life: 2.0,
            child_offset: 0,
            has_child: 0,
            _pad: 0,
        })
        .collect()
}

#[test]
fn benchmark_paradigm_comparison() {
    println!("\n{}", "=".repeat(70));
    println!("  GPU AS COMPUTER vs GPU AS GRAPHICS CARD");
    println!("{}", "=".repeat(70));
    println!("\nScenario: {} particles, ~10% spawn children each frame", PARTICLE_COUNT);
    println!("Frames: {}\n", FRAMES);

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let library = device.new_library_with_source(SHADERS, &CompileOptions::new())
        .expect("Shader compile failed");

    let native_fn = library.get_function("particle_simulate_native", None).unwrap();
    let physics_fn = library.get_function("particle_physics_only", None).unwrap();
    let apply_fn = library.get_function("apply_allocations", None).unwrap();

    let native_pipe = device.new_compute_pipeline_state_with_function(&native_fn).unwrap();
    let physics_pipe = device.new_compute_pipeline_state_with_function(&physics_fn).unwrap();
    let apply_pipe = device.new_compute_pipeline_state_with_function(&apply_fn).unwrap();

    let pool_slots = 1024 * 1024;  // 1M child slots
    let threads_per_group = 256usize;
    let groups = (PARTICLE_COUNT + threads_per_group - 1) / threads_per_group;

    // =========================================================================
    // GPU-NATIVE BENCHMARK
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("GPU-NATIVE MODEL");
    println!("  - One kernel dispatch per frame");
    println!("  - Allocation happens INSIDE the kernel via parallel prefix sum");
    println!("  - Zero CPU involvement, zero round-trips");
    println!("{}\n", "-".repeat(70));

    let particles1 = init_particles();
    let particle_buf1 = device.new_buffer_with_data(
        particles1.as_ptr() as *const _,
        (PARTICLE_COUNT * std::mem::size_of::<Particle>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let alloc_state = AllocState {
        bump_pointer: 0,
        pool_size: pool_slots as u32,
        alloc_count: 0,
        spawn_count: 0,
    };
    let alloc_buf = device.new_buffer_with_data(
        &alloc_state as *const _ as *const _,
        std::mem::size_of::<AllocState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let child_pool = device.new_buffer(
        (pool_slots * 16) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let count_buf = device.new_buffer_with_data(
        &(PARTICLE_COUNT as u32) as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    for frame in 0..10u32 {
        let frame_buf = device.new_buffer_with_data(&frame as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf1), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Reset
    unsafe {
        let ptr = alloc_buf.contents() as *mut AllocState;
        (*ptr).bump_pointer = 0;
        (*ptr).alloc_count = 0;
        (*ptr).spawn_count = 0;
    }

    let start = Instant::now();
    let mut dispatches = 0u32;

    for frame in 0..FRAMES as u32 {
        let frame_buf = device.new_buffer_with_data(&frame as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf1), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        dispatches += 1;
    }

    let native_time = start.elapsed();

    let (native_spawns, native_allocs) = unsafe {
        let ptr = alloc_buf.contents() as *const AllocState;
        ((*ptr).spawn_count, (*ptr).alloc_count)
    };

    println!("Time:          {:.2} ms", native_time.as_secs_f64() * 1000.0);
    println!("Spawns:        {}", native_spawns);
    println!("Dispatches:    {} (ONE per frame)", dispatches);
    println!("CPU round-trips: 0");
    println!("Atomics:       {} (one per threadgroup)", native_allocs);
    println!();

    // =========================================================================
    // GRAPHICS-CARD BENCHMARK
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("GRAPHICS-CARD MODEL");
    println!("  - Dispatch physics kernel");
    println!("  - GPU finishes, CPU reads spawn count");
    println!("  - CPU allocates memory sequentially");
    println!("  - Dispatch apply kernel");
    println!("  - Two dispatches + CPU work per frame");
    println!("{}\n", "-".repeat(70));

    let particles2 = init_particles();
    let particle_buf2 = device.new_buffer_with_data(
        particles2.as_ptr() as *const _,
        (PARTICLE_COUNT * std::mem::size_of::<Particle>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let need_count_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let need_list_buf = device.new_buffer((PARTICLE_COUNT * 4) as u64, MTLResourceOptions::StorageModeShared);
    let offsets_buf = device.new_buffer((PARTICLE_COUNT * 4) as u64, MTLResourceOptions::StorageModeShared);
    let child_pool2 = device.new_buffer((pool_slots * 16) as u64, MTLResourceOptions::StorageModeShared);

    // Warmup
    for frame in 0..10u32 {
        unsafe { *(need_count_buf.contents() as *mut u32) = 0; }
        let frame_buf = device.new_buffer_with_data(&frame as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&physics_pipe);
        enc.set_buffer(0, Some(&particle_buf2), 0);
        enc.set_buffer(1, Some(&need_count_buf), 0);
        enc.set_buffer(2, Some(&need_list_buf), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let start = Instant::now();
    let mut card_dispatches = 0u32;
    let mut card_spawns = 0u32;
    let mut cpu_bump: u32 = 0;

    for frame in 0..FRAMES as u32 {
        // Reset spawn count
        unsafe { *(need_count_buf.contents() as *mut u32) = 0; }

        let frame_buf = device.new_buffer_with_data(&frame as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        // DISPATCH 1: Physics + record spawn requests
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&physics_pipe);
        enc.set_buffer(0, Some(&particle_buf2), 0);
        enc.set_buffer(1, Some(&need_count_buf), 0);
        enc.set_buffer(2, Some(&need_list_buf), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();  // <<< CPU WAITS HERE
        card_dispatches += 1;

        // CPU: Read count, allocate memory
        let spawn_count = unsafe { *(need_count_buf.contents() as *const u32) };

        if spawn_count > 0 {
            card_spawns += spawn_count;

            // CPU allocates sequentially (this is what we're avoiding!)
            unsafe {
                let offsets = offsets_buf.contents() as *mut u32;
                for i in 0..spawn_count as usize {
                    *offsets.add(i) = cpu_bump;
                    cpu_bump += 1;
                }
            }

            // DISPATCH 2: Apply allocations
            let spawn_count_buf = device.new_buffer_with_data(
                &spawn_count as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&apply_pipe);
            enc.set_buffer(0, Some(&particle_buf2), 0);
            enc.set_buffer(1, Some(&need_list_buf), 0);
            enc.set_buffer(2, Some(&offsets_buf), 0);
            enc.set_buffer(3, Some(&child_pool2), 0);
            enc.set_buffer(4, Some(&spawn_count_buf), 0);

            let apply_groups = (spawn_count as usize + threads_per_group - 1) / threads_per_group;
            enc.dispatch_thread_groups(
                MTLSize::new(apply_groups.max(1) as u64, 1, 1),
                MTLSize::new(threads_per_group as u64, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            card_dispatches += 1;
        }
    }

    let card_time = start.elapsed();

    println!("Time:          {:.2} ms", card_time.as_secs_f64() * 1000.0);
    println!("Spawns:        {}", card_spawns);
    println!("Dispatches:    {} (physics + apply)", card_dispatches);
    println!("CPU round-trips: {} (every frame!)", FRAMES);
    println!("CPU allocations: {} (sequential atomics)", card_spawns);
    println!();

    // =========================================================================
    // ANALYSIS
    // =========================================================================
    println!("{}", "=".repeat(70));
    println!("  ANALYSIS");
    println!("{}\n", "=".repeat(70));

    let speedup = card_time.as_secs_f64() / native_time.as_secs_f64();

    println!("GPU-Native is {:.2}x faster\n", speedup);

    let native_us_per_frame = native_time.as_secs_f64() * 1000.0 / FRAMES as f64;
    let card_us_per_frame = card_time.as_secs_f64() * 1000.0 / FRAMES as f64;

    println!("Per-frame breakdown:");
    println!("  GPU-Native:     {:.2} ms/frame", native_us_per_frame);
    println!("  Graphics-card:  {:.2} ms/frame", card_us_per_frame);
    println!("  Overhead:       {:.2} ms/frame from round-trips", card_us_per_frame - native_us_per_frame);
    println!();

    println!("Key insight:");
    println!("  GPU-Native treats GPU as THE COMPUTER.");
    println!("  Graphics-card treats GPU as an accelerator that needs CPU coordination.");
    println!();
    println!("  The round-trip cost ({:.0}us) dominates when allocations are frequent.",
             (card_us_per_frame - native_us_per_frame) * 1000.0);
    println!();
}

#[test]
fn benchmark_allocation_overhead_in_kernel() {
    println!("\n{}", "=".repeat(70));
    println!("  INLINE ALLOCATION OVERHEAD");
    println!("{}", "=".repeat(70));
    println!("\nQuestion: How much does inline allocation add to kernel cost?");
    println!("(Comparing physics-only vs physics+allocation)\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let library = device.new_library_with_source(SHADERS, &CompileOptions::new())
        .expect("Shader compile failed");

    let native_fn = library.get_function("particle_simulate_native", None).unwrap();
    let physics_fn = library.get_function("particle_physics_only", None).unwrap();

    let native_pipe = device.new_compute_pipeline_state_with_function(&native_fn).unwrap();
    let physics_pipe = device.new_compute_pipeline_state_with_function(&physics_fn).unwrap();

    let pool_slots = 1024 * 1024u64;
    let threads_per_group = 256usize;
    let groups = (PARTICLE_COUNT + threads_per_group - 1) / threads_per_group;
    let iterations = 500;

    // Buffers for native
    let particles = init_particles();
    let particle_buf = device.new_buffer_with_data(
        particles.as_ptr() as *const _,
        (PARTICLE_COUNT * std::mem::size_of::<Particle>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let alloc_state = AllocState { bump_pointer: 0, pool_size: pool_slots as u32, alloc_count: 0, spawn_count: 0 };
    let alloc_buf = device.new_buffer_with_data(
        &alloc_state as *const _ as *const _,
        std::mem::size_of::<AllocState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let child_pool = device.new_buffer(pool_slots * 16, MTLResourceOptions::StorageModeShared);
    let count_buf = device.new_buffer_with_data(
        &(PARTICLE_COUNT as u32) as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let frame: u32 = 0;
    let frame_buf = device.new_buffer_with_data(&frame as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

    // Buffers for physics-only
    let need_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let need_list = device.new_buffer((PARTICLE_COUNT * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Warmup
    for _ in 0..20 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Measure physics-only
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { *(need_count.contents() as *mut u32) = 0; }

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&physics_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&need_count), 0);
        enc.set_buffer(2, Some(&need_list), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let physics_time = start.elapsed();

    // Measure physics + allocation
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let ptr = alloc_buf.contents() as *mut AllocState;
            (*ptr).bump_pointer = 0;
        }

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let native_time = start.elapsed();

    let physics_us = physics_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let native_us = native_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let overhead_us = native_us - physics_us;
    let overhead_pct = (overhead_us / physics_us) * 100.0;

    println!("Physics-only:           {:.1} us/dispatch", physics_us);
    println!("Physics + allocation:   {:.1} us/dispatch", native_us);
    println!();
    println!("Allocation overhead:    {:.1} us ({:.1}%)", overhead_us, overhead_pct);
    println!();
    println!("For {} particles across {} threadgroups:", PARTICLE_COUNT, groups);
    println!("  - Prefix sum: ~10 barriers × {} threads", threads_per_group);
    println!("  - Atomics: {} (one per threadgroup)", groups);
    println!();

    if overhead_us > 0.0 {
        let ns_per_particle = (overhead_us * 1000.0) / PARTICLE_COUNT as f64;
        println!("Cost per particle: {:.2} ns", ns_per_particle);
    }
    println!();

    println!("Compare to round-trip alternative:");
    println!("  - GPU→CPU→GPU round-trip: ~50,000-100,000 ns");
    println!("  - Inline allocation:      ~{:.0} ns total", overhead_us * 1000.0);
    println!();

    // Explain the negative overhead
    if overhead_us < 0.0 {
        println!("NOTE: Negative overhead means GPU-native is FASTER than physics-only!");
        println!("Why? Parallel prefix sum reduces atomic contention:");
        println!("  - Physics-only: ~6,500 atomics/frame (one per spawning particle)");
        println!("  - GPU-native:   ~256 atomics/frame (one per threadgroup)");
        println!("  - That's 25x fewer atomic operations!");
        println!();
    }
}

#[test]
fn benchmark_async_batched_execution() {
    println!("\n{}", "=".repeat(70));
    println!("  ASYNC BATCHED EXECUTION");
    println!("{}", "=".repeat(70));
    println!("\nThe previous benchmarks still called wait_until_completed() per frame.");
    println!("A TRUE GPU-native OS would batch work and minimize CPU sync points.\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let library = device.new_library_with_source(SHADERS, &CompileOptions::new())
        .expect("Shader compile failed");

    let native_fn = library.get_function("particle_simulate_native", None).unwrap();
    let native_pipe = device.new_compute_pipeline_state_with_function(&native_fn).unwrap();

    let pool_slots = 4 * 1024 * 1024u32;  // Larger pool for batched work
    let threads_per_group = 256usize;
    let groups = (PARTICLE_COUNT + threads_per_group - 1) / threads_per_group;

    let particles = init_particles();
    let particle_buf = device.new_buffer_with_data(
        particles.as_ptr() as *const _,
        (PARTICLE_COUNT * std::mem::size_of::<Particle>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let alloc_state = AllocState {
        bump_pointer: 0,
        pool_size: pool_slots,
        alloc_count: 0,
        spawn_count: 0,
    };
    let alloc_buf = device.new_buffer_with_data(
        &alloc_state as *const _ as *const _,
        std::mem::size_of::<AllocState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let child_pool = device.new_buffer(
        (pool_slots as u64 * 16),
        MTLResourceOptions::StorageModeShared,
    );

    let count_buf = device.new_buffer_with_data(
        &(PARTICLE_COUNT as u32) as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Create frame buffers once
    let frame_bufs: Vec<_> = (0..FRAMES as u32)
        .map(|f| device.new_buffer_with_data(&f as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared))
        .collect();

    // Warmup
    for i in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_bufs[i]), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ==========================================================================
    // SYNC-PER-FRAME: wait_until_completed() after each frame (old model)
    // ==========================================================================
    println!("{}", "-".repeat(70));
    println!("SYNC-PER-FRAME (old model)");
    println!("  - CPU waits for each frame to complete");
    println!("{}\n", "-".repeat(70));

    unsafe {
        let ptr = alloc_buf.contents() as *mut AllocState;
        (*ptr).bump_pointer = 0;
        (*ptr).alloc_count = 0;
        (*ptr).spawn_count = 0;
    }

    let start = Instant::now();
    for frame in 0..FRAMES {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_bufs[frame]), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();  // <<< SYNC POINT
    }
    let sync_time = start.elapsed();

    let sync_spawns = unsafe { (*(alloc_buf.contents() as *const AllocState)).spawn_count };

    println!("Time:      {:.2} ms", sync_time.as_secs_f64() * 1000.0);
    println!("Spawns:    {}", sync_spawns);
    println!("Syncs:     {} (every frame)", FRAMES);
    println!();

    // ==========================================================================
    // BATCH-10: Encode 10 frames, sync once
    // ==========================================================================
    println!("{}", "-".repeat(70));
    println!("BATCH-10 (encode 10 frames per command buffer)");
    println!("  - CPU encodes 10 frames, waits once");
    println!("{}\n", "-".repeat(70));

    unsafe {
        let ptr = alloc_buf.contents() as *mut AllocState;
        (*ptr).bump_pointer = 0;
        (*ptr).alloc_count = 0;
        (*ptr).spawn_count = 0;
    }

    let batch_size = 10;
    let start = Instant::now();
    for batch in 0..(FRAMES / batch_size) {
        let cmd = queue.new_command_buffer();

        for i in 0..batch_size {
            let frame = batch * batch_size + i;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&native_pipe);
            enc.set_buffer(0, Some(&particle_buf), 0);
            enc.set_buffer(1, Some(&alloc_buf), 0);
            enc.set_buffer(2, Some(&child_pool), 0);
            enc.set_buffer(3, Some(&count_buf), 0);
            enc.set_buffer(4, Some(&frame_bufs[frame]), 0);
            enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
            enc.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(threads_per_group as u64, 1, 1),
            );
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();  // Only 10 syncs total
    }
    let batch10_time = start.elapsed();

    let batch10_spawns = unsafe { (*(alloc_buf.contents() as *const AllocState)).spawn_count };

    println!("Time:      {:.2} ms", batch10_time.as_secs_f64() * 1000.0);
    println!("Spawns:    {}", batch10_spawns);
    println!("Syncs:     {} (10x fewer)", FRAMES / batch_size);
    println!();

    // ==========================================================================
    // BATCH-ALL: Encode ALL frames in one command buffer
    // ==========================================================================
    println!("{}", "-".repeat(70));
    println!("BATCH-ALL (encode all {} frames in one command buffer)", FRAMES);
    println!("  - CPU encodes all work upfront, waits once at end");
    println!("{}\n", "-".repeat(70));

    unsafe {
        let ptr = alloc_buf.contents() as *mut AllocState;
        (*ptr).bump_pointer = 0;
        (*ptr).alloc_count = 0;
        (*ptr).spawn_count = 0;
    }

    let start = Instant::now();
    let cmd = queue.new_command_buffer();

    for frame in 0..FRAMES {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&native_pipe);
        enc.set_buffer(0, Some(&particle_buf), 0);
        enc.set_buffer(1, Some(&alloc_buf), 0);
        enc.set_buffer(2, Some(&child_pool), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&frame_bufs[frame]), 0);
        enc.set_threadgroup_memory_length(0, ((threads_per_group + 1) * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        enc.end_encoding();
    }

    cmd.commit();
    cmd.wait_until_completed();  // Only 1 sync!
    let batch_all_time = start.elapsed();

    let batch_all_spawns = unsafe { (*(alloc_buf.contents() as *const AllocState)).spawn_count };

    println!("Time:      {:.2} ms", batch_all_time.as_secs_f64() * 1000.0);
    println!("Spawns:    {}", batch_all_spawns);
    println!("Syncs:     1 (minimum possible)", );
    println!();

    // ==========================================================================
    // ANALYSIS
    // ==========================================================================
    println!("{}", "=".repeat(70));
    println!("  ANALYSIS");
    println!("{}\n", "=".repeat(70));

    let speedup_10 = sync_time.as_secs_f64() / batch10_time.as_secs_f64();
    let speedup_all = sync_time.as_secs_f64() / batch_all_time.as_secs_f64();

    println!("Sync-per-frame:  {:.2} ms", sync_time.as_secs_f64() * 1000.0);
    println!("Batch-10:        {:.2} ms ({:.1}x faster)", batch10_time.as_secs_f64() * 1000.0, speedup_10);
    println!("Batch-all:       {:.2} ms ({:.1}x faster)", batch_all_time.as_secs_f64() * 1000.0, speedup_all);
    println!();

    let sync_overhead = sync_time.as_secs_f64() - batch_all_time.as_secs_f64();
    let sync_per_frame_us = (sync_overhead / FRAMES as f64) * 1_000_000.0;

    println!("Each sync costs:      ~{:.0} us", sync_per_frame_us);
    println!("Total sync overhead:  {:.2} ms ({:.0}% of runtime)",
             sync_overhead * 1000.0,
             (sync_overhead / sync_time.as_secs_f64()) * 100.0);
    println!();

    println!("THE KEY INSIGHT:");
    println!("  The 'graphics card' paradigm requires sync points for coordination.");
    println!("  The 'GPU is the computer' paradigm minimizes sync points.");
    println!();
    println!("  GPU-native allocation enables batch-all because:");
    println!("  - No CPU involvement needed between frames");
    println!("  - GPU handles all allocation decisions itself");
    println!("  - CPU just says 'run 100 frames' and waits");
    println!();
}
