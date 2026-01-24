// PERSISTENT KERNEL: Threadgroups Talking to Each Other
//
// THE GPU IS THE COMPUTER.
//
// This demonstrates that threadgroups CAN communicate directly
// without CPU involvement. The only thing stopping us was the
// mental model of "dispatch = unit of work."

use metal::*;
use std::time::Instant;

const NUM_GROUPS: usize = 64;
const THREADS_PER_GROUP: usize = 256;
const ITERATIONS: usize = 100;

/// Persistent kernel where threadgroups coordinate directly
const PERSISTENT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GlobalState {
    // Barrier synchronization
    atomic_uint phase_counter;      // Which phase are we in?
    atomic_uint groups_arrived;     // How many groups reached barrier?

    // Shared work
    atomic_uint work_counter;       // Work items completed
    uint num_groups;
    uint iterations;

    // Output
    atomic_uint total_work_done;
};

// =============================================================================
// PERSISTENT KERNEL: Runs continuously, threadgroups sync via atomics
// =============================================================================

kernel void persistent_worker(
    device GlobalState* state [[buffer(0)]],
    device uint* output [[buffer(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each threadgroup runs for multiple iterations
    for (uint iter = 0; iter < state->iterations; iter++) {

        // =====================================================================
        // PHASE 1: Each group does independent work
        // =====================================================================

        // Simulate work: each thread increments counter
        if (lid == 0) {
            atomic_fetch_add_explicit(&state->work_counter, tg_size, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // INTER-GROUP BARRIER: Wait for ALL groups to finish Phase 1
        // This is the key - threadgroups synchronize WITHOUT CPU!
        // =====================================================================

        if (lid == 0) {
            // Signal that this group has arrived at barrier
            uint arrived = atomic_fetch_add_explicit(&state->groups_arrived, 1, memory_order_relaxed) + 1;

            // Last group to arrive advances the phase
            if (arrived == state->num_groups) {
                // Reset arrival counter for next barrier
                atomic_store_explicit(&state->groups_arrived, 0, memory_order_relaxed);
                // Advance phase counter (releases waiting groups)
                atomic_fetch_add_explicit(&state->phase_counter, 1, memory_order_relaxed);
            }

            // Wait for phase to advance (spin-wait)
            uint expected_phase = iter + 1;
            while (atomic_load_explicit(&state->phase_counter, memory_order_relaxed) < expected_phase) {
                // Spin - Metal lacks yield, so we just busy-wait
            }
        }

        // Sync within threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // PHASE 2: Collective work using results from Phase 1
        // All groups now see consistent state
        // =====================================================================

        if (lid == 0 && gid == 0) {
            // Only group 0, thread 0 reads the total
            uint total = atomic_load_explicit(&state->work_counter, memory_order_relaxed);
            output[iter] = total;
            atomic_fetch_add_explicit(&state->total_work_done, total, memory_order_relaxed);

            // Reset for next iteration
            atomic_store_explicit(&state->work_counter, 0, memory_order_relaxed);
        }

        // Another barrier before next iteration
        if (lid == 0) {
            uint arrived = atomic_fetch_add_explicit(&state->groups_arrived, 1, memory_order_relaxed) + 1;
            if (arrived == state->num_groups) {
                atomic_store_explicit(&state->groups_arrived, 0, memory_order_relaxed);
                atomic_fetch_add_explicit(&state->phase_counter, 1, memory_order_relaxed);
            }

            uint expected_phase = (iter + 1) * 2;
            while (atomic_load_explicit(&state->phase_counter, memory_order_relaxed) < expected_phase) {
                // Spin
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// TRADITIONAL MODEL: Separate kernels, CPU coordinates
// =============================================================================

kernel void phase1_kernel(
    device atomic_uint* work_counter [[buffer(0)]],
    uint tg_size [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (lid == 0) {
        atomic_fetch_add_explicit(work_counter, tg_size, memory_order_relaxed);
    }
}

kernel void phase2_kernel(
    device atomic_uint* work_counter [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device atomic_uint* total [[buffer(2)]],
    constant uint& iter [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (lid == 0 && gid == 0) {
        uint count = atomic_load_explicit(work_counter, memory_order_relaxed);
        output[iter] = count;
        atomic_fetch_add_explicit(total, count, memory_order_relaxed);
        atomic_store_explicit(work_counter, 0, memory_order_relaxed);
    }
}
"#;

#[repr(C)]
struct GlobalState {
    phase_counter: u32,
    groups_arrived: u32,
    work_counter: u32,
    num_groups: u32,
    iterations: u32,
    total_work_done: u32,
    _pad: [u32; 2],
}

#[test]
fn benchmark_persistent_vs_traditional() {
    println!("\n{}", "=".repeat(70));
    println!("  PERSISTENT KERNEL: THREADGROUPS TALKING DIRECTLY");
    println!("{}", "=".repeat(70));
    println!("\nScenario: {} threadgroups, {} iterations of 2-phase work", NUM_GROUPS, ITERATIONS);
    println!("Each iteration: Phase 1 (independent) → Barrier → Phase 2 (collective)\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let library = device.new_library_with_source(PERSISTENT_SHADER, &CompileOptions::new())
        .expect("Shader compile failed");

    let persistent_fn = library.get_function("persistent_worker", None).unwrap();
    let phase1_fn = library.get_function("phase1_kernel", None).unwrap();
    let phase2_fn = library.get_function("phase2_kernel", None).unwrap();

    let persistent_pipe = device.new_compute_pipeline_state_with_function(&persistent_fn).unwrap();
    let phase1_pipe = device.new_compute_pipeline_state_with_function(&phase1_fn).unwrap();
    let phase2_pipe = device.new_compute_pipeline_state_with_function(&phase2_fn).unwrap();

    // =========================================================================
    // PERSISTENT KERNEL BENCHMARK
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("PERSISTENT KERNEL (GPU-native)");
    println!("  - ONE dispatch");
    println!("  - Threadgroups synchronize via atomics in device memory");
    println!("  - Zero CPU involvement during execution");
    println!("{}\n", "-".repeat(70));

    let state = GlobalState {
        phase_counter: 0,
        groups_arrived: 0,
        work_counter: 0,
        num_groups: NUM_GROUPS as u32,
        iterations: ITERATIONS as u32,
        total_work_done: 0,
        _pad: [0; 2],
    };
    let state_buf = device.new_buffer_with_data(
        &state as *const _ as *const _,
        std::mem::size_of::<GlobalState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (ITERATIONS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&persistent_pipe);
    enc.set_buffer(0, Some(&state_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(NUM_GROUPS as u64, 1, 1),
        MTLSize::new(THREADS_PER_GROUP as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let persistent_time = start.elapsed();

    let total_work = unsafe {
        (*(state_buf.contents() as *const GlobalState)).total_work_done
    };

    println!("Time:           {:.2} ms", persistent_time.as_secs_f64() * 1000.0);
    println!("Dispatches:     1");
    println!("CPU syncs:      1 (at the end)");
    println!("Total work:     {}", total_work);
    println!();

    // =========================================================================
    // TRADITIONAL MODEL BENCHMARK
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("TRADITIONAL MODEL (Graphics-card)");
    println!("  - {} dispatches (2 per iteration)", ITERATIONS * 2);
    println!("  - CPU coordinates Phase 1 → Phase 2 transitions");
    println!("{}\n", "-".repeat(70));

    let work_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let output_buf2 = device.new_buffer((ITERATIONS * 4) as u64, MTLResourceOptions::StorageModeShared);
    let total_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    unsafe {
        *(work_counter.contents() as *mut u32) = 0;
        *(total_buf.contents() as *mut u32) = 0;
    }

    let start = Instant::now();

    for iter in 0..ITERATIONS as u32 {
        // Phase 1: Independent work
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&phase1_pipe);
        enc.set_buffer(0, Some(&work_counter), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(NUM_GROUPS as u64, 1, 1),
            MTLSize::new(THREADS_PER_GROUP as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();  // <<< CPU SYNC

        // Phase 2: Collective work
        let iter_buf = device.new_buffer_with_data(
            &iter as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&phase2_pipe);
        enc.set_buffer(0, Some(&work_counter), 0);
        enc.set_buffer(1, Some(&output_buf2), 0);
        enc.set_buffer(2, Some(&total_buf), 0);
        enc.set_buffer(3, Some(&iter_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(NUM_GROUPS as u64, 1, 1),
            MTLSize::new(THREADS_PER_GROUP as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();  // <<< CPU SYNC
    }

    let traditional_time = start.elapsed();

    let total_work2 = unsafe { *(total_buf.contents() as *const u32) };

    println!("Time:           {:.2} ms", traditional_time.as_secs_f64() * 1000.0);
    println!("Dispatches:     {}", ITERATIONS * 2);
    println!("CPU syncs:      {}", ITERATIONS * 2);
    println!("Total work:     {}", total_work2);
    println!();

    // =========================================================================
    // ANALYSIS
    // =========================================================================
    println!("{}", "=".repeat(70));
    println!("  ANALYSIS");
    println!("{}\n", "=".repeat(70));

    let speedup = traditional_time.as_secs_f64() / persistent_time.as_secs_f64();

    println!("Persistent kernel is {:.1}x faster\n", speedup);

    println!("Why?");
    println!("  Traditional: {} CPU→GPU round-trips", ITERATIONS * 2);
    println!("  Persistent:  1 CPU→GPU round-trip");
    println!();

    let overhead_per_sync = (traditional_time.as_secs_f64() - persistent_time.as_secs_f64())
                           / (ITERATIONS * 2 - 1) as f64;
    println!("  Each round-trip costs: ~{:.0} us", overhead_per_sync * 1_000_000.0);
    println!();

    println!("THE KEY INSIGHT:");
    println!("  Threadgroups CAN synchronize directly via device memory atomics.");
    println!("  The CPU is not required for coordination.");
    println!();
    println!("  The 'graphics card' API (dispatch-wait-dispatch) is a SOFTWARE");
    println!("  abstraction, not a hardware limitation.");
    println!();
    println!("  GPU-native OS design: Launch persistent kernels that run 'forever',");
    println!("  synchronizing internally, with CPU only involved for I/O.");
    println!();

    // Verify correctness
    let expected = (NUM_GROUPS * THREADS_PER_GROUP * ITERATIONS) as u32;
    assert_eq!(total_work, expected, "Persistent kernel work mismatch");
    assert_eq!(total_work2, expected, "Traditional kernel work mismatch");
}

#[test]
fn explain_the_barrier() {
    println!("\n{}", "=".repeat(70));
    println!("  HOW INTER-THREADGROUP BARRIERS WORK");
    println!("{}", "=".repeat(70));
    println!();

    println!("The pattern for threadgroups to synchronize WITHOUT CPU:\n");

    println!("```metal");
    println!("// Shared state in device memory");
    println!("device atomic_uint* groups_arrived;");
    println!("device atomic_uint* phase_counter;");
    println!();
    println!("// Each threadgroup's thread 0 runs this:");
    println!("void inter_group_barrier(uint num_groups, uint expected_phase) {{");
    println!("    // 1. Signal arrival");
    println!("    uint arrived = atomic_fetch_add(groups_arrived, 1) + 1;");
    println!();
    println!("    // 2. Last to arrive advances the phase");
    println!("    if (arrived == num_groups) {{");
    println!("        atomic_store(groups_arrived, 0);  // Reset for next barrier");
    println!("        atomic_fetch_add(phase_counter, 1);  // Release waiters");
    println!("    }}");
    println!();
    println!("    // 3. Wait for phase to advance");
    println!("    while (atomic_load(phase_counter) < expected_phase) {{");
    println!("        // Spin-wait (or yield)");
    println!("    }}");
    println!("}}");
    println!("```\n");

    println!("REQUIREMENTS:");
    println!("  1. All threadgroups must be resident (fit in GPU memory)");
    println!("  2. GPU scheduler must be fair (no starvation)");
    println!("  3. Must handle the 'last group hasn't launched yet' case");
    println!();

    println!("CUDA calls this 'Cooperative Groups' and has explicit support.");
    println!("Metal can do it with careful atomic usage (as shown above).");
    println!();

    println!("WHY THIS MATTERS:");
    println!("  - Enables GPU-native multi-phase algorithms");
    println!("  - Eliminates CPU as synchronization bottleneck");
    println!("  - Allows persistent kernels that run 'forever'");
    println!("  - Foundation for GPU-native OS: GPU manages itself");
    println!();
}
