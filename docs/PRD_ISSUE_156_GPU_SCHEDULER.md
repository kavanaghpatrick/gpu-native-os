# PRD: GPU Multi-App Scheduler (Issue #156)

## Overview

Implement GPU-native "scheduling" where each app evaluates ITSELF in parallel - no central scheduler.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| Priority queue data structure | Each thread evaluates priority predicate |
| Work stealing between threads | All threads run simultaneously |
| Central scheduler loop | Parallel predicate evaluation |
| "Pick next task" | "All tasks evaluate: should I run?" |

## The GPU Insight

On GPU, there IS no scheduler. All 64 app slots evaluate simultaneously in one cycle:

```metal
// WRONG: CPU-style scheduler
for (uint i = 0; i < app_count; i++) {
    if (should_run(apps[i])) {
        run(apps[i]);  // Sequential!
    }
}

// RIGHT: GPU-style parallel evaluation
// Each thread IS one app slot
uint slot_id = thread_position_in_grid;
if (should_i_run(apps[slot_id])) {
    run_my_update();  // 64 apps run simultaneously
}
```

## Design

### Parallel Predicate Evaluation

Each app evaluates its own "should I run?" predicate in O(1):

```metal
// Each app decides for ITSELF - no central scheduler
inline bool should_i_run(
    device const GpuAppDescriptor* app,
    uint current_frame
) {
    // Not active? Exit in 1 cycle
    if (!(app->flags & APP_FLAG_ACTIVE)) return false;

    // Not dirty? Exit in 1 cycle
    if (!(app->flags & APP_FLAG_DIRTY)) return false;

    // Suspended? Exit in 1 cycle
    if (app->flags & APP_FLAG_SUSPENDED) return false;

    // Starvation check: always run if starving
    uint frames_since_run = current_frame - app->last_run_frame;
    if (frames_since_run > STARVATION_THRESHOLD) return true;

    // Budget check: high priority always, low priority if budget available
    // (budget is just a counter, not a queue)
    return true;  // For MVP, always run if dirty
}
```

### Priority as Execution Order (Not Scheduling)

Instead of "scheduling" (picking what runs), use priority for:

1. **Frame budget**: High priority apps consume budget first
2. **Starvation prevention**: Boost priority after N frames without running
3. **Suspension**: Background apps can be suspended entirely

```metal
// Frame budget is just a counter - O(1) atomic
struct FrameBudget {
    atomic_uint remaining;    // Cycles remaining this frame
    uint per_frame_limit;     // Reset each frame
};

kernel void gpu_app_megakernel(
    device GpuAppDescriptor* apps [[buffer(0)]],
    device FrameBudget* budget [[buffer(1)]],
    constant uint& frame_number [[buffer(2)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= MAX_SLOTS) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    // O(1) predicate evaluation
    if (!should_i_run(app, frame_number)) return;

    // Estimate my cost (could be refined based on history)
    uint my_cost = app->thread_count;

    // Try to claim budget - O(1) atomic
    uint old_budget = atomic_fetch_sub_explicit(
        &budget->remaining, my_cost, memory_order_relaxed
    );

    if (old_budget < my_cost) {
        // Over budget - check priority
        if (app->priority < PRIORITY_HIGH) {
            // Roll back and skip
            atomic_fetch_add_explicit(&budget->remaining, my_cost, memory_order_relaxed);
            return;
        }
        // High priority runs anyway
    }

    // Run my update
    run_app_update(app, slot_id);

    app->last_run_frame = frame_number;
    app->flags &= ~APP_FLAG_DIRTY;
}
```

### Starvation Prevention (Parallel)

Each app checks its OWN starvation in O(1):

```metal
// Part of should_i_run() - each app checks itself
inline bool am_i_starving(
    device const GpuAppDescriptor* app,
    uint current_frame,
    uint threshold
) {
    return (current_frame - app->last_run_frame) > threshold;
}

// Auto-boost priority when starving
inline uint effective_priority(
    device const GpuAppDescriptor* app,
    uint current_frame
) {
    uint base = app->priority;

    // Starving apps get boosted
    if (am_i_starving(app, current_frame, 10)) {
        base = min(base + 1, PRIORITY_REALTIME);
    }

    return base;
}
```

### No Work Stealing (It's a CPU Concept)

Work stealing makes sense on CPU where:
- Threads finish at different times
- Idle threads should help busy ones

On GPU:
- All threads execute in lockstep within warp
- "Idle" threads just hit early return (1 cycle)
- No benefit from "stealing" - just run more threads

```metal
// WRONG: Work stealing
if (my_work_done) {
    steal_work_from_neighbor();  // Adds complexity, no benefit
}

// RIGHT: Just exit early
if (!should_i_run(app)) {
    return;  // 1 cycle, warp continues with other threads
}
```

## Implementation

### Rust API

```rust
impl GpuAppSystem {
    /// Set frame budget (cycles)
    pub fn set_frame_budget(&mut self, budget: u32) {
        // Write to GPU buffer
    }

    /// Set priority for an app
    pub fn set_priority(&mut self, slot: u32, priority: u32) {
        // Update app descriptor
    }

    /// Suspend an app (won't run until resumed)
    pub fn suspend(&mut self, slot: u32) {
        // Set SUSPENDED flag
    }

    /// Resume a suspended app
    pub fn resume(&mut self, slot: u32) {
        // Clear SUSPENDED flag, set DIRTY
    }

    /// Get scheduling stats
    pub fn scheduler_stats(&self) -> SchedulerStats {
        // Count by priority, starvation counts
    }
}

pub struct SchedulerStats {
    pub apps_by_priority: [u32; 4],
    pub starving_count: u32,
    pub suspended_count: u32,
    pub budget_remaining: u32,
}
```

## Tests

```rust
#[test]
fn test_all_apps_evaluate_simultaneously() {
    let mut system = GpuAppSystem::new(&device)?;

    // Launch 64 apps
    for _ in 0..64 {
        system.launch_app(app_type::CUSTOM, 1024, 512);
    }

    system.mark_all_dirty();
    system.run_frame();

    // All 64 should have run (parallel evaluation)
    for slot in 0..64 {
        if let Some(app) = system.get_app(slot) {
            assert_eq!(app.last_run_frame, 1, "App {} should have run", slot);
        }
    }
}

#[test]
fn test_suspended_apps_skip() {
    let mut system = GpuAppSystem::new(&device)?;

    let active = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    let suspended = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

    system.suspend(suspended);
    system.mark_all_dirty();
    system.run_frame();

    assert_eq!(system.get_app(active).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(suspended).unwrap().last_run_frame, 0);
}

#[test]
fn test_starvation_boost() {
    let mut system = GpuAppSystem::new(&device)?;

    let app = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.set_priority(app, priority::BACKGROUND);

    // Simulate starvation (don't mark dirty for many frames)
    for _ in 0..15 {
        system.run_frame();
    }

    // Now mark dirty - should run due to starvation boost
    system.mark_dirty(app);
    system.run_frame();

    // Check it ran despite being background priority
    assert!(system.get_app(app).unwrap().last_run_frame > 0);
}

#[test]
fn test_budget_enforcement() {
    let mut system = GpuAppSystem::new(&device)?;

    // Very low budget
    system.set_frame_budget(1000);

    // Launch many expensive apps
    for _ in 0..20 {
        let slot = system.launch_app(app_type::CUSTOM, 10000, 512).unwrap();
        system.set_priority(slot, priority::BACKGROUND);
    }

    system.mark_all_dirty();
    system.run_frame();

    // Some should have been skipped due to budget
    let stats = system.scheduler_stats();
    assert!(stats.budget_remaining == 0 || stats.apps_by_priority[0] < 20);
}
```

## Success Metrics

1. **Evaluation time**: O(1) per app (parallel predicate)
2. **No scheduling overhead**: Zero barriers, zero data structure traversal
3. **Starvation prevention**: All apps run at least every 10 frames
4. **Budget enforcement**: Frame time variance < 2ms
