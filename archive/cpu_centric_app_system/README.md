# Archived: CPU-Centric App System (Issues #148-#153)

## Why Archived

These files implemented a "GPU-resident" app system that was **still CPU-centric**:

- CPU processed dispatch lists every frame
- CPU looked up pipelines by handle
- CPU encoded Metal commands
- GPU just queued work for CPU to do

This violates the core principle: **THE GPU IS THE COMPUTER**.

## What Was Built

| File | Issue | Problem |
|------|-------|---------|
| `app_descriptor.rs` | #148 | Good data structure, but CPU manages it |
| `slot_allocator.rs` | #149 | GPU allocates, but CPU orchestrates |
| `gpu_memory_pool.rs` | #150 | GPU allocates, but CPU orchestrates |
| `pipeline_table.rs` | #152 | CPU-side pipeline storage with "dispatch list" |
| `app_lifecycle.rs` | #151 | CPU manages full lifecycle |
| `gpu_resident.rs` | #153 | Integration layer - all CPU-driven |

## The Problem

The "dispatch list" pattern:
```
GPU: "Here's a list of apps that need rendering"
CPU: "OK, I'll look up pipelines and encode commands for each"
GPU: "Thanks, I'll execute them"
```

This is just a complicated way of doing CPU-driven rendering.

## The Goal

True GPU autonomy:
```
GPU: "I have all the app descriptors. I have all the pipelines compiled into me.
      I'll decide what to run, bind my own buffers, and dispatch my own work.
      CPU, you just need to submit my command buffer once per frame."
```

## Correct Approach

1. **Persistent Mega-Kernel**: Single compute kernel with all app logic
   - GPU reads app descriptors
   - GPU calls app-specific functions via switch on `app_type`
   - No per-app CPU encoding

2. **Indirect Command Buffers (ICB)**: GPU encodes its own draw commands
   - Requires Argument Buffers Tier 2
   - GPU builds command buffer, CPU just submits

3. **Unified Buffer Model**: All app state in one GPU buffer
   - No per-app CPU buffer binding
   - GPU indexes into monolithic buffer

## Salvageable Parts

- `GpuAppDescriptor` struct layout is useful
- Memory pool block allocation concept is good
- Atomic free list pattern is correct

But the orchestration layer needs complete redesign.
