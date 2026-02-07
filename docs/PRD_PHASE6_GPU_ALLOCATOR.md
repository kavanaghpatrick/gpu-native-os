# PRD: Phase 6 - GPU Allocator (Vec, String, Box Support)

## Overview

**THE GPU IS THE COMPUTER.**

Phase 6 provides a GPU-resident memory allocator that enables Rust's `alloc` crate, allowing developers to use `Vec`, `String`, `Box`, and other heap-allocated types in GPU programs.

## Problem Statement

Currently, GPU programs cannot use dynamic allocation:
```rust
#![no_std]
let v = Vec::new();  // ❌ No allocator available
```

This forces developers to use fixed-size arrays, limiting program complexity.

## Solution: GPU Slab Allocator

We already have GPU memory management infrastructure. We expose it as a Rust global allocator.

**Research Note: Unified Memory Architecture**
On Apple Silicon, GPU and CPU share the same physical memory (unified memory architecture). This means:
- **No CPU-GPU copy needed**: Buffers allocated with `MTLResourceStorageModeShared` are directly accessible by both CPU and GPU
- **Zero-copy allocation**: `newBufferWithBytesNoCopy` can wrap existing memory without copying
- **Coherent memory**: Changes by GPU are immediately visible to CPU (and vice versa) without explicit synchronization commands

This is a major advantage over discrete GPU architectures (NVIDIA/AMD on x86) where data must be explicitly copied between CPU and GPU memory spaces. Our allocator can provide GPU-resident memory that the CPU can also inspect for debugging, without any transfer overhead.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Heap Buffer                       │
│  ┌─────────┬─────────┬─────────┬─────────┬───────────┐ │
│  │ 16-byte │ 32-byte │ 64-byte │ 128-byte│  Large    │ │
│  │  slabs  │  slabs  │  slabs  │  slabs  │  blocks   │ │
│  └─────────┴─────────┴─────────┴─────────┴───────────┘ │
│                                                         │
│  Free lists managed by atomic operations                │
└─────────────────────────────────────────────────────────┘
```

### Size Classes

| Class | Size | Use Case |
|-------|------|----------|
| 0 | 16 bytes | Small structs, Option<T> |
| 1 | 32 bytes | Small Vec headers |
| 2 | 64 bytes | Strings < 48 chars |
| 3 | 128 bytes | Medium allocations |
| 4 | 256 bytes | Larger allocations |
| 5 | 512 bytes | - |
| 6 | 1024 bytes | - |
| 7 | 2048+ bytes | Large block allocator |

### Intrinsics

```rust
// WASM imports (what Rust's alloc calls)
extern "C" {
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
    fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;
}
```

### GPU Implementation (Metal)

```metal
// Slab allocator state in heap buffer
struct SlabAllocator {
    uint free_heads[8];        // Free list heads per size class
    uint slab_counts[8];       // Number of slabs per class
    uint heap_base;            // Start of heap region
    uint heap_size;            // Total heap size
};

// Allocation (lock-free)
uint gpu_alloc(uint size, uint align, device SlabAllocator& alloc, device uint8_t* heap) {
    uint class_idx = size_to_class(size);
    uint slab_size = class_to_size(class_idx);

    // Pop from free list (atomic CAS loop)
    uint head;
    uint next;
    do {
        head = atomic_load_explicit(&alloc.free_heads[class_idx], memory_order_relaxed);
        if (head == 0xFFFFFFFF) {
            // Free list empty - allocate new slab
            return allocate_new_slab(class_idx, alloc, heap);
        }
        next = *(device uint*)(heap + head);  // Next pointer stored in free block
    } while (!atomic_compare_exchange_weak_explicit(
        &alloc.free_heads[class_idx], &head, next,
        memory_order_relaxed, memory_order_relaxed));

    return head;
}

// Deallocation (lock-free)
void gpu_dealloc(uint ptr, uint size, device SlabAllocator& alloc, device uint8_t* heap) {
    uint class_idx = size_to_class(size);

    // Push to free list (atomic)
    uint head;
    do {
        head = atomic_load_explicit(&alloc.free_heads[class_idx], memory_order_relaxed);
        *(device uint*)(heap + ptr) = head;  // Store next pointer
    } while (!atomic_compare_exchange_weak_explicit(
        &alloc.free_heads[class_idx], &head, ptr,
        memory_order_relaxed, memory_order_relaxed));
}

uint size_to_class(uint size) {
    if (size <= 16) return 0;
    if (size <= 32) return 1;
    if (size <= 64) return 2;
    if (size <= 128) return 3;
    if (size <= 256) return 4;
    if (size <= 512) return 5;
    if (size <= 1024) return 6;
    return 7;  // Large allocation
}
```

### Bytecode Opcodes

| Opcode | Name | Description |
|--------|------|-------------|
| 0x60 | ALLOC | `regs[d] = gpu_alloc(regs[s1], regs[s2])` |
| 0x61 | DEALLOC | `gpu_dealloc(regs[s1], regs[s2])` |
| 0x62 | REALLOC | `regs[d] = gpu_realloc(regs[s1], regs[s2], regs[s3])` |
| 0x63 | ALLOC_ZERO | `regs[d] = gpu_alloc_zeroed(regs[s1], regs[s2])` |

### Rust SDK

```rust
// gpu_sdk/src/alloc.rs
#![no_std]

use core::alloc::{GlobalAlloc, Layout};

pub struct GpuAllocator;

unsafe impl GlobalAlloc for GpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        extern "C" {
            fn __gpu_alloc(size: usize, align: usize) -> *mut u8;
        }
        __gpu_alloc(layout.size(), layout.align())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        extern "C" {
            fn __gpu_dealloc(ptr: *mut u8, size: usize, align: usize);
        }
        __gpu_dealloc(ptr, layout.size(), layout.align())
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        extern "C" {
            fn __gpu_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
        }
        __gpu_realloc(ptr, layout.size(), layout.align(), new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        extern "C" {
            fn __gpu_alloc_zeroed(size: usize, align: usize) -> *mut u8;
        }
        __gpu_alloc_zeroed(layout.size(), layout.align())
    }
}

#[global_allocator]
static ALLOCATOR: GpuAllocator = GpuAllocator;
```

### User Code

```rust
#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::boxed::Box;
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    // All of these now work!
    let mut numbers = Vec::new();
    numbers.push(1);
    numbers.push(2);
    numbers.push(3);

    let sum: i32 = numbers.iter().sum();

    let mut s = String::from("Hello");
    s.push_str(" GPU!");

    let boxed = Box::new(42);

    sum + *boxed  // Returns 48
}
```

## Test Cases

### Test 1: Vec Basic Operations
```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    v[0] + v[1] + v[2]  // 60
}
```

### Test 2: Vec Growth (Reallocation)
```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut v = Vec::new();
    for i in 0..100 {
        v.push(i);
    }
    v.iter().sum()  // 4950
}
```

### Test 3: String Operations
```rust
#![no_std]
extern crate alloc;
use alloc::string::String;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut s = String::new();
    s.push('H');
    s.push('i');
    s.len() as i32  // 2
}
```

### Test 4: Box Allocation
```rust
#![no_std]
extern crate alloc;
use alloc::boxed::Box;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let b = Box::new(42);
    *b
}
```

### Test 5: Nested Allocations
```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use alloc::boxed::Box;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut v: Vec<Box<i32>> = Vec::new();
    v.push(Box::new(1));
    v.push(Box::new(2));
    v.push(Box::new(3));
    v.iter().map(|b| **b).sum()  // 6
}
```

### Test 6: Deallocation (Drop)
```rust
#![no_std]
extern crate alloc;
use alloc::vec::Vec;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    for _ in 0..1000 {
        let v: Vec<i32> = (0..100).collect();
        // v is dropped here, memory freed
    }
    1  // Didn't run out of memory
}
```

### Test 7: BTreeMap (Complex Data Structure)
```rust
#![no_std]
extern crate alloc;
use alloc::collections::BTreeMap;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut map = BTreeMap::new();
    map.insert(1, 100);
    map.insert(2, 200);
    map.insert(3, 300);
    *map.get(&2).unwrap_or(&0)  // 200
}
```

## Memory Layout

```
GPU Heap Buffer (e.g., 64MB):
┌────────────────────────────────────────────────────────┐
│ Allocator Metadata (4KB)                               │
│   - Free list heads                                    │
│   - Slab counts                                        │
│   - Statistics                                         │
├────────────────────────────────────────────────────────┤
│ Size Class 0: 16-byte slabs (1MB)                      │
│   [block][block][block]...                             │
├────────────────────────────────────────────────────────┤
│ Size Class 1: 32-byte slabs (2MB)                      │
│   [block][block][block]...                             │
├────────────────────────────────────────────────────────┤
│ ...                                                    │
├────────────────────────────────────────────────────────┤
│ Large Block Region (remaining space)                   │
│   Variable-size allocations                            │
└────────────────────────────────────────────────────────┘
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Allocation latency | < 100 cycles |
| Deallocation latency | < 50 cycles |
| Memory overhead | < 10% |
| Fragmentation | < 20% after 1M ops |
| Vec push throughput | > 100M/sec |

## Dependencies

- Phase 5 (function calls) - for calling alloc intrinsics
- GPU heap infrastructure (already exists)

## Risks

| Risk | Mitigation |
|------|------------|
| Memory exhaustion | Return null, let Rust panic |
| Fragmentation | Slab allocator minimizes this |
| Concurrent allocation contention | Lock-free design |
| Large allocations | Separate large block allocator |

## Research References: Academic GPU Allocators

Our slab allocator design is informed by proven academic implementations:

**BaM (Big Accelerator Memory) - ASPLOS 2023**
- Demonstrates GPU-managed memory allocation at scale
- Uses lock-free slab allocation similar to our design
- Achieves near-native performance for GPU-resident data structures
- Paper: "BaM: A Case for Enabling Fine-grain High Throughput GPU-Orchestrated Access to Storage"

**GIDS (GPU-Initiated Direct Storage) - OSDI 2022**
- Shows GPU can manage its own memory without CPU involvement
- Lock-free allocation critical for avoiding SIMD divergence
- Validates our approach of per-size-class free lists with atomic CAS

**Key Insights from Research:**
1. Lock-free is mandatory - any lock causes SIMD divergence catastrophe
2. Size classes (slabs) dramatically reduce fragmentation vs general allocators
3. Atomic CAS loops have acceptable overhead when contention is low
4. Pre-allocated slab regions perform better than on-demand allocation
