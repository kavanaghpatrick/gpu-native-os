# AI Review Summary: GPU-Native OS Issues

**Reviewers**: Gemini, Grok-4, Codex
**Date**: 2026-01-23

---

## Critical Bugs by Issue

### Issue #1: Syscall Queue

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Ring buffer overflow (no full check) | ✓ | ✓ | ✓ | CRITICAL |
| GPU/CPU pointer mismatch (VA spaces) | ✓ | | ✓ | CRITICAL |
| Request status race (head before write) | ✓ | ✓ | ✓ | CRITICAL |
| Response correlation failure | ✓ | | | HIGH |
| Unchecked shared buffer bounds | ✓ | ✓ | | CRITICAL |

**Consensus Fix**: Add per-slot status field with release/acquire semantics, bounds checking, and use buffer offsets instead of raw pointers.

---

### Issue #2: GPU Event Loop

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Multi-threadgroup race on state write | ✓ | ✓ | | CRITICAL |
| draw_count never reset (overflow) | ✓ | | | CRITICAL |
| O(N) batch counting loop (TDR risk) | ✓ | | | HIGH |
| Sort key discards depth (key_low) | ✓ | ✓ | | CRITICAL |
| Float depth sorting breaks for negatives | ✓ | ✓ | | HIGH |

**Consensus Fix**: Split into separate kernel dispatches, use atomic_max for single state writer, fix sort key construction.

---

### Issue #3: Memory Manager

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| ABA problem in lock-free slab alloc | ✓ | | | CRITICAL |
| Batch alloc assumes contiguous memory | ✓ | | | CRITICAL |
| Cache lookup atomic on local copy | ✓ | | | HIGH |
| No large allocation handling (>16KB) | ✓ | ✓ | | CRITICAL |
| Framebuffer size mismatch (32MB vs 66MB) | | | ✓ | CRITICAL |

**Consensus Fix**: Add version counters to free list, implement large block allocator, fix framebuffer allocation.

---

### Issue #4: Widget System

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Widget struct alignment mismatch | ✓ | | | CRITICAL |
| Uninitialized vertex data for hidden | ✓ | | | HIGH |
| GPU timeout in child iteration | ✓ | | | CRITICAL |
| Hit-test assumes ID correlates with Z | ✓ | | | HIGH |
| Serial layout negates GPU benefit | ✓ | | | MEDIUM |

**Consensus Fix**: Verify struct layout with `#[repr(C)]`, initialize all vertices, parallelize child iteration.

---

### Issue #5: Input Processing

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Input buffer write before increment | ✓ | | | CRITICAL |
| Widget event buffer overflow | ✓ | ✓ | | CRITICAL |
| Unsigned integer underflow (cursor) | ✓ | | | HIGH |
| Multi-group race on single state | ✓ | ✓ | ✓ | CRITICAL |
| Ring buffer wraparound math wrong | | ✓ | | CRITICAL |

**Consensus Fix**: Write event before incrementing cursor, add bounds checking, ensure single SIMD group dispatch.

---

### Issue #6: Branchless Logic Library

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| when_eq claims epsilon but is exact | | ✓ | | MEDIUM |
| simd_broadcast_leader ignores param | | ✓ | | LOW |
| No critical bugs found | ✓ | | | - |

**Consensus**: Library is mostly sound. Minor documentation fixes needed.

---

### Issue #7: Work Sorting

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Radix sort stability violated (atomics) | ✓ | | | CRITICAL |
| Onesweep spin-loop deadlock | ✓ | ✓ | | CRITICAL |
| O(N) serial loop in draw commands | ✓ | ✓ | | CRITICAL |
| Float depth sorting breaks for negatives | ✓ | ✓ | | HIGH |
| Scatter loop underflow trick is brittle | | ✓ | | MEDIUM |

**Consensus Fix**: Use prefix sum instead of atomics for radix sort, avoid spin-locks, use parallel reduction for batch counting.

---

### Issue #8: Stream Compaction

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| out-of-bounds predicate[gid+1] access | | ✓ | | CRITICAL |
| Decoupled look-back can deadlock | | ✓ | | CRITICAL |
| Multiple "last" threads race on count | | ✓ | | HIGH |
| SIMD size hardcoded (32 vs 64) | | ✓ | | MEDIUM |
| Partial review (rate limited) | ✓ | | | - |

**Consensus Fix**: Add bounds checking, handle partial threadgroups, use proper leader election.

---

### Issue #9: SIMD Utilities

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Inactive lane 0 causes data corruption | ✓ | | | CRITICAL |
| Missing bounds checks (buffer access) | ✓ | | | CRITICAL |
| Matrix intrinsic discards result | ✓ | | | HIGH |
| Unbounded queue consumption | ✓ | | | HIGH |
| simd_barrier is NOP without flags | ✓ | | | MEDIUM |

**Consensus Fix**: Use `simd_is_first()` instead of `lane == 0`, add bounds checks, fix matrix wrapper.

---

### Issue #10: Tile-Based Renderer

| Bug | Gemini | Grok | Codex | Severity |
|-----|--------|------|-------|----------|
| Missing in.bounds in fragment shader | | ✓ | | CRITICAL |
| SDF assumes square widgets | | ✓ | | HIGH |
| UV reused for progress value | | ✓ | | HIGH |
| Branchless dispatch needs when_eq | | ✓ | | MEDIUM |
| Indirect draw buffer bounds unchecked | | ✓ | ✓ | CRITICAL |
| Partial review (rate limited) | ✓ | | | - |

**Consensus Fix**: Add bounds to VertexOut, fix aspect ratio in SDF, separate progress value from UV.

---

## Top 10 Most Critical Bugs (Cross-Issue)

1. **threadgroup_barrier only syncs within threadgroup** - Affects #1, #2, #5, #7
2. **GPU/CPU virtual address mismatch** - Affects #1, #3 (use offsets, not pointers)
3. **Ring buffer overflow (no full check)** - Affects #1, #5
4. **Request/response race (no ready flag)** - Affects #1
5. **Multi-threadgroup race on shared state** - Affects #2, #5
6. **Onesweep spin-lock deadlock** - Affects #7, #8
7. **O(N) loops causing GPU timeout** - Affects #2, #4, #7
8. **Framebuffer size allocation (32MB vs 66MB)** - Affects #3
9. **ABA problem in lock-free allocator** - Affects #3
10. **Float-to-uint sorting breaks for negatives** - Affects #2, #7

---

## Recommended Priority

### P0 (Must Fix Before Prototype)
- Fix threadgroup vs device-scope synchronization
- Fix GPU/CPU pointer handling (use offsets)
- Add ring buffer full/empty checks
- Fix framebuffer allocation size

### P1 (Fix Before Alpha)
- Fix sort key construction
- Implement proper ready flag for IPC
- Add bounds checking everywhere
- Fix ABA problem in allocator

### P2 (Fix Before Beta)
- Handle negative float depth sorting
- Optimize O(N) loops
- Fix struct alignment mismatches
- Improve error handling

---

*Generated from Gemini, Grok-4, and Codex reviews*
