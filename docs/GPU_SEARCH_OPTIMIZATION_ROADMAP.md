# GPU Search Optimization Roadmap

## Current Performance (UPDATED)

| Test | Throughput | Notes |
|------|------------|-------|
| Raw memory bandwidth | 75.5 GB/s | M4 Pro ceiling |
| **Production GPU search** | **55-80 GB/s** | **4-7x faster than ripgrep!** |
| ripgrep (baseline) | 10-13 GB/s | Includes file I/O |

### Benchmark Results (114 MB test)
| Pattern | GPU | ripgrep | Speedup |
|---------|-----|---------|---------|
| Rare (17 char) | 79.9 GB/s | 10.5 GB/s | **7x** |
| Common (3 char) | 62.4 GB/s | 12.5 GB/s | **5x** |
| Very common (4 char) | 58.2 GB/s | 12.8 GB/s | **4-5x** |

## Key Techniques Discovered

### 1. Vectorized Loads (uchar4)
```metal
// Instead of byte-by-byte:
char c = data[i];  // BAD: non-coalesced

// Use vector loads:
uchar4 v = data[i / 4];  // GOOD: 4 bytes at once, coalesced
```

### 2. SIMD Prefix Sum for Match Offsets
```metal
// Instead of global atomic per match:
uint global_idx = atomic_fetch_add(&count, 1);  // BAD: contention

// Use SIMD reduction:
uint local_count = /* my matches */;
uint simd_total = simd_sum(local_count);
uint my_offset = simd_prefix_exclusive_sum(local_count);
if (simd_lane == 0) {
    group_base = atomic_fetch_add(&count, simd_total);  // ONE atomic per 32 threads
}
```

### 3. Incremental Line Tracking
```metal
// Instead of counting from start per match:
uint line = count_lines(data, pos);  // BAD: O(pos) per match

// Track incrementally:
while (scanned_to < pos) {
    if (data[scanned_to] == '\n') current_line++;
    scanned_to++;
}
// O(1) per match
```

### 4. Teddy/Nybble Trick (from Hyperscan)
For multi-pattern search, use nybble-based lookups:
```metal
// Split byte into two 4-bit nybbles
uchar lo = byte & 0x0F;
uchar hi = byte >> 4;

// Lookup in 16-element tables (fits in SIMD register!)
uchar mask_lo = simd_shuffle(table_lo, lo);
uchar mask_hi = simd_shuffle(table_hi, hi);

// Only exact matches pass through
uchar result = mask_lo & mask_hi;
```

## Optimization Tiers

### Tier 1: COMPLETED (55-80 GB/s achieved!)
- [x] Vectorized loads (uchar4)
- [x] SIMD prefix sum
- [x] Pre-allocated slots per thread
- [x] Early exit on mismatch
- [x] Single atomic per SIMD group (32 threads)

### Tier 2: COMPLETED (Integrated into GpuContentSearch)
- [x] Integrated vectorized kernel into production
- [x] 64 bytes per thread (16 x uchar4)
- [x] Line numbers calculated on CPU post-search
- [x] 100MB+ data processing working

### Tier 3: Advanced Techniques (Target: 60+ GB/s)
- [ ] Streaming I/O (MTLIOCommandQueue while GPU computes)
- [ ] Multi-pattern search (Teddy nybble trick)
- [ ] Persistent kernels (eliminate dispatch overhead)
- [ ] Shared memory prefetching

### Tier 4: Theoretical Maximum (Target: approach raw bandwidth)
- [ ] GPU-side decompression (LZ4 decode on GPU for 4x effective bandwidth)
- [ ] FM-index for compressed text search
- [ ] PFAC (Parallel Failureless Aho-Corasick) for regex

## Hardware Limits

| Hardware | Memory Bandwidth |
|----------|------------------|
| M4 Pro | ~75 GB/s |
| M4 Max | ~150 GB/s |
| M4 Ultra | ~300 GB/s |

With compression (4:1 ratio), effective bandwidth could reach 300-1200 GB/s.

## Research Sources

- Teddy algorithm: https://github.com/BurntSushi/aho-corasick/blob/master/src/packed/teddy/README.md
- Hyperscan: https://branchfree.org/2024/06/29/hyperscan-papers-harry-and-teddy-simd-literal-matchers/
- CUDA Grep: CMU 15-418 parallel computing course
- HybridSA: https://dl.acm.org/doi/10.1145/3689771
- PFAC: GPU Aho-Corasick implementations
