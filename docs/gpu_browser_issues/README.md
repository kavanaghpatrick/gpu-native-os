# GPU Browser Implementation Issues

## Overview

This directory contains detailed PRDs, pseudocode, benchmarks, and tests for implementing a GPU-native browser engine based on our analysis of Servo.

## Issue Dependency Graph

```
                    ┌─────────────────────┐
                    │ #112: HTML5 Parser  │
                    │ (html5ever)         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ #114: CSS Parser    │
                    │ (cssparser)         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ #115: GPU Selector  │
                    │ Matching            │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ #116: GPU Cascade   │
                    │ Resolution          │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    │
┌──────────────────┐ ┌──────────────────┐          │
│ #117: GPU Layout │ │ #119: Parallel   │          │
│ Engine           │ │ Prefix Sum       │          │
└────────┬─────────┘ └──────────────────┘          │
         │                                          │
         ▼                                          │
┌──────────────────┐                               │
│ #118: GPU Text   │                               │
│ Measurement      │                               │
└────────┬─────────┘                               │
         │                                          │
         └────────────────┬────────────────────────┘
                          │
               ┌──────────▼──────────┐
               │ #120: Benchmarking  │
               │ Framework           │
               └─────────────────────┘
```

## Issues Summary

| Issue | Title | Speedup Target | Status |
|-------|-------|----------------|--------|
| [#112](https://github.com/kavanaghpatrick/gpu-native-os/issues/112) | HTML5 Parser (html5ever) | - | 🔴 TODO |
| [#114](https://github.com/kavanaghpatrick/gpu-native-os/issues/114) | CSS Parser (cssparser) | - | 🔴 TODO |
| [#115](https://github.com/kavanaghpatrick/gpu-native-os/issues/115) | GPU Selector Matching | 50-250x | 🔴 TODO |
| [#116](https://github.com/kavanaghpatrick/gpu-native-os/issues/116) | GPU Cascade Resolution | 10-50x | 🔴 TODO |
| [#117](https://github.com/kavanaghpatrick/gpu-native-os/issues/117) | GPU Layout Engine | 5-20x | 🔴 TODO |
| [#118](https://github.com/kavanaghpatrick/gpu-native-os/issues/118) | GPU Text Measurement | 25-100x | 🔴 TODO |
| [#119](https://github.com/kavanaghpatrick/gpu-native-os/issues/119) | Parallel Prefix Sum | 5-50x | 🔴 TODO |
| [#120](https://github.com/kavanaghpatrick/gpu-native-os/issues/120) | Benchmarking Framework | - | 🔴 TODO |

## PRD Files

| Issue | PRD File |
|-------|----------|
| #112 | [ISSUE_102_HTML5_PARSER.md](./ISSUE_102_HTML5_PARSER.md) |
| #114 | [ISSUE_103_CSS_PARSER.md](./ISSUE_103_CSS_PARSER.md) |
| #115 | [ISSUE_104_GPU_SELECTOR_MATCHING.md](./ISSUE_104_GPU_SELECTOR_MATCHING.md) |
| #116 | [ISSUE_105_GPU_CASCADE.md](./ISSUE_105_GPU_CASCADE.md) |
| #117 | [ISSUE_106_GPU_LAYOUT.md](./ISSUE_106_GPU_LAYOUT.md) |
| #118 | [ISSUE_107_GPU_TEXT.md](./ISSUE_107_GPU_TEXT.md) |
| #119 | [ISSUE_108_PARALLEL_PREFIX_SUM.md](./ISSUE_108_PARALLEL_PREFIX_SUM.md) |
| #120 | [ISSUE_109_BENCHMARKING.md](./ISSUE_109_BENCHMARKING.md) |

## Expected Performance Targets (vs Servo)

Benchmarking against **Servo** (Rayon-parallelized) - the real baseline:

### Full Pipeline (Wikipedia-class page, ~5000 elements)

| Stage | Naive CPU | Servo (Rayon) | GPU Target | vs Servo |
|-------|-----------|---------------|------------|----------|
| HTML Parsing | 5ms | 5ms | 5ms | 1x (CPU-bound) |
| CSS Parsing | 20ms | 20ms | 20ms | 1x (CPU-bound) |
| Selector Matching | 500ms | 50ms | 2ms | **25x** |
| Cascade Resolution | 50ms | 10ms | 1ms | **10x** |
| Layout | 100ms | 20ms | 3ms | **7x** |
| Text Shaping | 10ms | 3ms | 0.1ms | **30x** |
| Paint Generation | 10ms | 5ms | 0.5ms | **10x** |
| **Total** | **695ms** | **~113ms** | **~32ms** | **~3.5x** |

**Note**: Servo already uses Rayon parallelization, so we're competing against optimized parallel CPU code. Speedups are smaller than vs naive CPU but still significant.

### Frame Budget Analysis

At 60fps, we have 16.67ms per frame:

- **Current (CPU)**: 640ms → ~1.5 FPS
- **GPU Target**: 32ms → ~30 FPS (initial layout)
- **Incremental**: <5ms → 60 FPS (scrolling, hover)

## Implementation Order

### Week 1-2: Foundation
1. #112: HTML5 Parser Integration
2. #114: CSS Parser Integration
3. #119: Parallel Prefix Sum (unblocks other GPU work)

### Week 3-4: GPU Pipeline
4. #115: GPU Selector Matching
5. #116: GPU Cascade Resolution

### Week 5-8: Layout & Text
6. #117: GPU Layout Engine
7. #118: GPU Text Measurement

### Week 9-10: Validation
8. #120: Benchmarking Framework
9. Integration testing

## Servo Benchmark Integration

### Running Servo Benchmarks

```bash
# Build Servo with tracing
cd /tmp/servo-src
./mach build --release --features=tracing

# Run with benchmark mode
./mach run --release -- --benchmark file://test_pages/wikipedia.html

# Or via our harness
cargo bench --features=servo-baseline
```

### Metrics We'll Compare

| Metric | Servo Source | Our Equivalent |
|--------|--------------|----------------|
| Style recalc | `servo::style::recalc_style` | `cascade.resolve()` |
| Layout | `servo::layout::perform_layout` | `layout_engine.layout()` |
| Paint | `servo::paint::build_display_list` | `paint.generate()` |
| First frame | DevTools `Page.loadEventFired` | End-to-end timing |

## Key Technical Insights from Servo Analysis

1. **~90% of layout work is embarrassingly parallel** - perfect for GPU
2. **Conditional `use_rayon` pattern** - easy to add `use_gpu` flag
3. **Prefix sum is a hidden bottleneck** - both in our code and Servo/WebRender
4. **Multi-pass by depth level** handles parent→child dependencies
5. **SDF text atlas** (already built!) solves GPU text measurement
6. **Servo baseline is ~3-5x faster than naive CPU** - we need to beat Servo, not naive
