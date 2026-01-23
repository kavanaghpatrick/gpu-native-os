# CLAUDE.md - GPU-Native OS Project

## Project Overview
Single-threadgroup OS running entirely on GPU via Metal compute shaders. All 1024 threads participate in every phase.

## Architecture
```
src/lib.rs              # Library entry
src/gpu_os/
  mod.rs                # Module exports
  kernel.rs             # #11 - Unified Worker Model (core compute kernel)
  memory.rs             # #12 - Memory Architecture (GPU buffers)
  input.rs              # #13 - Input Pipeline (HID to GPU)
  layout.rs             # #14 - Layout Engine (constraint solving)
  widget.rs             # #15 - Widget System (compressed state)
  text.rs               # #16 - Text Rendering (7-segment + font atlas)
  render.rs             # #17 - Hybrid Rendering (compute + fragment)
  vsync.rs              # #18 - VSync Execution (frame timing)
  shaders/kernel.metal  # GPU kernel source
examples/gpu_os_demo.rs # Visual demo
tests/test_issue_*.rs   # Per-issue integration tests
docs/                   # PRDs and design docs
archive/                # Old code (spinning ball demo)
```

## Commands
```bash
cargo check                        # Fast type check
cargo build --release              # Optimized build
cargo run --example gpu_os_demo    # Run visual demo
cargo test                         # Run all tests
cargo test test_issue_11           # Run specific issue tests
```

## Key Patterns

**GPU Structs**: Always use `#[repr(C)]` with explicit padding for Metal alignment
```rust
#[repr(C)]
struct GpuStruct {
    data: [f32; 2],
    _padding: [f32; 2],  // Metal float4 = 16-byte alignment
}
```

**Issue Mapping**: Each module maps to a GitHub issue (#11-#18)

**Testing**: Each issue has dedicated integration tests in `tests/`

## Debug Tips
- Purple background = geometry renders but wrong color
- Nothing visible = check clip space [-1,1] range
- Corrupted data = check struct alignment/padding
