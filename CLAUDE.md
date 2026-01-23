# CLAUDE.md - Rust Development Guide

## Project Context
**Type**: Metal GPU graphics application (macOS)
**Crate**: metal-rs, winit, cocoa

---

## Rust Development Frameworks

### 1. Memory & Alignment
```rust
// GPU buffers require specific alignment - ALWAYS check struct layout
#[repr(C)]  // Use C layout for FFI/GPU interop
struct GpuStruct {
    position: [f32; 2],
    _padding: [f32; 2],  // Metal float4 needs 16-byte alignment
    color: [f32; 4],
}
```

**Rules**:
- `#[repr(C)]` for any struct crossing FFI boundaries
- GPU APIs often require 16-byte alignment for vec4/float4
- Add explicit padding rather than relying on compiler
- Use `std::mem::size_of` to verify struct sizes

### 2. Error Handling Patterns
```rust
// Prefer explicit unwrap with context in prototypes
let device = Device::system_default().expect("No Metal device found");

// For production, use Result chains
fn init() -> Result<Self, Box<dyn Error>> {
    let device = Device::system_default().ok_or("No Metal device")?;
    Ok(Self { device })
}
```

### 3. Matrix Math Conventions

**Critical for graphics**:
```rust
// Row-major (Rust) vs Column-major (Metal/GPU)
// When sending matrices to GPU, transpose if needed
let uniforms = Uniforms {
    matrix: mat4_transpose(cpu_matrix),
};

// Perspective projection - Metal uses Z range [0,1], not OpenGL's [-1,1]
fn mat4_perspective_metal(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4]

// Triangle winding - Counter-clockwise for front-facing
// Quad vertices: TL(0), TR(1), BR(2), BL(3)
// Triangle 1: 0 -> 3 -> 2 (TL -> BL -> BR)
// Triangle 2: 0 -> 2 -> 1 (TL -> BR -> TR)
```

### 4. Iterative Debugging Strategy

When something doesn't render:
1. **Simplify first** - Use identity matrices, orthographic projection
2. **Isolate components** - Test geometry without view transform
3. **Add visual debugging** - Bright clear colors to distinguish "not rendering" from "rendering black"
4. **Check clip space** - Verify vertices are in [-1,1] range after transforms

```rust
// Debug: Use bright purple background to see if geometry renders
color_attachment.set_clear_color(MTLClearColor::new(0.5, 0.0, 0.5, 1.0));

// Debug: Skip view matrix to test projection alone
let view_matrix = mat4_identity();
```

### 5. GPU Resource Management
```rust
// Create buffers with appropriate storage mode
let buffer = device.new_buffer_with_data(
    data.as_ptr() as *const _,
    size as u64,
    MTLResourceOptions::StorageModeShared,  // CPU+GPU access
);

// For GPU-only data, use Private storage
MTLResourceOptions::StorageModePrivate
```

### 6. Frame Timing & Stats
```rust
// Track frame times with a rolling window
frame_times: VecDeque<f64>,  // Capacity ~120 frames

// Calculate stats
let avg = frame_times.iter().sum::<f64>() / frame_times.len() as f64;
let fps = 1000.0 / avg;  // avg is in milliseconds
```

---

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Struct alignment | Corrupted GPU data, visual artifacts | Add `_padding` fields, use `#[repr(C)]` |
| Wrong winding order | Missing triangles, backface culled | Use CCW: 0->3->2, 0->2->1 for quads |
| Matrix convention | Nothing renders, or renders wrong | Transpose for Metal, use Metal Z range |
| Clip space range | Geometry clipped | Ensure final positions in [-1,1] x/y, [0,1] z |

---

## Build Commands
```bash
cargo build --release          # Optimized build
cargo run --release            # Run optimized
cargo check                    # Fast syntax/type check
cargo clippy                   # Lints and suggestions
```

---

## Metal-Specific Notes

### Shader Compilation
```rust
let library = device
    .new_library_with_source(SHADER_SOURCE, &compile_options)
    .expect("Failed to compile shaders");
```

### Render Pipeline Setup
1. Create vertex/fragment functions from library
2. Configure pipeline descriptor with pixel formats
3. Set depth attachment format if using depth testing
4. Create pipeline state (expensive - do once at init)

### Render Pass
1. Get next drawable from layer
2. Create render pass descriptor
3. Set color/depth attachments
4. Create encoder, set pipeline, bind buffers
5. Draw primitives
6. End encoding, present, commit

---

## Text Rendering (7-Segment Style)

For simple GPU text without font libraries:
```rust
// Define segment patterns for characters
const DIGIT_SEGMENTS: [[bool; 7]; 10] = [/*...*/];

// Generate quads for active segments
// Each segment = 2 triangles = 6 vertices
```

---

## Performance Tips

1. **Batch draw calls** - Combine geometry into single buffer
2. **Avoid per-frame allocations** - Reuse buffers
3. **Profile with Instruments** - Metal System Trace
4. **Use release builds** - Debug builds are 10-100x slower
