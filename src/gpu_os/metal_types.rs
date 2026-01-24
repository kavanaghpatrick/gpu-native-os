//! Metal-Safe Type Definitions
//!
//! This module provides Rust struct definitions that are guaranteed to match
//! Metal's memory layout. Metal has specific alignment requirements:
//!
//! | Metal Type | Size    | Alignment |
//! |------------|---------|-----------|
//! | float      | 4 bytes | 4 bytes   |
//! | float2     | 8 bytes | 8 bytes   |
//! | float3     | 12 bytes| 16 bytes! |
//! | float4     | 16 bytes| 16 bytes  |
//! | uint       | 4 bytes | 4 bytes   |
//! | uint2      | 8 bytes | 8 bytes   |
//! | uint3      | 12 bytes| 16 bytes! |
//! | uint4      | 16 bytes| 16 bytes  |
//!
//! CRITICAL: float3/uint3 have 16-byte alignment despite being 12 bytes!
//!
//! When a struct field has alignment N, it must start at offset divisible by N.
//! This often requires explicit padding in Rust to match Metal's implicit padding.
//!
//! # Example
//! ```ignore
//! // WRONG - Rust: 24 bytes, Metal: 32 bytes (color needs 16-byte alignment)
//! struct BadVertex {
//!     position: [f32; 2],  // 8 bytes at offset 0
//!     color: [f32; 4],     // 16 bytes at offset 8 (Rust) vs offset 16 (Metal!)
//! }
//!
//! // CORRECT - Both Rust and Metal: 32 bytes
//! struct GoodVertex {
//!     position: [f32; 2],  // 8 bytes at offset 0
//!     _pad: [f32; 2],      // 8 bytes padding
//!     color: [f32; 4],     // 16 bytes at offset 16
//! }
//! ```

use std::mem;

// ============================================================================
// Compile-Time Size Assertions
// ============================================================================

/// Assert struct size at compile time
macro_rules! assert_metal_size {
    ($t:ty, $expected:expr) => {
        const _: () = {
            let actual = mem::size_of::<$t>();
            let expected = $expected;
            if actual != expected {
                panic!("Metal struct size mismatch");
            }
        };
    };
}

/// Assert struct alignment at compile time
macro_rules! assert_metal_align {
    ($t:ty, $expected:expr) => {
        const _: () = {
            let actual = mem::align_of::<$t>();
            let expected = $expected;
            if actual != expected {
                panic!("Metal struct alignment mismatch");
            }
        };
    };
}

// ============================================================================
// Common Vertex Types (Metal-Safe)
// ============================================================================

/// Basic vertex with position and color (32 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct Vertex {
///     float2 position;
///     float2 _pad;
///     float4 color;
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalVertex {
    pub position: [f32; 2],  // 8 bytes at offset 0
    pub _pad: [f32; 2],      // 8 bytes at offset 8 (padding for color alignment)
    pub color: [f32; 4],     // 16 bytes at offset 16
}
assert_metal_size!(MetalVertex, 32);
assert_metal_align!(MetalVertex, 4);

/// Textured vertex with position, UV, and color (32 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct TexturedVertex {
///     float2 position;
///     float2 tex_coord;
///     float4 color;
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalTexturedVertex {
    pub position: [f32; 2],   // 8 bytes at offset 0
    pub tex_coord: [f32; 2],  // 8 bytes at offset 8
    pub color: [f32; 4],      // 16 bytes at offset 16
}
assert_metal_size!(MetalTexturedVertex, 32);
assert_metal_align!(MetalTexturedVertex, 4);

/// Full vertex with position, UV, color, and flags (48 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct FullVertex {
///     float2 position;
///     float2 tex_coord;
///     float4 color;
///     uint flags;
///     uint _pad[3];
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalFullVertex {
    pub position: [f32; 2],   // 8 bytes at offset 0
    pub tex_coord: [f32; 2],  // 8 bytes at offset 8
    pub color: [f32; 4],      // 16 bytes at offset 16
    pub flags: u32,           // 4 bytes at offset 32
    pub _pad: [u32; 3],       // 12 bytes at offset 36
}
assert_metal_size!(MetalFullVertex, 48);
assert_metal_align!(MetalFullVertex, 4);

// ============================================================================
// Uniform Types (Metal-Safe)
// ============================================================================

/// Simple uniforms with scroll offset (16 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct Uniforms {
///     float scroll_y;
///     float _pad[3];
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalScrollUniforms {
    pub scroll_y: f32,       // 4 bytes at offset 0
    pub _pad: [f32; 3],      // 12 bytes padding (total 16 for alignment)
}
assert_metal_size!(MetalScrollUniforms, 16);

/// Viewport uniforms (16 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct Viewport {
///     float width;
///     float height;
///     float _pad[2];
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalViewport {
    pub width: f32,
    pub height: f32,
    pub _pad: [f32; 2],
}
assert_metal_size!(MetalViewport, 16);

/// Transform uniforms with MVP matrix (64 bytes)
///
/// Metal equivalent:
/// ```metal
/// struct TransformUniforms {
///     float4x4 mvp;  // 64 bytes, 16-byte aligned
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MetalTransformUniforms {
    pub mvp: [[f32; 4]; 4],  // 64 bytes at offset 0
}
assert_metal_size!(MetalTransformUniforms, 64);
assert_metal_align!(MetalTransformUniforms, 4);

impl Default for MetalTransformUniforms {
    fn default() -> Self {
        Self {
            mvp: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Verify a type's size matches expected Metal size at runtime
/// Useful for debugging alignment issues
pub fn verify_metal_size<T>(expected: usize) -> Result<(), String> {
    let actual = mem::size_of::<T>();
    if actual != expected {
        Err(format!(
            "Metal size mismatch for {}: Rust={} bytes, expected={} bytes",
            std::any::type_name::<T>(),
            actual,
            expected
        ))
    } else {
        Ok(())
    }
}

/// Calculate required padding to align offset to boundary
pub const fn padding_for_alignment(offset: usize, alignment: usize) -> usize {
    let remainder = offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

// ============================================================================
// Metal Shader Code Generator (for documentation)
// ============================================================================

/// Returns Metal shader code for the standard vertex types
/// Use this as a reference when writing Metal shaders
pub fn metal_shader_types() -> &'static str {
    r#"
// Metal-Safe Vertex Types
// These MUST match the Rust definitions in metal_types.rs

struct MetalVertex {
    float2 position;  // 8 bytes at offset 0
    float2 _pad;      // 8 bytes at offset 8
    float4 color;     // 16 bytes at offset 16
};  // Total: 32 bytes

struct MetalTexturedVertex {
    float2 position;   // 8 bytes at offset 0
    float2 tex_coord;  // 8 bytes at offset 8
    float4 color;      // 16 bytes at offset 16
};  // Total: 32 bytes

struct MetalFullVertex {
    float2 position;   // 8 bytes at offset 0
    float2 tex_coord;  // 8 bytes at offset 8
    float4 color;      // 16 bytes at offset 16
    uint flags;        // 4 bytes at offset 32
    uint _pad[3];      // 12 bytes at offset 36
};  // Total: 48 bytes

struct MetalScrollUniforms {
    float scroll_y;
    float _pad[3];
};  // Total: 16 bytes

struct MetalViewport {
    float width;
    float height;
    float _pad[2];
};  // Total: 16 bytes

struct MetalTransformUniforms {
    float4x4 mvp;
};  // Total: 64 bytes
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_sizes() {
        assert_eq!(mem::size_of::<MetalVertex>(), 32);
        assert_eq!(mem::size_of::<MetalTexturedVertex>(), 32);
        assert_eq!(mem::size_of::<MetalFullVertex>(), 48);
    }

    #[test]
    fn test_uniform_sizes() {
        assert_eq!(mem::size_of::<MetalScrollUniforms>(), 16);
        assert_eq!(mem::size_of::<MetalViewport>(), 16);
        assert_eq!(mem::size_of::<MetalTransformUniforms>(), 64);
    }

    #[test]
    fn test_padding_calculation() {
        // offset 8, need 16-byte alignment -> pad 8
        assert_eq!(padding_for_alignment(8, 16), 8);
        // offset 16, need 16-byte alignment -> pad 0
        assert_eq!(padding_for_alignment(16, 16), 0);
        // offset 4, need 4-byte alignment -> pad 0
        assert_eq!(padding_for_alignment(4, 4), 0);
        // offset 5, need 4-byte alignment -> pad 3
        assert_eq!(padding_for_alignment(5, 4), 3);
    }

    #[test]
    fn test_verify_metal_size() {
        assert!(verify_metal_size::<MetalVertex>(32).is_ok());
        assert!(verify_metal_size::<MetalVertex>(24).is_err());
    }
}
