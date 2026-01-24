//! Metal Struct Alignment Validation Tests
//!
//! This test file validates that all GPU vertex structs in the codebase
//! have correct alignment for Metal. Run with:
//!   cargo test --test test_metal_alignment
//!
//! CRITICAL RULE: Any struct containing float4/[f32; 4] that follows
//! a field smaller than 16 bytes MUST have explicit padding.

use std::mem;

// ============================================================================
// Alignment Rules Reference
// ============================================================================
//
// Metal Type  | Size     | Alignment | Rust Equivalent
// ------------|----------|-----------|----------------
// float       | 4 bytes  | 4 bytes   | f32
// float2      | 8 bytes  | 8 bytes   | [f32; 2]
// float3      | 12 bytes | 16 bytes! | [f32; 3] + padding
// float4      | 16 bytes | 16 bytes  | [f32; 4]
// uint        | 4 bytes  | 4 bytes   | u32
// uint2       | 8 bytes  | 8 bytes   | [u32; 2]
// uint3       | 12 bytes | 16 bytes! | [u32; 3] + padding
// uint4       | 16 bytes | 16 bytes  | [u32; 4]
//
// DANGER PATTERN: position[2] followed by color[4] = WRONG!
//   Rust sees:  8 bytes + 16 bytes = 24 bytes
//   Metal sees: 8 bytes + 8 pad + 16 bytes = 32 bytes
//
// ============================================================================

/// Helper to check struct size at compile time
macro_rules! assert_size {
    ($t:ty, $expected:expr, $name:expr) => {
        const _: () = assert!(
            mem::size_of::<$t>() == $expected,
            // Note: concat! doesn't work in const context error messages
        );
    };
}

/// Check that a struct's size is a multiple of 16 (good for Metal)
macro_rules! assert_16_aligned_size {
    ($t:ty) => {
        const _: () = assert!(
            mem::size_of::<$t>() % 16 == 0,
        );
    };
}

// ============================================================================
// Import all vertex structs from the codebase
// ============================================================================

use rust_experiment::gpu_os::metal_types::{
    MetalVertex, MetalTexturedVertex, MetalFullVertex,
    MetalScrollUniforms, MetalViewport, MetalTransformUniforms,
};

use rust_experiment::gpu_os::document::{
    PaintVertex, Viewport as DocViewport, ComputedStyle, Element, LayoutBox,
};

use rust_experiment::gpu_os::text_render::TextChar;

// ============================================================================
// Size Validation Tests
// ============================================================================

#[test]
fn test_metal_types_sizes() {
    // These are the canonical Metal-safe types
    assert_eq!(mem::size_of::<MetalVertex>(), 32, "MetalVertex should be 32 bytes");
    assert_eq!(mem::size_of::<MetalTexturedVertex>(), 32, "MetalTexturedVertex should be 32 bytes");
    assert_eq!(mem::size_of::<MetalFullVertex>(), 48, "MetalFullVertex should be 48 bytes");
    assert_eq!(mem::size_of::<MetalScrollUniforms>(), 16, "MetalScrollUniforms should be 16 bytes");
    assert_eq!(mem::size_of::<MetalViewport>(), 16, "MetalViewport should be 16 bytes");
    assert_eq!(mem::size_of::<MetalTransformUniforms>(), 64, "MetalTransformUniforms should be 64 bytes");
}

#[test]
fn test_document_struct_sizes() {
    // PaintVertex: position[2] + tex_coord[2] + color[4] + flags + padding[3]
    // = 8 + 8 + 16 + 4 + 12 = 48 bytes
    assert_eq!(mem::size_of::<PaintVertex>(), 48, "PaintVertex should be 48 bytes");

    // Viewport: width + height + padding[2] = 16 bytes
    assert_eq!(mem::size_of::<DocViewport>(), 16, "Viewport should be 16 bytes");

    // Element should be 32 bytes (GPU-friendly)
    assert_eq!(mem::size_of::<Element>(), 32, "Element should be 32 bytes");
}

#[test]
fn test_text_char_size() {
    // TextChar: x, y, char_code, color, scale = 4+4+4+4+4 = 20 bytes
    // But should be padded to 24 or 32 for GPU
    let size = mem::size_of::<TextChar>();
    println!("TextChar size: {} bytes", size);
    // TextChar is used in instanced rendering, size should be reasonable
    assert!(size <= 32, "TextChar should be <= 32 bytes, got {}", size);
}

#[test]
fn test_sizes_are_multiples_of_4() {
    // All GPU structs should have sizes that are multiples of 4 (basic alignment)
    assert_eq!(mem::size_of::<PaintVertex>() % 4, 0, "PaintVertex not 4-byte aligned");
    assert_eq!(mem::size_of::<DocViewport>() % 4, 0, "Viewport not 4-byte aligned");
    assert_eq!(mem::size_of::<ComputedStyle>() % 4, 0, "ComputedStyle not 4-byte aligned");
    assert_eq!(mem::size_of::<Element>() % 4, 0, "Element not 4-byte aligned");
    assert_eq!(mem::size_of::<LayoutBox>() % 4, 0, "LayoutBox not 4-byte aligned");
}

// ============================================================================
// Dangerous Pattern Detection
// ============================================================================

/// This test documents the DANGEROUS pattern that caused the garbage triangles bug.
/// The pattern is: [f32; 2] followed directly by [f32; 4] without padding.
///
/// Rust layout:  offset 0: position (8 bytes), offset 8: color (16 bytes) = 24 bytes
/// Metal layout: offset 0: position (8 bytes), offset 16: color (16 bytes) = 32 bytes
///
/// This causes Metal to read garbage data as color values!
#[test]
fn document_dangerous_pattern() {
    // This struct demonstrates the WRONG pattern (DO NOT USE!)
    #[repr(C)]
    struct DangerousVertex {
        position: [f32; 2],  // 8 bytes
        color: [f32; 4],     // 16 bytes - BUT Metal expects this at offset 16!
    }

    // Rust sees 24 bytes
    assert_eq!(mem::size_of::<DangerousVertex>(), 24);

    // But Metal expects 32 bytes! This mismatch causes garbage rendering.
    // The fix is to add padding:
    #[repr(C)]
    struct SafeVertex {
        position: [f32; 2],  // 8 bytes at offset 0
        _pad: [f32; 2],      // 8 bytes at offset 8
        color: [f32; 4],     // 16 bytes at offset 16
    }

    // Now Rust and Metal agree: 32 bytes
    assert_eq!(mem::size_of::<SafeVertex>(), 32);
}

// ============================================================================
// Field Offset Validation
// ============================================================================

/// Validate that color fields are at 16-byte aligned offsets
#[test]
fn test_color_field_alignment() {
    use std::ptr;

    // MetalVertex: color should be at offset 16
    let v = MetalVertex::default();
    let base = ptr::addr_of!(v) as usize;
    let color_offset = ptr::addr_of!(v.color) as usize - base;
    assert_eq!(color_offset, 16, "MetalVertex.color should be at offset 16, got {}", color_offset);

    // MetalTexturedVertex: color should be at offset 16
    let v = MetalTexturedVertex::default();
    let base = ptr::addr_of!(v) as usize;
    let color_offset = ptr::addr_of!(v.color) as usize - base;
    assert_eq!(color_offset, 16, "MetalTexturedVertex.color should be at offset 16, got {}", color_offset);

    // MetalFullVertex: color should be at offset 16
    let v = MetalFullVertex::default();
    let base = ptr::addr_of!(v) as usize;
    let color_offset = ptr::addr_of!(v.color) as usize - base;
    assert_eq!(color_offset, 16, "MetalFullVertex.color should be at offset 16, got {}", color_offset);
}

/// Validate PaintVertex field offsets match Metal shader expectations
#[test]
fn test_paint_vertex_offsets() {
    use std::ptr;

    let v = PaintVertex::default();
    let base = ptr::addr_of!(v) as usize;

    let pos_offset = ptr::addr_of!(v.position) as usize - base;
    let tex_offset = ptr::addr_of!(v.tex_coord) as usize - base;
    let color_offset = ptr::addr_of!(v.color) as usize - base;
    let flags_offset = ptr::addr_of!(v.flags) as usize - base;

    assert_eq!(pos_offset, 0, "position should be at offset 0");
    assert_eq!(tex_offset, 8, "tex_coord should be at offset 8");
    assert_eq!(color_offset, 16, "color should be at offset 16");
    assert_eq!(flags_offset, 32, "flags should be at offset 32");
}

// ============================================================================
// Comprehensive Struct Audit
// ============================================================================

#[test]
fn audit_all_gpu_structs() {
    println!("\n=== GPU Struct Size Audit ===\n");

    macro_rules! audit {
        ($t:ty) => {
            let size = mem::size_of::<$t>();
            let align = mem::align_of::<$t>();
            let ok = size % 4 == 0;
            let status = if ok { "OK" } else { "WARN" };
            println!("{:40} {:>4} bytes  align={:>2}  [{}]",
                stringify!($t), size, align, status);
        };
    }

    println!("Metal Types (canonical):");
    audit!(MetalVertex);
    audit!(MetalTexturedVertex);
    audit!(MetalFullVertex);
    audit!(MetalScrollUniforms);
    audit!(MetalViewport);
    audit!(MetalTransformUniforms);

    println!("\nDocument Pipeline:");
    audit!(PaintVertex);
    audit!(DocViewport);
    audit!(ComputedStyle);
    audit!(Element);
    audit!(LayoutBox);

    println!("\nText Rendering:");
    audit!(TextChar);

    println!("\n=== Audit Complete ===\n");
}
