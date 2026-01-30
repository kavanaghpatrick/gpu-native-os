//! GPU Standard Library for WASM Apps
//!
//! THE GPU IS THE COMPUTER.
//!
//! This crate provides a clean Rust API for GPU-native applications.
//! Apps using this crate compile to WASM and run entirely on the GPU.
//!
//! # Example
//!
//! ```no_run
//! #![no_std]
//! #![no_main]
//!
//! use gpu_std::prelude::*;
//!
//! #[no_mangle]
//! pub extern "C" fn main() -> i32 {
//!     let f = frame();
//!
//!     // Draw a red square
//!     emit_quad(100.0, 100.0, 50.0, 50.0, Color::RED);
//!
//!     // Draw animated circle
//!     let x = 400.0 + sin(f as f32 * 0.1) * 100.0;
//!     emit_quad(x, 300.0, 20.0, 20.0, Color::BLUE);
//!
//!     2 // Return number of quads drawn
//! }
//!
//! #[panic_handler]
//! fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
//! ```

// Only use no_std when not using std feature
#![cfg_attr(not(feature = "std"), no_std)]

// ============================================================================
// COLLECTIONS - GPU-native data structures
// ============================================================================

#[cfg(feature = "alloc")]
pub mod collections;

// ============================================================================
// ALLOCATOR - Enables Vec, String, Box
// Only used for WASM targets (not when using std feature for testing)
// ============================================================================

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use core::alloc::{GlobalAlloc, Layout};

/// GPU heap allocator - routes to GPU slab allocator via WASM intrinsics
/// Only used for WASM targets, not for host testing
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub struct GpuAllocator;

#[cfg(all(feature = "alloc", not(feature = "std")))]
unsafe impl GlobalAlloc for GpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        extern "C" {
            fn __rust_alloc(size: usize, align: usize) -> *mut u8;
        }
        __rust_alloc(layout.size(), layout.align())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        extern "C" {
            fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
        }
        __rust_dealloc(ptr, layout.size(), layout.align())
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        extern "C" {
            fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
        }
        __rust_realloc(ptr, layout.size(), layout.align(), new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        extern "C" {
            fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;
        }
        __rust_alloc_zeroed(layout.size(), layout.align())
    }
}

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[global_allocator]
static ALLOCATOR: GpuAllocator = GpuAllocator;

// Re-export alloc types when feature enabled
// Use std when available, alloc when in no_std mode
#[cfg(all(feature = "alloc", feature = "std"))]
pub use std::boxed::Box;
#[cfg(all(feature = "alloc", feature = "std"))]
pub use std::string::String;
#[cfg(all(feature = "alloc", feature = "std"))]
pub use std::vec::Vec;
#[cfg(all(feature = "alloc", feature = "std"))]
pub use std::vec;
#[cfg(all(feature = "alloc", feature = "std"))]
pub use std::format;

#[cfg(all(feature = "alloc", not(feature = "std")))]
pub use alloc::boxed::Box;
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub use alloc::string::String;
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub use alloc::vec::Vec;
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub use alloc::vec;
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub use alloc::format;

// ============================================================================
// INTRINSICS - Raw GPU functions
// ============================================================================

mod intrinsics {
    extern "C" {
        // Core
        pub fn frame() -> i32;
        pub fn thread_id() -> i32;
        pub fn threadgroup_size() -> i32;

        // Rendering
        pub fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);

        // Input
        pub fn get_cursor_x() -> f32;
        pub fn get_cursor_y() -> f32;
        pub fn get_mouse_down() -> i32;
        pub fn get_time() -> f32;
        pub fn get_screen_width() -> f32;
        pub fn get_screen_height() -> f32;

        // Math - use __gpu_ prefix to prevent LLVM from substituting builtins
        #[link_name = "__gpu_sin"]
        pub fn sin(x: f32) -> f32;
        #[link_name = "__gpu_cos"]
        pub fn cos(x: f32) -> f32;
        #[link_name = "__gpu_sqrt"]
        pub fn sqrt(x: f32) -> f32;

        // Debug
        pub fn __gpu_debug_i32(val: i32);
        pub fn __gpu_debug_f32(val: f32);
        pub fn __gpu_debug_str(ptr: i32, len: i32);
    }
}

// ============================================================================
// PUBLIC API - Clean Rust interface
// ============================================================================

/// Get the current frame number (starts at 0, increments each frame)
#[inline(always)]
pub fn frame() -> i32 {
    unsafe { intrinsics::frame() }
}

/// Get the GPU thread ID within the threadgroup
#[inline(always)]
pub fn thread_id() -> i32 {
    unsafe { intrinsics::thread_id() }
}

/// Get the threadgroup size
#[inline(always)]
pub fn threadgroup_size() -> i32 {
    unsafe { intrinsics::threadgroup_size() }
}

/// Emit a colored quad (rectangle) to the screen
///
/// # Arguments
/// * `x` - X position (left edge)
/// * `y` - Y position (top edge)
/// * `w` - Width
/// * `h` - Height
/// * `color` - Color as packed u32 (use `Color::*` or `rgba()`)
#[inline(always)]
pub fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32) {
    unsafe { intrinsics::emit_quad(x, y, w, h, color) }
}

/// Get the current mouse/cursor X position
#[inline(always)]
pub fn cursor_x() -> f32 {
    unsafe { intrinsics::get_cursor_x() }
}

/// Get the current mouse/cursor Y position
#[inline(always)]
pub fn cursor_y() -> f32 {
    unsafe { intrinsics::get_cursor_y() }
}

/// Get cursor position as a tuple
#[inline(always)]
pub fn cursor_pos() -> (f32, f32) {
    (cursor_x(), cursor_y())
}

/// Check if the mouse button is pressed
#[inline(always)]
pub fn mouse_down() -> bool {
    unsafe { intrinsics::get_mouse_down() != 0 }
}

/// Get elapsed time in seconds
#[inline(always)]
pub fn time() -> f32 {
    unsafe { intrinsics::get_time() }
}

/// Get screen width in pixels
#[inline(always)]
pub fn screen_width() -> f32 {
    unsafe { intrinsics::get_screen_width() }
}

/// Get screen height in pixels
#[inline(always)]
pub fn screen_height() -> f32 {
    unsafe { intrinsics::get_screen_height() }
}

/// Get screen size as a tuple
#[inline(always)]
pub fn screen_size() -> (f32, f32) {
    (screen_width(), screen_height())
}

/// Compute sine of angle (radians)
#[inline(always)]
pub fn sin(x: f32) -> f32 {
    unsafe { intrinsics::sin(x) }
}

/// Compute cosine of angle (radians)
#[inline(always)]
pub fn cos(x: f32) -> f32 {
    unsafe { intrinsics::cos(x) }
}

/// Compute square root
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    unsafe { intrinsics::sqrt(x) }
}

/// Print a debug integer (visible in debug buffer)
#[inline(always)]
pub fn debug_i32(val: i32) {
    unsafe { intrinsics::__gpu_debug_i32(val) }
}

/// Print a debug float (visible in debug buffer)
#[inline(always)]
pub fn debug_f32(val: f32) {
    unsafe { intrinsics::__gpu_debug_f32(val) }
}

// ============================================================================
// COLOR - Helper type for colors
// ============================================================================

/// Pack RGBA values into a u32 color (0xRRGGBBAA format)
#[inline(always)]
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

/// Pack RGB values with full alpha
#[inline(always)]
pub const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    rgba(r, g, b, 255)
}

/// Color constants and utilities
pub struct Color;

impl Color {
    // Primary colors
    pub const RED: u32 = rgba(255, 0, 0, 255);
    pub const GREEN: u32 = rgba(0, 255, 0, 255);
    pub const BLUE: u32 = rgba(0, 0, 255, 255);

    // Secondary colors
    pub const YELLOW: u32 = rgba(255, 255, 0, 255);
    pub const CYAN: u32 = rgba(0, 255, 255, 255);
    pub const MAGENTA: u32 = rgba(255, 0, 255, 255);

    // Grayscale
    pub const WHITE: u32 = rgba(255, 255, 255, 255);
    pub const BLACK: u32 = rgba(0, 0, 0, 255);
    pub const GRAY: u32 = rgba(128, 128, 128, 255);
    pub const DARK_GRAY: u32 = rgba(64, 64, 64, 255);
    pub const LIGHT_GRAY: u32 = rgba(192, 192, 192, 255);

    // UI colors
    pub const ORANGE: u32 = rgba(255, 165, 0, 255);
    pub const PINK: u32 = rgba(255, 192, 203, 255);
    pub const PURPLE: u32 = rgba(128, 0, 128, 255);
    pub const BROWN: u32 = rgba(139, 69, 19, 255);

    // Material Design colors
    pub const DEEP_ORANGE: u32 = rgba(255, 87, 34, 255);
    pub const AMBER: u32 = rgba(255, 193, 7, 255);
    pub const TEAL: u32 = rgba(0, 150, 136, 255);
    pub const INDIGO: u32 = rgba(63, 81, 181, 255);

    // Transparent
    pub const TRANSPARENT: u32 = rgba(0, 0, 0, 0);

    /// Create a color with modified alpha
    #[inline(always)]
    pub const fn with_alpha(color: u32, alpha: u8) -> u32 {
        (color & 0xFFFFFF00) | (alpha as u32)
    }

    /// Blend two colors (simple linear interpolation)
    #[inline(always)]
    pub fn lerp(a: u32, b: u32, t: f32) -> u32 {
        let t = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
        let inv_t = 1.0 - t;

        let r_a = ((a >> 24) & 0xFF) as f32;
        let g_a = ((a >> 16) & 0xFF) as f32;
        let b_a = ((a >> 8) & 0xFF) as f32;
        let a_a = (a & 0xFF) as f32;

        let r_b = ((b >> 24) & 0xFF) as f32;
        let g_b = ((b >> 16) & 0xFF) as f32;
        let b_b = ((b >> 8) & 0xFF) as f32;
        let a_b = (b & 0xFF) as f32;

        let r = (r_a * inv_t + r_b * t) as u8;
        let g = (g_a * inv_t + g_b * t) as u8;
        let b = (b_a * inv_t + b_b * t) as u8;
        let a = (a_a * inv_t + a_b * t) as u8;

        rgba(r, g, b, a)
    }
}

// ============================================================================
// PRELUDE - Common imports
// ============================================================================

pub mod prelude {
    pub use crate::{
        // Core
        frame, thread_id, threadgroup_size,
        // Rendering
        emit_quad, rgba, rgb, Color,
        // Input
        cursor_x, cursor_y, cursor_pos, mouse_down,
        time, screen_width, screen_height, screen_size,
        // Math
        sin, cos, sqrt,
        // Debug
        debug_i32, debug_f32,
    };

    // Re-export alloc types in prelude when available
    #[cfg(feature = "alloc")]
    pub use crate::{Vec, String, Box, vec, format};

    // Re-export collections
    #[cfg(feature = "alloc")]
    pub use crate::collections::HashMap;
}

// ============================================================================
// PANIC HANDLER HELPER
// ============================================================================

/// Default panic handler for GPU apps (just loops forever)
///
/// Apps can use this macro instead of writing their own:
/// ```ignore
/// gpu_std::default_panic_handler!();
/// ```
#[macro_export]
macro_rules! default_panic_handler {
    () => {
        #[panic_handler]
        fn panic(_: &core::panic::PanicInfo) -> ! {
            loop {}
        }
    };
}
