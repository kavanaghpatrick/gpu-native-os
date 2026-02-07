//! Minimal Mandelbrot-like test app
//!
//! Tests the pattern: outer loop calls an inline function that has its own loop.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

/// Inner function with a loop - prevent full inlining with #[inline(never)]
#[inline(never)]
fn compute(px: i32, py: i32) -> i32 {
    // Mimic mandelbrot pattern: loop until condition or max iterations
    let max_iter = 5;
    let mut iter = 0;

    while iter < max_iter {
        // Some dummy computation
        let _sum = px + py + iter;
        iter += 1;
    }

    iter
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut count = 0;

    // Outer loop: 0 to 2
    let mut py = 0;
    while py < 2 {
        // Inner loop: 0 to 2
        let mut px = 0;
        while px < 2 {
            // Call function with loop inside
            let result = compute(px, py);
            count += result;
            px += 1;
        }
        py += 1;
    }

    // Expected: 2 * 2 * 5 = 20
    count
}
