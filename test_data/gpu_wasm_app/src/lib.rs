//! Standard Rust app - works on CPU or GPU
//!
//! This is a minimal no_std Rust crate. The developer doesn't need to know
//! whether it runs on CPU or GPU - our toolchain handles that.
//!
//! Write normal Rust. We handle the rest.

#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Computes sum of 1 to n
///
/// Standard Rust function - no GPU-specific code.
#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    let mut sum = 0i32;
    let mut i = 1i32;

    while i <= n {
        sum += i;
        i += 1;
    }

    sum
}
