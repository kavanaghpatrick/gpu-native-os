#![no_std]

//! fastrand - Fast random number generator from crates.io
//! https://crates.io/crates/fastrand
//!
//! THE GPU IS THE COMPUTER - unmodified algorithm from the wild

use fastrand::Rng;

/// Generate a random number with seed
#[no_mangle]
pub extern "C" fn main(seed: i32) -> i32 {
    let mut rng = Rng::with_seed(seed as u64);
    rng.i32(..)
}

/// Generate random in range
#[no_mangle]
pub extern "C" fn rand_range(seed: i32, max: i32) -> i32 {
    let mut rng = Rng::with_seed(seed as u64);
    rng.i32(0..max)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
