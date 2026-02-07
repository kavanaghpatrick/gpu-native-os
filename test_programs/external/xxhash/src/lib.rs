#![no_std]

//! xxHash Implementation - Real crate from crates.io
//! https://crates.io/crates/xxhash-rust
//!
//! THE GPU IS THE COMPUTER - unmodified algorithm from the wild

use xxhash_rust::xxh32;

/// Hash a single i32 value using xxHash32
#[no_mangle]
pub extern "C" fn main(value: i32) -> i32 {
    let bytes = value.to_le_bytes();
    xxh32::xxh32(&bytes, 0) as i32
}

/// Hash with a seed
#[no_mangle]
pub extern "C" fn xxhash_seeded(value: i32, seed: u32) -> i32 {
    let bytes = value.to_le_bytes();
    xxh32::xxh32(&bytes, seed) as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
