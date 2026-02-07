#![no_std]

//! CRC32 Implementation - Real crate from crates.io
//! https://crates.io/crates/crc32fast
//!
//! THE GPU IS THE COMPUTER - unmodified algorithm from the wild

use crc32fast::Hasher;

/// CRC32 hash a single i32 value
#[no_mangle]
pub extern "C" fn main(value: i32) -> i32 {
    let bytes = value.to_le_bytes();
    let mut hasher = Hasher::new();
    hasher.update(&bytes);
    hasher.finalize() as i32
}

/// CRC32 with initial value
#[no_mangle]
pub extern "C" fn crc32_with_init(value: i32, init: u32) -> i32 {
    let bytes = value.to_le_bytes();
    let mut hasher = Hasher::new_with_initial(init);
    hasher.update(&bytes);
    hasher.finalize() as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
