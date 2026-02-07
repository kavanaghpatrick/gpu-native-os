#![no_std]

//! SipHash Implementation - Real crate from crates.io
//! https://crates.io/crates/siphasher
//! Used by Rust's HashMap!
//!
//! THE GPU IS THE COMPUTER - unmodified algorithm from the wild

use siphasher::sip::SipHasher13;
use core::hash::Hasher;

/// SipHash a single i32 value
#[no_mangle]
pub extern "C" fn main(value: i32) -> i32 {
    let bytes = value.to_le_bytes();
    let mut hasher = SipHasher13::new();
    hasher.write(&bytes);
    hasher.finish() as i32
}

/// SipHash with keys
#[no_mangle]
pub extern "C" fn siphash_keyed(value: i32, key0: i32, key1: i32) -> i32 {
    let bytes = value.to_le_bytes();
    let mut hasher = SipHasher13::new_with_keys(key0 as u64, key1 as u64);
    hasher.write(&bytes);
    hasher.finish() as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
