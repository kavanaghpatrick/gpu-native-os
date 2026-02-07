#![no_std]

//! FNV-1a Hash Implementation
//! Source: https://github.com/servo/rust-fnv (adapted for no_std/WASM)
//!
//! THE GPU IS THE COMPUTER - unmodified algorithm from the wild
//!
//! This uses 64-bit operations which are natively supported by Apple Silicon GPUs.
//! Issue #188 RESOLVED: i64 operations fully implemented.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x0100000001b3;

/// FNV-1a hash a single i32 value
/// Returns lower 32 bits of hash
#[no_mangle]
pub extern "C" fn main(value: i32) -> i32 {
    let bytes = value.to_le_bytes();
    let mut hash = FNV_OFFSET;

    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Return lower 32 bits
    hash as i32
}

/// Hash multiple values (demonstrates more complex logic)
#[no_mangle]
pub extern "C" fn fnv_hash_pair(a: i32, b: i32) -> i32 {
    let mut hash = FNV_OFFSET;

    // Hash first value
    for byte in a.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Hash second value
    for byte in b.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
