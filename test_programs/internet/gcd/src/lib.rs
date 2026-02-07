#![no_std]

mod original;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Test: gcd(1071, 462) = 21
#[no_mangle]
pub extern "C" fn main() -> i32 {
    original::gcd(1071, 462) as i32
}
