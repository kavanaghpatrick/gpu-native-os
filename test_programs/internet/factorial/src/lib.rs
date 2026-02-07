#![no_std]

mod original;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Test: factorial(10) = 3628800
#[no_mangle]
pub extern "C" fn main() -> i32 {
    original::factorial(10) as i32
}
