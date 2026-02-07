#![no_std]

mod original;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Test: sum_digits(1234, 10) = 1+2+3+4 = 10
#[no_mangle]
pub extern "C" fn main() -> i32 {
    original::sum_digits(1234, 10) as i32
}
