#![no_std]

mod original;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Test: collatz_steps(27) = 111 steps to reach 1
// Sequence: 27 -> 82 -> 41 -> 124 -> ... -> 4 -> 2 -> 1 (112 elements, 111 steps)
#[no_mangle]
pub extern "C" fn main() -> i32 {
    original::collatz_steps(27) as i32
}
