#![no_std]

mod original;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Test: Sum of numbers 1-100 that are NOT divisible by 3 or 5
// Total 1-100 = 5050, minus Fizz/Buzz/FizzBuzz numbers
// Expected: 2632
#[no_mangle]
pub extern "C" fn main() -> i32 {
    original::fizzbuzz_sum(100) as i32
}
