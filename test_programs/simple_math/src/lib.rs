#![no_std]

// Main entry point - takes parameter to prevent constant folding
#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    // Compute factorial of n
    factorial(n)
}

// Simple factorial - tests loops
#[inline(never)]
fn factorial(n: i32) -> i32 {
    let mut result = 1;
    let mut i = 2;
    while i <= n {
        result *= i;
        i += 1;
    }
    result
}

// Fibonacci with parameter
#[no_mangle]
pub extern "C" fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    let mut a = 0;
    let mut b = 1;
    let mut i = 2;
    while i <= n {
        let temp = a + b;
        a = b;
        b = temp;
        i += 1;
    }
    b
}

// Sum 1 to n
#[no_mangle]
pub extern "C" fn sum_to_n(n: i32) -> i32 {
    let mut sum = 0;
    let mut i = 1;
    while i <= n {
        sum += i;
        i += 1;
    }
    sum
}

// Panic handler for no_std
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
