#![no_std]

//! Prime Sieve Implementation
//! Count primes up to N using trial division
//!
//! THE GPU IS THE COMPUTER - classic number theory algorithm

/// Check if n is prime using trial division
#[inline(never)]
fn is_prime(n: i32) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Count primes from 2 to n (inclusive)
#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    let mut count = 0;
    for i in 2..=n {
        if is_prime(i) {
            count += 1;
        }
    }
    count
}

/// Find the nth prime number
#[no_mangle]
pub extern "C" fn nth_prime(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }

    let mut count = 0;
    let mut candidate = 2;

    while count < n {
        if is_prime(candidate) {
            count += 1;
            if count == n {
                return candidate;
            }
        }
        candidate += 1;
    }

    candidate - 1
}

/// Sum of all primes up to n
#[no_mangle]
pub extern "C" fn sum_primes(n: i32) -> i32 {
    let mut sum = 0;
    for i in 2..=n {
        if is_prime(i) {
            sum += i;
        }
    }
    sum
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
