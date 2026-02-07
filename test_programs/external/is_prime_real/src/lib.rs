// Integer-only is_prime from https://users.rust-lang.org/t/u32-integer-square-root/86909
// Uses i*i <= num instead of sqrt to avoid floats
// Using u32 to avoid 64-bit library calls

fn is_prime(num: u32) -> bool {
    if num <= 1 {
        return false;
    }
    if num <= 3 {
        return true;
    }
    if num % 2 == 0 {
        return false;
    }
    let mut i: u32 = 3;
    while i * i <= num {
        if num % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

// Entry point - count primes up to 100 using while loop (no iterator)
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut count = 0i32;
    let mut n: u32 = 2;
    while n <= 100 {
        if is_prime(n) {
            count += 1;
        }
        n += 1;
    }
    count  // Should return 25 (there are 25 primes <= 100)
}
