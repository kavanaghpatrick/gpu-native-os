// Source: Inspired by https://rosettacode.org/wiki/FizzBuzz#Rust
// MODIFIED: Returns sum of non-FizzBuzz numbers (no_std compatible)
// Numbers 1-100: skip if divisible by 3 (Fizz) or 5 (Buzz), sum the rest

pub fn fizzbuzz_sum(limit: u32) -> u32 {
    let mut sum = 0;
    let mut i = 1;
    while i <= limit {
        // Only sum numbers that are NOT Fizz, Buzz, or FizzBuzz
        if i % 3 != 0 && i % 5 != 0 {
            sum += i;
        }
        i += 1;
    }
    sum
}
