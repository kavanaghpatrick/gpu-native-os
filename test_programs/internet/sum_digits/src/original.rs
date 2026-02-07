// Source: https://rosettacode.org/wiki/Sum_digits_of_an_integer#Rust
// UNMODIFIED - Rosetta Code implementation

pub fn sum_digits(mut n: u64, base: u64) -> u64 {
    let mut sum = 0;
    while n > 0 {
        sum += n % base;
        n /= base;
    }
    sum
}
