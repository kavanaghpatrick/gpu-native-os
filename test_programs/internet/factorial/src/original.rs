// Source: https://rosettacode.org/wiki/Factorial#Rust
// UNMODIFIED - Iterative using fold

pub fn factorial(n: u32) -> u32 {
    (1..=n).fold(1, |acc, x| acc * x)
}
