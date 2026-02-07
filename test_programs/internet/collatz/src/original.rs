// Source: https://rosettacode.org/wiki/Collatz_conjecture#Rust
// MODIFIED: Returns step count instead of Vec (no_std compatible)
// Original algorithm preserved: n/2 if even, 3n+1 if odd

pub fn collatz_steps(mut n: u64) -> u32 {
    let mut steps = 0;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        steps += 1;
    }
    steps
}
