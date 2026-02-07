// Source: https://rosettacode.org/wiki/Greatest_common_divisor#Rust
// UNMODIFIED - Iterative Euclid Algorithm

pub fn gcd(mut u: u64, mut v: u64) -> u64 {
    while v != 0 {
        let t = v;
        v = u % v;
        u = t;
    }
    u
}
