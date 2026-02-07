// EXACT CODE FROM https://github.com/eliovir/rust-examples/blob/master/fibonacci.rs
// NO MODIFICATIONS TO THE ALGORITHM

/// Non reccursive function.
///
/// `n` the rank used to compute the member of the sequence.
pub fn fibonacci(n: i32) -> u64 {
    if n < 0 {
        panic!("{} is negative!", n);
    } else if n == 0 {
        panic!("zero is not a right argument to fibonacci()!");
    } else if n == 1 {
        return 1;
    }

    let mut sum = 0;
    let mut last = 0;
    let mut curr = 1;
    for _i in 1..n {
        sum = last + curr;
        last = curr;
        curr = sum;
    }
    sum
}

// Entry point for WASM - calls the REAL unmodified function
#[no_mangle]
pub extern "C" fn main() -> i32 {
    fibonacci(10) as i32  // Should return 55
}
