# External Test Programs

Real-world Rust code compiled to WASM to validate GPU bytecode execution.
These are NOT workarounds - they represent unmodified algorithms proving we can run REAL code.

## Pure Algorithm Tests (no dependencies)

| Program | What it Tests | Expected Output |
|---------|--------------|-----------------|
| `bubble_sort` | Nested loops, conditionals, array mutation, swap | Checksum of sorted array |
| `prime_sieve` | Trial division, modulo, while loops | Count of primes (25 primes <= 100) |
| `mandelbrot` | Floating point math, iteration, complex arithmetic | Escape iteration count |
| `fnv_hash` | 64-bit operations, byte manipulation, wrapping multiply | Lower 32 bits of FNV-1a hash |
| `fibonacci_real` | Loop-based computation, u64 arithmetic | Fibonacci(10) = 55 |
| `is_prime_real` | Integer-only primality (no sqrt), while loop | 25 (primes <= 100) |
| `for_loop_test` | For vs while vs iterator compilation differences | 5050 (sum 1..100) |

## External Crate Tests (real crates.io dependencies)

| Program | Crate | What it Proves |
|---------|-------|---------------|
| `xxhash` | xxhash-rust 0.8 | Real hashing crate compiles to GPU bytecode |
| `crc32` | crc32fast 1.3 | Real checksum crate compiles to GPU bytecode |
| `fastrand` | fastrand 2.0 | Real RNG crate compiles to GPU bytecode |
| `siphasher` | siphasher 1.0 | Real hash crate (used by Rust HashMap!) compiles |

## Building

Each program is a standalone Cargo project. Build with:

```bash
cd <program>
cargo build --release --target wasm32-unknown-unknown
```

The resulting `.wasm` file can be loaded and executed by the GPU bytecode VM.

## Verification

Each program has known expected outputs that can be verified against native Rust execution:

- `prime_sieve(100)` = 25 primes
- `fibonacci(10)` = 55
- `is_prime_real` main() = 25 primes
- `bubble_sort` with [8,3,5,1,7,2,6,4] should sort to [1,2,3,4,5,6,7,8]
- `mandelbrot` origin point (0,0) should iterate 100 times (in set)
- All sum functions in `for_loop_test` should return 5050
