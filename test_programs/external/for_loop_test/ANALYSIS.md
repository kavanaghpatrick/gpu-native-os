# Rust For Loop WASM Indirect Call Analysis

## Summary

**Finding: Rust `for` loops do NOT inherently generate `call_indirect` instructions in WASM.**

The indirect calls observed in WASM output are from:
1. **Panic/formatting infrastructure** (std library)
2. **Trait objects** (`&dyn Trait` or `Box<dyn Trait>`)
3. **Boxed iterators** (`Box<dyn Iterator>`)

## Test Results

### Test 1: no_std with Regular For Loop
```rust
#![no_std]
for n in 1..=100i32 { sum += n; }
```
**Result: 0 `call_indirect` instructions**

The for loop desugars to `Iterator::next()` calls, but with monomorphization,
these become direct `call` instructions to concrete implementations.

### Test 2: std with Panic-Inducing Code
```rust
assert!(n > 0, "n must be positive: {}", n);
```
**Result: 9 `call_indirect` instructions**

All indirect calls are in:
- `core::fmt::Formatter::pad_integral` - formatting with padding
- `core::fmt::Display::fmt` implementations for i32, usize
- `core::panic::PanicPayload` trait methods

These are NOT from the for loop itself.

### Test 3: Trait Objects
```rust
let adder: &dyn Adder = if flag == 0 { &add_one } else { &add_two };
adder.add(n)
```
**Result: 1 `call_indirect` instruction**

Trait objects require vtable dispatch, which becomes `call_indirect`.

### Test 4: Boxed Iterator
```rust
let iter: Box<dyn Iterator<Item = i32>> = ...;
for n in iter { sum += n; }
```
**Result: 1 `call_indirect` instruction per iteration**

`Box<dyn Iterator>` erases the concrete type, requiring vtable dispatch for `next()`.

## Why While Loop Optimizes Better

```rust
// Constant-folded to: i32.const 5050
while n <= 100 { sum += n; n += 1; }

// Still a loop at runtime
for n in 1..=100 { sum += n; }
```

`RangeInclusive<i32>` uses a 3-field struct (`start`, `end`, `exhausted` flag) with
special handling for the inclusive boundary. This complexity prevents LLVM from
recognizing the closed-form solution.

`Range<i32>` (`1..101`) optimizes identically to the while loop.

## Conclusions

1. **For loops don't cause indirect calls** - It's the associated infrastructure
2. **Use `#![no_std]` for minimal WASM** - Eliminates panic formatting vtables
3. **Avoid `Box<dyn Iterator>`** - Use concrete iterator types when possible
4. **Prefer exclusive ranges** - `1..101` optimizes better than `1..=100`
5. **Indirect calls come from trait objects** - The `dyn` keyword is the signal

## Recommendations for WASM Translation

If your WASM-to-X translator sees `call_indirect`:

1. **From formatting code**: Can ignore if panic paths aren't taken
2. **From trait objects**: Must handle vtable dispatch
3. **From iterator loops**: Only if using `dyn Iterator`

For a bytecode VM targeting GPU execution, the cleanest approach is:
- Require `#![no_std]` code
- Use while loops or explicit iterator stepping
- Avoid trait objects (`&dyn T`)
- The resulting WASM will have zero `call_indirect` instructions
