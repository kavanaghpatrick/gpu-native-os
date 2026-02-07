// Comprehensive test: For loops and indirect calls in WASM
//
// Summary: Rust for loops do NOT inherently generate call_indirect.
// Indirect calls come from:
// 1. Panic/formatting infrastructure (std)
// 2. Trait objects (&dyn T)
// 3. Boxed iterators (Box<dyn Iterator>)

#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// ============================================================================
// PART 1: For loops - NO indirect calls
// ============================================================================

// RangeInclusive (..=) - generates loop with direct calls
#[no_mangle]
pub extern "C" fn sum_for_inclusive() -> i32 {
    let mut sum: i32 = 0;
    for n in 1..=100i32 {
        sum += n;
    }
    sum
}

// Range (..) - often constant-folded by LLVM
#[no_mangle]
pub extern "C" fn sum_for_exclusive() -> i32 {
    let mut sum: i32 = 0;
    for n in 1..101i32 {
        sum += n;
    }
    sum
}

// While loop - equivalent to Range, often constant-folded
#[no_mangle]
pub extern "C" fn sum_while() -> i32 {
    let mut sum: i32 = 0;
    let mut n: i32 = 1;
    while n <= 100 {
        sum += n;
        n += 1;
    }
    sum
}

// Array iteration - NO indirect calls
#[no_mangle]
pub extern "C" fn sum_array() -> i32 {
    let arr = [1i32, 2, 3, 4, 5];
    let mut sum: i32 = 0;
    for &n in arr.iter() {
        sum += n;
    }
    sum
}

// ============================================================================
// PART 2: Trait objects - WILL generate indirect calls
// ============================================================================

trait Adder {
    fn add(&self, x: i32) -> i32;
}

struct AddOne;
impl Adder for AddOne {
    fn add(&self, x: i32) -> i32 { x + 1 }
}

struct AddTwo;
impl Adder for AddTwo {
    fn add(&self, x: i32) -> i32 { x + 2 }
}

// This function WILL use call_indirect for adder.add()
#[no_mangle]
pub extern "C" fn sum_with_dyn_trait(flag: i32) -> i32 {
    let add_one = AddOne;
    let add_two = AddTwo;
    // &dyn Adder requires vtable dispatch
    let adder: &dyn Adder = if flag == 0 { &add_one } else { &add_two };

    let mut sum: i32 = 0;
    for n in 1..=10i32 {
        sum += adder.add(n);  // <-- call_indirect here
    }
    sum
}

// ============================================================================
// PART 3: Static dispatch alternative - NO indirect calls
// ============================================================================

// Using generics instead of trait objects
fn sum_with_adder<A: Adder>(adder: &A) -> i32 {
    let mut sum: i32 = 0;
    for n in 1..=10i32 {
        sum += adder.add(n);  // Direct call due to monomorphization
    }
    sum
}

#[no_mangle]
pub extern "C" fn sum_static_dispatch_one() -> i32 {
    sum_with_adder(&AddOne)
}

#[no_mangle]
pub extern "C" fn sum_static_dispatch_two() -> i32 {
    sum_with_adder(&AddTwo)
}
