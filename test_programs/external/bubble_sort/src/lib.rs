#![no_std]

//! Bubble Sort Implementation
//! Classic sorting algorithm - tests loops, conditionals, array access
//!
//! THE GPU IS THE COMPUTER - unmodified sorting algorithm

/// Sort an array of 8 values using bubble sort
/// Input: packed into two i32s (4 bytes each as i8s)
/// Returns: checksum of sorted values
#[no_mangle]
pub extern "C" fn main(packed_lo: i32, packed_hi: i32) -> i32 {
    // Unpack 8 values from two i32s
    let mut arr: [i8; 8] = [0; 8];

    let lo_bytes = packed_lo.to_le_bytes();
    let hi_bytes = packed_hi.to_le_bytes();

    arr[0] = lo_bytes[0] as i8;
    arr[1] = lo_bytes[1] as i8;
    arr[2] = lo_bytes[2] as i8;
    arr[3] = lo_bytes[3] as i8;
    arr[4] = hi_bytes[0] as i8;
    arr[5] = hi_bytes[1] as i8;
    arr[6] = hi_bytes[2] as i8;
    arr[7] = hi_bytes[3] as i8;

    // Bubble sort
    let n = arr.len();
    for i in 0..n {
        for j in 0..(n - i - 1) {
            if arr[j] > arr[j + 1] {
                // Swap
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    // Return checksum: sum of (value * position)
    let mut checksum: i32 = 0;
    for i in 0..n {
        checksum += (arr[i] as i32) * (i as i32 + 1);
    }
    checksum
}

/// Check if array is sorted
#[no_mangle]
pub extern "C" fn is_sorted(packed_lo: i32, packed_hi: i32) -> i32 {
    let lo_bytes = packed_lo.to_le_bytes();
    let hi_bytes = packed_hi.to_le_bytes();

    let arr: [i8; 8] = [
        lo_bytes[0] as i8,
        lo_bytes[1] as i8,
        lo_bytes[2] as i8,
        lo_bytes[3] as i8,
        hi_bytes[0] as i8,
        hi_bytes[1] as i8,
        hi_bytes[2] as i8,
        hi_bytes[3] as i8,
    ];

    for i in 0..7 {
        if arr[i] > arr[i + 1] {
            return 0; // Not sorted
        }
    }
    1 // Sorted
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
