#![no_std]

// THE GPU IS THE COMPUTER
// Game of Life that tests register pressure
// Values are used to prevent optimizer from eliminating code

const WIDTH: i32 = 64;
const HEIGHT: i32 = 64;

/// Wrap coordinates to handle toroidal grid
/// ~15 WASM operations when expanded
#[inline(always)]
fn get_index(row: i32, col: i32) -> i32 {
    let row = ((row % HEIGHT) + HEIGHT) % HEIGHT;
    let col = ((col % WIDTH) + WIDTH) % WIDTH;
    row * WIDTH + col
}

/// Count live neighbors - uses all index values to prevent optimization
#[inline(always)]
fn count_neighbors(row: i32, col: i32) -> i32 {
    // Sum all 8 neighbor indices - this prevents dead code elimination
    // Each get_index call expands to ~15 operations
    // 8 calls = ~120 operations with many intermediate values
    let n = get_index(row - 1, col);
    let s = get_index(row + 1, col);
    let w = get_index(row, col - 1);
    let e = get_index(row, col + 1);
    let nw = get_index(row - 1, col - 1);
    let ne = get_index(row - 1, col + 1);
    let sw = get_index(row + 1, col - 1);
    let se = get_index(row + 1, col + 1);

    // Use all values to prevent optimization
    // This creates register pressure as all 8 values are live simultaneously
    (n + s + w + e + nw + ne + sw + se) % 9
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main(cell_index: i32) -> i32 {
    let row = cell_index / WIDTH;
    let col = cell_index % WIDTH;
    let current = cell_index % 2;
    let neighbors = count_neighbors(row, col);

    if current == 1 {
        if neighbors == 2 || neighbors == 3 { 1 } else { 0 }
    } else {
        if neighbors == 3 { 1 } else { 0 }
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! { loop {} }
