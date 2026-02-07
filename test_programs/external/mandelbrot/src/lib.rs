#![no_std]

//! Mandelbrot Set Calculator
//! Classic fractal computation - tests floating point, loops, complex logic
//!
//! THE GPU IS THE COMPUTER - this is PERFECT for GPU parallelization

const MAX_ITER: i32 = 100;

/// Calculate escape iteration for a point in the Mandelbrot set
/// Uses fixed-point representation: x and y are scaled by 1000
/// Returns iteration count (0-100) or -1 if in set
#[no_mangle]
pub extern "C" fn main(x_scaled: i32, y_scaled: i32) -> i32 {
    // Convert from fixed-point (scaled by 1000) to float
    let x0 = (x_scaled as f32) / 1000.0;
    let y0 = (y_scaled as f32) / 1000.0;

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut iter = 0;

    while x * x + y * y <= 4.0 && iter < MAX_ITER {
        let x_temp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = x_temp;
        iter += 1;
    }

    if iter == MAX_ITER {
        -1 // In set
    } else {
        iter
    }
}

/// Calculate escape for a grid point
/// row and col are 0-based indices, width is grid width
/// Maps to complex plane: [-2, 1] x [-1.5, 1.5]
#[no_mangle]
pub extern "C" fn mandelbrot_pixel(row: i32, col: i32, width: i32, height: i32) -> i32 {
    // Map pixel coordinates to complex plane
    let x0 = -2.0 + (col as f32) / (width as f32) * 3.0;
    let y0 = -1.5 + (row as f32) / (height as f32) * 3.0;

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut iter = 0;

    while x * x + y * y <= 4.0 && iter < MAX_ITER {
        let x_temp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = x_temp;
        iter += 1;
    }

    iter
}

/// Sum escape times for a row (for verification)
#[no_mangle]
pub extern "C" fn mandelbrot_row_sum(row: i32, width: i32, height: i32) -> i32 {
    let mut sum = 0;
    for col in 0..width {
        sum += mandelbrot_pixel(row, col, width, height);
    }
    sum
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
