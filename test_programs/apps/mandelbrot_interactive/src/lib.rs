//! Interactive Mandelbrot Viewer - GPU Edition
//!
//! Mandelbrot fractal with zoom animation.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Display constants
const GRID_SIZE: usize = 64;  // 64x64 pixels
const CELL_SIZE: f32 = 10.0;
const MAX_ITER: u32 = 50;

// Interesting point to zoom into
const ZOOM_X: f32 = -0.7436;
const ZOOM_Y: f32 = 0.1318;

/// Compute Mandelbrot iteration count
fn mandelbrot(cx: f32, cy: f32) -> u32 {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut iter = 0u32;

    while iter < MAX_ITER {
        let x2 = x * x;
        let y2 = y * y;

        if x2 + y2 > 4.0 {
            return iter;
        }

        let xtemp = x2 - y2 + cx;
        y = 2.0 * x * y + cy;
        x = xtemp;
        iter += 1;
    }

    MAX_ITER
}

/// Map iteration count to color
fn iter_to_color(iter: u32) -> u32 {
    if iter >= MAX_ITER {
        return 0x000000FF;  // Black for inside set
    }

    // Create a gradient based on iteration count
    let t = (iter as f32) / (MAX_ITER as f32);

    // Simple RGB gradient
    let r = ((1.0 - t) * 255.0) as u32;
    let g = ((t * 0.5) * 255.0) as u32;
    let b = ((t) * 255.0) as u32;

    (r << 24) | (g << 16) | (b << 8) | 0xFF
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };

    // Zoom level increases over time
    let zoom_factor = 1.0 + (current_frame as f32) * 0.01;
    let scale = 3.0 / zoom_factor;

    // Center of view (slowly approach zoom point)
    let t = (current_frame as f32) / 1000.0;
    let cx = -0.5 + (ZOOM_X - (-0.5)) * t.min(1.0);
    let cy = 0.0 + (ZOOM_Y - 0.0) * t.min(1.0);

    let mut quad_count = 0;

    // Render each pixel as a colored quad
    let mut py = 0usize;
    while py < GRID_SIZE {
        let mut px = 0usize;
        while px < GRID_SIZE {
            // Map pixel to complex plane
            let x = cx + (px as f32 / GRID_SIZE as f32 - 0.5) * scale;
            let y = cy + (py as f32 / GRID_SIZE as f32 - 0.5) * scale;

            let iter = mandelbrot(x, y);
            let color = iter_to_color(iter);

            // Screen position
            let sx = (px as f32) * CELL_SIZE;
            let sy = (py as f32) * CELL_SIZE;

            unsafe {
                emit_quad(sx, sy, CELL_SIZE, CELL_SIZE, color);
            }
            quad_count += 1;

            px += 1;
        }
        py += 1;
    }

    quad_count
}
