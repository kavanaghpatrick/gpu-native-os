//! Drawing App - GPU Edition
//!
//! Simple canvas that tracks cursor position.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn get_cursor_x() -> f32;
    fn get_cursor_y() -> f32;
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Canvas constants
const CANVAS_WIDTH: f32 = 700.0;
const CANVAS_HEIGHT: f32 = 500.0;
const CANVAS_X: f32 = 50.0;
const CANVAS_Y: f32 = 50.0;
const BRUSH_SIZE: f32 = 20.0;

// Palette colors
const PALETTE: [u32; 8] = [
    0xFF0000FF, // Red
    0x00FF00FF, // Green
    0x0000FFFF, // Blue
    0xFFFF00FF, // Yellow
    0xFF00FFFF, // Magenta
    0x00FFFFFF, // Cyan
    0xFFFFFFFF, // White
    0x000000FF, // Black
];

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };
    let cursor_x = unsafe { get_cursor_x() };
    let cursor_y = unsafe { get_cursor_y() };

    let mut quad_count = 0;

    // Draw canvas background
    unsafe {
        emit_quad(CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT, 0xF5F5F5FF);
    }
    quad_count += 1;

    // Draw canvas border
    unsafe {
        // Top
        emit_quad(CANVAS_X - 3.0, CANVAS_Y - 3.0, CANVAS_WIDTH + 6.0, 3.0, 0x333333FF);
        // Bottom
        emit_quad(CANVAS_X - 3.0, CANVAS_Y + CANVAS_HEIGHT, CANVAS_WIDTH + 6.0, 3.0, 0x333333FF);
        // Left
        emit_quad(CANVAS_X - 3.0, CANVAS_Y, 3.0, CANVAS_HEIGHT, 0x333333FF);
        // Right
        emit_quad(CANVAS_X + CANVAS_WIDTH, CANVAS_Y, 3.0, CANVAS_HEIGHT, 0x333333FF);
    }
    quad_count += 4;

    // Draw color palette
    let palette_y = CANVAS_Y + CANVAS_HEIGHT + 20.0;
    let mut i = 0usize;
    while i < 8 {
        let px = CANVAS_X + (i as f32) * 50.0;
        unsafe {
            emit_quad(px, palette_y, 40.0, 40.0, PALETTE[i]);
        }
        quad_count += 1;
        i += 1;
    }

    // Draw cursor indicator (brush preview)
    // Change color based on position over palette
    let selected_color = if cursor_y > palette_y && cursor_y < palette_y + 40.0 {
        let idx = ((cursor_x - CANVAS_X) / 50.0) as usize;
        if idx < 8 {
            PALETTE[idx]
        } else {
            PALETTE[0]
        }
    } else {
        // Cycle through colors based on frame
        PALETTE[(current_frame / 60) as usize % 8]
    };

    // Draw brush at cursor position
    unsafe {
        emit_quad(
            cursor_x - BRUSH_SIZE / 2.0,
            cursor_y - BRUSH_SIZE / 2.0,
            BRUSH_SIZE,
            BRUSH_SIZE,
            selected_color,
        );
    }
    quad_count += 1;

    // Draw some demo strokes based on frame (sine wave pattern)
    let num_points = 20;
    let mut i = 0;
    while i < num_points {
        let t = (i as f32) / (num_points as f32);
        let x = CANVAS_X + 50.0 + t * (CANVAS_WIDTH - 100.0);

        // Sine wave
        let phase = t * 6.28 + (current_frame as f32) * 0.05;
        let y = CANVAS_Y + CANVAS_HEIGHT / 2.0 + sine_approx(phase) * 100.0;

        let color = PALETTE[i as usize % 8];
        unsafe {
            emit_quad(x - 5.0, y - 5.0, 10.0, 10.0, color);
        }
        quad_count += 1;
        i += 1;
    }

    quad_count
}

/// Sine approximation
fn sine_approx(x: f32) -> f32 {
    let mut x = x % 6.28318;
    if x > 3.14159 {
        x -= 6.28318;
    }
    if x < -3.14159 {
        x += 6.28318;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    x - x3 / 6.0 + x5 / 120.0
}
