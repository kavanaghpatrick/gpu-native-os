//! Pong - GPU Edition
//!
//! Classic Pong game running entirely on the GPU.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Game constants
const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;
const PADDLE_WIDTH: f32 = 15.0;
const PADDLE_HEIGHT: f32 = 80.0;
const BALL_SIZE: f32 = 15.0;
const PADDLE_MARGIN: f32 = 30.0;

// Colors
const COLOR_PADDLE: u32 = 0xFFFFFFFF;  // White
const COLOR_BALL: u32 = 0xFFFFFFFF;    // White
const COLOR_BG: u32 = 0x000000FF;      // Black

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };

    // Animate ball position using sine-like oscillation
    // Ball bounces across the screen
    let time = current_frame as f32 / 60.0;  // ~60fps

    // Ball position oscillates
    let ball_phase = (time * 2.0) % 6.28;
    let ball_x = 200.0 + sine_approx(ball_phase) * 200.0;
    let ball_y = 300.0 + sine_approx(ball_phase * 1.3) * 200.0;

    // Paddles follow ball Y with some delay
    let left_paddle_y = ball_y - PADDLE_HEIGHT / 2.0;
    let right_paddle_y = ball_y - PADDLE_HEIGHT / 2.0 + sine_approx(time) * 50.0;

    let mut quad_count = 0;

    // Background (center line)
    let mut y = 0.0f32;
    while y < SCREEN_HEIGHT {
        unsafe {
            emit_quad(SCREEN_WIDTH / 2.0 - 2.0, y, 4.0, 20.0, 0x404040FF);
        }
        quad_count += 1;
        y += 40.0;
    }

    // Left paddle
    unsafe {
        emit_quad(PADDLE_MARGIN, clamp(left_paddle_y, 0.0, SCREEN_HEIGHT - PADDLE_HEIGHT),
                  PADDLE_WIDTH, PADDLE_HEIGHT, COLOR_PADDLE);
    }
    quad_count += 1;

    // Right paddle
    unsafe {
        emit_quad(SCREEN_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH,
                  clamp(right_paddle_y, 0.0, SCREEN_HEIGHT - PADDLE_HEIGHT),
                  PADDLE_WIDTH, PADDLE_HEIGHT, COLOR_PADDLE);
    }
    quad_count += 1;

    // Ball
    unsafe {
        emit_quad(clamp(ball_x, 0.0, SCREEN_WIDTH - BALL_SIZE),
                  clamp(ball_y, 0.0, SCREEN_HEIGHT - BALL_SIZE),
                  BALL_SIZE, BALL_SIZE, COLOR_BALL);
    }
    quad_count += 1;

    quad_count
}

/// Approximate sine using polynomial (no libm needed)
fn sine_approx(x: f32) -> f32 {
    // Normalize to -PI to PI range
    let mut x = x % 6.28318;
    if x > 3.14159 {
        x -= 6.28318;
    }
    if x < -3.14159 {
        x += 6.28318;
    }

    // Taylor series approximation
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    x - x3 / 6.0 + x5 / 120.0
}

/// Clamp value to range
fn clamp(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}
