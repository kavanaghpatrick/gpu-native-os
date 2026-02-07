//! Bouncing Balls - GPU Edition
//! Simple animated balls using frame counter (no complex math)

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;  // Get current frame number for animation
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

const NUM_BALLS: usize = 5;
const BALL_SIZE: f32 = 40.0;
const BG_COLOR: u32 = 0x1A1A2EFF;

// Ball colors
const COLORS: [u32; 5] = [
    0xFF0000FF,  // Red
    0x00FF00FF,  // Green
    0x0000FFFF,  // Blue
    0xFFFF00FF,  // Yellow
    0xFF00FFFF,  // Magenta
];

/// Main entry point - called every frame
#[no_mangle]
pub extern "C" fn main() -> i32 {
    // Get current frame for animation
    let current_frame = unsafe { frame() };

    // Draw background
    unsafe { emit_quad(0.0, 0.0, 800.0, 600.0, BG_COLOR); }

    // X positions for each ball (spread across screen)
    let x_positions: [f32; NUM_BALLS] = [80.0, 200.0, 340.0, 480.0, 620.0];

    // Simple bounce: each ball moves at different speed
    // Y oscillates between 100 and 400 based on frame counter
    let mut i = 0usize;
    while i < NUM_BALLS {
        let x = x_positions[i];

        // Each ball bounces with different phase
        // Use simple modulo-based oscillation
        let phase = (i as i32) * 20;  // Different offset per ball
        let cycle = ((current_frame + phase) % 120) as f32;  // 120 frame cycle

        // Linear bounce: goes down then up
        let y = if cycle < 60.0 {
            100.0 + cycle * 4.0  // Going down: 100 -> 340
        } else {
            340.0 - (cycle - 60.0) * 4.0  // Going up: 340 -> 100
        };

        // Draw the ball with its color
        let color = COLORS[i];
        unsafe { emit_quad(x, y, BALL_SIZE, BALL_SIZE, color); }

        i += 1;
    }

    (NUM_BALLS + 1) as i32  // Return total quads (bg + balls)
}
