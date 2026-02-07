//! Digital Clock - GPU Edition
//!
//! Displays a 7-segment style digital clock using quads.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Display constants
const DIGIT_WIDTH: f32 = 60.0;
const DIGIT_HEIGHT: f32 = 100.0;
const SEGMENT_THICKNESS: f32 = 12.0;
const DIGIT_SPACING: f32 = 20.0;
const START_X: f32 = 150.0;
const START_Y: f32 = 200.0;

// Colors
const COLOR_ON: u32 = 0x00FF00FF;   // Bright green
const COLOR_OFF: u32 = 0x003300FF;  // Dark green

// 7-segment patterns for digits 0-9
// Bits: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
const SEGMENT_PATTERNS: [u8; 10] = [
    0b1110111, // 0
    0b0100100, // 1
    0b1011101, // 2
    0b1101101, // 3
    0b0101110, // 4
    0b1101011, // 5
    0b1111011, // 6
    0b0100101, // 7
    0b1111111, // 8
    0b1101111, // 9
];

/// Draw a single 7-segment digit
fn draw_digit(x: f32, y: f32, digit: u8) -> i32 {
    let pattern = if digit < 10 {
        SEGMENT_PATTERNS[digit as usize]
    } else {
        0
    };

    let mut count = 0;

    // Segment positions (relative to digit origin)
    // Top horizontal
    let on = (pattern & 0b1000000) != 0;
    unsafe {
        emit_quad(x + SEGMENT_THICKNESS, y,
                  DIGIT_WIDTH - 2.0 * SEGMENT_THICKNESS, SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Top-right vertical
    let on = (pattern & 0b0100000) != 0;
    unsafe {
        emit_quad(x + DIGIT_WIDTH - SEGMENT_THICKNESS, y + SEGMENT_THICKNESS,
                  SEGMENT_THICKNESS, DIGIT_HEIGHT / 2.0 - SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Bottom-right vertical
    let on = (pattern & 0b0010000) != 0;
    unsafe {
        emit_quad(x + DIGIT_WIDTH - SEGMENT_THICKNESS, y + DIGIT_HEIGHT / 2.0,
                  SEGMENT_THICKNESS, DIGIT_HEIGHT / 2.0 - SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Bottom horizontal
    let on = (pattern & 0b0001000) != 0;
    unsafe {
        emit_quad(x + SEGMENT_THICKNESS, y + DIGIT_HEIGHT - SEGMENT_THICKNESS,
                  DIGIT_WIDTH - 2.0 * SEGMENT_THICKNESS, SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Bottom-left vertical
    let on = (pattern & 0b0000100) != 0;
    unsafe {
        emit_quad(x, y + DIGIT_HEIGHT / 2.0,
                  SEGMENT_THICKNESS, DIGIT_HEIGHT / 2.0 - SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Top-left vertical
    let on = (pattern & 0b0000010) != 0;
    unsafe {
        emit_quad(x, y + SEGMENT_THICKNESS,
                  SEGMENT_THICKNESS, DIGIT_HEIGHT / 2.0 - SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    // Middle horizontal
    let on = (pattern & 0b0000001) != 0;
    unsafe {
        emit_quad(x + SEGMENT_THICKNESS, y + DIGIT_HEIGHT / 2.0 - SEGMENT_THICKNESS / 2.0,
                  DIGIT_WIDTH - 2.0 * SEGMENT_THICKNESS, SEGMENT_THICKNESS,
                  if on { COLOR_ON } else { COLOR_OFF });
    }
    count += 1;

    count
}

/// Draw colon separator
fn draw_colon(x: f32, y: f32) -> i32 {
    unsafe {
        emit_quad(x, y + DIGIT_HEIGHT / 3.0, SEGMENT_THICKNESS, SEGMENT_THICKNESS, COLOR_ON);
        emit_quad(x, y + 2.0 * DIGIT_HEIGHT / 3.0, SEGMENT_THICKNESS, SEGMENT_THICKNESS, COLOR_ON);
    }
    2
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };

    // Convert frame to time (assume 60fps)
    let total_seconds = current_frame / 60;
    let hours = ((total_seconds / 3600) % 24) as u8;
    let minutes = ((total_seconds / 60) % 60) as u8;
    let seconds = (total_seconds % 60) as u8;

    let mut x = START_X;
    let mut quad_count = 0;

    // Hours
    quad_count += draw_digit(x, START_Y, hours / 10);
    x += DIGIT_WIDTH + DIGIT_SPACING;
    quad_count += draw_digit(x, START_Y, hours % 10);
    x += DIGIT_WIDTH + DIGIT_SPACING;

    // Colon (blinks every second)
    if seconds % 2 == 0 {
        quad_count += draw_colon(x, START_Y);
    }
    x += DIGIT_SPACING * 2.0;

    // Minutes
    quad_count += draw_digit(x, START_Y, minutes / 10);
    x += DIGIT_WIDTH + DIGIT_SPACING;
    quad_count += draw_digit(x, START_Y, minutes % 10);
    x += DIGIT_WIDTH + DIGIT_SPACING;

    // Colon
    if seconds % 2 == 0 {
        quad_count += draw_colon(x, START_Y);
    }
    x += DIGIT_SPACING * 2.0;

    // Seconds
    quad_count += draw_digit(x, START_Y, seconds / 10);
    x += DIGIT_WIDTH + DIGIT_SPACING;
    quad_count += draw_digit(x, START_Y, seconds % 10);

    quad_count
}
