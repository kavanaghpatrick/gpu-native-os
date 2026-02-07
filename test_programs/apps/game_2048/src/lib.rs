//! 2048 Puzzle Game - GPU Edition
//!
//! The classic sliding tile puzzle game.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Grid constants
const GRID_SIZE: usize = 4;
const CELL_SIZE: f32 = 100.0;
const CELL_GAP: f32 = 10.0;
const MARGIN: f32 = 100.0;

// Colors for different tile values
const COLORS: [u32; 12] = [
    0xCDC1B4FF, // Empty
    0xEEE4DAFF, // 2
    0xEDE0C8FF, // 4
    0xF2B179FF, // 8
    0xF59563FF, // 16
    0xF67C5FFF, // 32
    0xF65E3BFF, // 64
    0xEDCF72FF, // 128
    0xEDCC61FF, // 256
    0xEDC850FF, // 512
    0xEDC53FFF, // 1024
    0xEDC22EFF, // 2048
];

/// Get color for tile value
fn get_tile_color(value: u32) -> u32 {
    if value == 0 {
        COLORS[0]
    } else {
        // log2 of value gives index (2=1, 4=2, 8=3, etc)
        let idx = log2_approx(value);
        if idx < COLORS.len() as u32 {
            COLORS[idx as usize]
        } else {
            COLORS[11]  // Max color for very high values
        }
    }
}

/// Approximate log2 for powers of 2
fn log2_approx(mut n: u32) -> u32 {
    let mut log = 0u32;
    while n > 1 {
        n /= 2;
        log += 1;
    }
    log
}

/// Simple pseudo-random based on frame
fn pseudo_random(frame: i32, seed: i32) -> u32 {
    ((frame.wrapping_mul(1103515245).wrapping_add(seed)) >> 16) as u32 & 0x7FFF
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };

    // Create a demo board state based on frame
    // In real game, this would be persistent state
    let mut board: [[u32; GRID_SIZE]; GRID_SIZE] = [[0; GRID_SIZE]; GRID_SIZE];

    // Place some tiles based on frame number for animation
    let phase = (current_frame / 60) % 16;

    // Distribute some values across the board
    let positions = [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (0, 3), (3, 0), (1, 2), (2, 1),
    ];

    let mut i = 0usize;
    while i < 8 {
        let (x, y) = positions[i];
        let val = pseudo_random(current_frame / 30, i as i32) % 12;
        board[y][x] = if val < 8 { 1 << (val + 1) } else { 0 };
        i += 1;
    }

    let mut quad_count = 0;

    // Draw background
    unsafe {
        emit_quad(
            MARGIN - CELL_GAP,
            MARGIN - CELL_GAP,
            (CELL_SIZE + CELL_GAP) * GRID_SIZE as f32 + CELL_GAP,
            (CELL_SIZE + CELL_GAP) * GRID_SIZE as f32 + CELL_GAP,
            0xBBADA0FF,  // Board background
        );
    }
    quad_count += 1;

    // Draw tiles
    let mut y = 0usize;
    while y < GRID_SIZE {
        let mut x = 0usize;
        while x < GRID_SIZE {
            let px = MARGIN + (x as f32) * (CELL_SIZE + CELL_GAP);
            let py = MARGIN + (y as f32) * (CELL_SIZE + CELL_GAP);

            let value = board[y][x];
            let color = get_tile_color(value);

            // Draw rounded tile (approximate with single quad)
            unsafe {
                emit_quad(px, py, CELL_SIZE, CELL_SIZE, color);
            }
            quad_count += 1;

            x += 1;
        }
        y += 1;
    }

    quad_count
}
