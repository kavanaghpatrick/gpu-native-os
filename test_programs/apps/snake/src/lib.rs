//! Snake Game - GPU Edition
//!
//! Classic snake game running entirely on the GPU.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Grid configuration
const GRID_WIDTH: usize = 20;
const GRID_HEIGHT: usize = 20;
const CELL_SIZE: f32 = 30.0;
const MARGIN: f32 = 50.0;
const MAX_SNAKE_LEN: usize = 100;

// Colors
const COLOR_SNAKE: u32 = 0x4CAF50FF;  // Green
const COLOR_FOOD: u32 = 0xF44336FF;   // Red
const COLOR_BG: u32 = 0x212121FF;     // Dark gray

// Direction constants
const DIR_RIGHT: u8 = 0;
const DIR_DOWN: u8 = 1;
const DIR_LEFT: u8 = 2;
const DIR_UP: u8 = 3;

/// Snake state stored in fixed arrays
struct SnakeState {
    body_x: [u8; MAX_SNAKE_LEN],
    body_y: [u8; MAX_SNAKE_LEN],
    length: usize,
    direction: u8,
    food_x: u8,
    food_y: u8,
}

impl SnakeState {
    const fn new() -> Self {
        Self {
            body_x: [0; MAX_SNAKE_LEN],
            body_y: [0; MAX_SNAKE_LEN],
            length: 0,
            direction: DIR_RIGHT,
            food_x: 0,
            food_y: 0,
        }
    }
}

/// Initialize snake in the middle of the grid
fn init_snake(state: &mut SnakeState) {
    state.length = 3;
    state.direction = DIR_RIGHT;

    // Start in middle, horizontal
    let start_x = (GRID_WIDTH / 2) as u8;
    let start_y = (GRID_HEIGHT / 2) as u8;

    state.body_x[0] = start_x;
    state.body_y[0] = start_y;
    state.body_x[1] = start_x - 1;
    state.body_y[1] = start_y;
    state.body_x[2] = start_x - 2;
    state.body_y[2] = start_y;

    // Initial food position
    state.food_x = start_x + 3;
    state.food_y = start_y;
}

/// Simple pseudo-random number based on frame
fn pseudo_random(frame: i32, seed: i32) -> u8 {
    let val = ((frame.wrapping_mul(1103515245).wrapping_add(seed)) >> 16) & 0x7FFF;
    (val % 20) as u8
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };

    // Create initial snake state
    let mut state = SnakeState::new();
    init_snake(&mut state);

    // Update food position based on frame (demo animation)
    state.food_x = pseudo_random(current_frame / 30, 12345);
    state.food_y = pseudo_random(current_frame / 30, 54321);

    // Render background grid
    let mut y = 0usize;
    while y < GRID_HEIGHT {
        let mut x = 0usize;
        while x < GRID_WIDTH {
            let px = MARGIN + (x as f32) * CELL_SIZE;
            let py = MARGIN + (y as f32) * CELL_SIZE;

            // Check if this cell contains snake or food
            let mut is_snake = false;
            let mut i = 0usize;
            while i < state.length {
                if state.body_x[i] == x as u8 && state.body_y[i] == y as u8 {
                    is_snake = true;
                }
                i += 1;
            }

            let is_food = state.food_x == x as u8 && state.food_y == y as u8;

            let color = if is_snake {
                COLOR_SNAKE
            } else if is_food {
                COLOR_FOOD
            } else {
                COLOR_BG
            };

            unsafe {
                emit_quad(px, py, CELL_SIZE - 2.0, CELL_SIZE - 2.0, color);
            }

            x += 1;
        }
        y += 1;
    }

    // Return total cells rendered
    (GRID_WIDTH * GRID_HEIGHT) as i32
}
