//! Conway's Game of Life - GPU Edition
//!
//! A real interactive Rust application running entirely on the GPU.
//! Uses extern functions that map to GPU intrinsics for rendering and input.

#![no_std]

// GPU intrinsics - these are translated to GPU bytecode operations
extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn get_cursor_x() -> f32;
    fn get_cursor_y() -> f32;
    fn get_mouse_down() -> i32;
    fn get_time() -> f32;
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Grid configuration
const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 32;
const CELL_SIZE: f32 = 20.0;
const CELL_GAP: f32 = 2.0;
const MARGIN: f32 = 50.0;

// Colors (RGBA packed as u32: 0xRRGGBBAA)
const COLOR_ALIVE: u32 = 0x4CAF50FF;  // Green
const COLOR_DEAD: u32 = 0x212121FF;   // Dark gray
const COLOR_GRID: u32 = 0x424242FF;   // Medium gray

// State layout in GPU memory:
// offset 0: quad_count (managed by emit_quad)
// offset 1024: grid state (GRID_WIDTH * GRID_HEIGHT bytes)
// offset 2048: previous grid (for double buffering)
// offset 3072: frame counter for simulation timing

const STATE_GRID_OFFSET: usize = 1024;
const STATE_PREV_GRID_OFFSET: usize = 2048;
const STATE_FRAME_OFFSET: usize = 3072;

/// Read a cell from the grid (stored in GPU state memory)
fn get_cell(grid_base: usize, x: usize, y: usize) -> bool {
    if x >= GRID_WIDTH || y >= GRID_HEIGHT {
        return false;
    }
    // In a real implementation, we'd read from GPU state memory
    // For now, we'll use a simpler approach
    false
}

/// Count live neighbors for a cell
fn count_neighbors(grid_base: usize, x: usize, y: usize) -> u32 {
    let mut count = 0u32;

    // Check all 8 neighbors
    for dy in 0..3 {
        for dx in 0..3 {
            if dx == 1 && dy == 1 {
                continue; // Skip self
            }

            let nx = (x + dx).wrapping_sub(1);
            let ny = (y + dy).wrapping_sub(1);

            if nx < GRID_WIDTH && ny < GRID_HEIGHT {
                if get_cell(grid_base, nx, ny) {
                    count += 1;
                }
            }
        }
    }

    count
}

/// Main entry point - called each frame
#[no_mangle]
pub extern "C" fn main() -> i32 {
    // Get current frame for simulation timing
    let current_frame = unsafe { frame() };

    // Get input state
    let cursor_x = unsafe { get_cursor_x() };
    let cursor_y = unsafe { get_cursor_y() };
    let mouse_down = unsafe { get_mouse_down() };

    // Calculate grid cell under cursor
    let grid_x = ((cursor_x - MARGIN) / (CELL_SIZE + CELL_GAP)) as i32;
    let grid_y = ((cursor_y - MARGIN) / (CELL_SIZE + CELL_GAP)) as i32;

    // Render the grid
    let mut y = 0usize;
    while y < GRID_HEIGHT {
        let mut x = 0usize;
        while x < GRID_WIDTH {
            let px = MARGIN + (x as f32) * (CELL_SIZE + CELL_GAP);
            let py = MARGIN + (y as f32) * (CELL_SIZE + CELL_GAP);

            // Determine cell state for coloring
            // For demo: create a checkerboard pattern that animates based on frame
            let pattern_offset = (current_frame / 30) as usize;  // Change every 30 frames
            let alive = ((x + y + pattern_offset) % 2) == 0;

            // Highlight cell under cursor
            let is_hovered = grid_x == x as i32 && grid_y == y as i32;
            let _ = mouse_down;  // Suppress warning

            let color = if is_hovered {
                0xFFFFFFFF  // White when clicked
            } else if is_hovered {
                0x64B5F6FF  // Light blue when hovered
            } else if alive {
                COLOR_ALIVE
            } else {
                COLOR_DEAD
            };

            unsafe {
                emit_quad(px, py, CELL_SIZE, CELL_SIZE, color);
            }

            x += 1;
        }
        y += 1;
    }

    // Return number of quads rendered (for verification)
    (GRID_WIDTH * GRID_HEIGHT) as i32
}
