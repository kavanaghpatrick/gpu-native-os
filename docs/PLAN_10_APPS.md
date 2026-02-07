# Plan: Launch 10 Real Rust Applications on GPU

## Goal
Take 10 REAL Rust applications from the internet and LAUNCH them on the GPU - meaning they render visuals and respond to input.

## The 10 Applications

| # | Name | Source | Type | Complexity |
|---|------|--------|------|------------|
| 1 | Game of Life | rust-wasm_game_of_life | Cellular automaton | Low |
| 2 | CHIP-8 VM | RSC8 | Virtual machine | Medium |
| 3 | Tetris | no_std_tetris | Game | Medium |
| 4 | Snake | rust-snake-wasm | Game | Low |
| 5 | Browser Shooter | tsoding/rust-browser-game | Game | Medium |
| 6 | Mandelbrot Viewer | Custom (interactive) | Fractal | Low |
| 7 | Pong | Custom based on patterns | Game | Low |
| 8 | 2048 | Custom based on patterns | Puzzle | Medium |
| 9 | Raymarcher | SDF-based | Graphics | Medium |
| 10 | Clock/Timer | Custom | Utility | Low |

## Architecture for GPU Apps

```
┌─────────────────────────────────────────────────────────────────┐
│                     Rust Application                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Game Logic  │  │   State     │  │   Render Commands       │  │
│  │  (pure fn)  │→ │ (GPU buffer)│→ │ (QUAD opcodes → verts)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        ↑                                        ↓
   FrameState                           unified_vertices
   (cursor, time)                       (48-byte RenderVertex)
```

## Key Constraint: WASM → GPU Bytecode

Our apps must:
1. Compile to WASM (wasm32-unknown-unknown target)
2. Translate via WasmTranslator to GPU bytecode
3. Execute in GPU megakernel

This means:
- NO println!, NO file I/O
- NO Vec (unless we implement allocator)
- Input via reading from state buffer
- Output via EMIT_QUAD instruction (or equivalent)

## Implementation Strategy

### Phase 1: Build GPU App Framework

Create a minimal framework that apps can use:
```rust
// gpu_app_framework.rs
pub struct GpuApp {
    frame: u32,
    cursor_x: f32,
    cursor_y: f32,
    mouse_down: bool,
}

impl GpuApp {
    // Read from state buffer offset 0
    pub fn frame(&self) -> u32;
    pub fn cursor(&self) -> (f32, f32);
    pub fn mouse_down(&self) -> bool;

    // Write quad to vertex output
    pub fn draw_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32);
}
```

### Phase 2: Port Each Application

For each app:
1. Download source code
2. Extract core game logic (pure functions)
3. Create wrapper that uses GpuApp framework
4. Compile to WASM
5. Translate to bytecode
6. Test on GPU

### Phase 3: Visual Verification

Create test harness that:
1. Launches app on GPU
2. Captures frame output
3. Displays in window
4. Handles input (keyboard/mouse)

## Detailed App Plans

### App 1: Game of Life
- Grid: 64x64 cells
- State: 64x64 bytes (one per cell)
- Render: 64x64 quads (different colors for alive/dead)
- Input: Click to toggle cells
- Source: Rosetta Code / rust-wasm_game_of_life

### App 2: CHIP-8 VM
- Memory: 4KB RAM in state buffer
- Display: 64x32 pixels
- Input: 16-key hex keypad mapped to keyboard
- ROMs: Pong, Tetris, Space Invaders
- Source: https://github.com/jerryshell/rsc8

### App 3: Tetris
- Grid: 10x20 cells
- State: Current piece, board, score
- Render: Grid + current piece + next piece preview
- Input: Arrow keys for movement, up for rotate
- Source: https://github.com/Hahihula/no_std_tetris

### App 4: Snake
- Grid: 32x32 cells
- State: Snake body (ring buffer), food position, direction
- Render: Snake body + food + grid
- Input: Arrow keys
- Source: https://github.com/yiransheng/rust-snake-wasm

### App 5: Browser Shooter
- State: Player position, bullets, enemies
- Render: Sprites as colored quads
- Input: Mouse aim + click to shoot
- Source: https://github.com/tsoding/rust-browser-game

### App 6: Mandelbrot Viewer
- State: Center x/y, zoom level
- Render: Compute Mandelbrot per-pixel, color based on iterations
- Input: Click to zoom, drag to pan
- Source: Existing example + interactivity

### App 7: Pong
- State: Ball position/velocity, paddle positions, scores
- Render: Ball + paddles + score display
- Input: W/S for left paddle, Up/Down for right paddle
- Source: Custom implementation

### App 8: 2048
- Grid: 4x4 tiles
- State: Tile values, score
- Render: Colored tiles with numbers
- Input: Arrow keys to slide
- Source: Custom implementation based on algorithm

### App 9: Raymarcher
- State: Camera position/rotation
- Render: SDF raymarching per-pixel
- Input: WASD to move, mouse to look
- Source: Custom implementation

### App 10: Clock/Timer
- State: Current time (from frame count)
- Render: Digital clock display using quads
- Input: Click to start/stop timer mode
- Source: Custom implementation

## Success Criteria

An app is "launched" when:
- [ ] It renders visuals to the screen
- [ ] It responds to user input
- [ ] It maintains state between frames
- [ ] The core logic comes from real Rust code (not hardcoded)

## Files to Create

```
test_programs/apps/
├── framework/
│   ├── gpu_app.rs        # Base framework
│   └── lib.rs            # Exports
├── game_of_life/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs        # Entry point
│       └── game.rs       # Game logic (from internet)
├── chip8/
│   └── ...
├── tetris/
│   └── ...
└── ... (8 more)

tests/test_gpu_apps.rs    # Visual test harness
```

## Execution Order

1. **First**: Build framework + Game of Life (simplest)
2. **Second**: Snake + Pong (simple games)
3. **Third**: Tetris + 2048 (grid puzzles)
4. **Fourth**: CHIP-8 (VM - more complex)
5. **Fifth**: Mandelbrot + Raymarcher (graphics)
6. **Sixth**: Browser Shooter + Clock (misc)

## Current Blockers

1. **No EMIT_QUAD in WASM**: Need to call GPU-specific function from WASM
   - Solution: Use memory writes to a "command buffer" that GPU interprets

2. **No input reading**: Need to read FrameState from WASM
   - Solution: Map state buffer offsets to input values

3. **Fixed-point rendering**: WASM uses i32/f32, need to emit proper vertices
   - Solution: Build rendering abstraction in the framework
