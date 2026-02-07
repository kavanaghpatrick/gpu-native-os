//! Particle System - GPU Edition
//!
//! Simple particle simulation with physics.

#![no_std]

extern "C" {
    fn emit_quad(x: f32, y: f32, w: f32, h: f32, color: u32);
    fn frame() -> i32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Simulation constants
const NUM_PARTICLES: usize = 50;
const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;
const PARTICLE_SIZE: f32 = 8.0;
const GRAVITY: f32 = 0.2;

/// Particle state
#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
}

/// Simple pseudo-random number generator
fn rand(seed: u32) -> f32 {
    let n = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((n >> 16) & 0x7FFF) as f32 / 32768.0
}

/// Initialize a particle
fn init_particle(p: &mut Particle, seed: u32) {
    p.x = SCREEN_WIDTH / 2.0 + (rand(seed) - 0.5) * 100.0;
    p.y = SCREEN_HEIGHT - 50.0;
    p.vx = (rand(seed.wrapping_add(1)) - 0.5) * 10.0;
    p.vy = -rand(seed.wrapping_add(2)) * 15.0 - 5.0;
    p.life = 1.0;
}

/// Update particle physics
fn update_particle(p: &mut Particle, dt: f32) {
    p.vy += GRAVITY;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.life -= 0.01;

    // Bounce off walls
    if p.x < 0.0 {
        p.x = 0.0;
        p.vx = -p.vx * 0.8;
    }
    if p.x > SCREEN_WIDTH - PARTICLE_SIZE {
        p.x = SCREEN_WIDTH - PARTICLE_SIZE;
        p.vx = -p.vx * 0.8;
    }
    if p.y > SCREEN_HEIGHT - PARTICLE_SIZE {
        p.y = SCREEN_HEIGHT - PARTICLE_SIZE;
        p.vy = -p.vy * 0.8;
    }
}

/// Main entry point
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let current_frame = unsafe { frame() };
    let time = current_frame as f32 / 60.0;

    let mut particles: [Particle; NUM_PARTICLES] = [Particle {
        x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, life: 0.0
    }; NUM_PARTICLES];

    // Initialize particles with deterministic positions based on frame
    let mut i = 0usize;
    while i < NUM_PARTICLES {
        let seed = (current_frame as u32).wrapping_mul(i as u32 + 1);
        init_particle(&mut particles[i], seed);

        // Simulate physics based on particle age
        let age = (i as f32) * 0.1;
        let mut t = 0.0f32;
        while t < age {
            update_particle(&mut particles[i], 1.0);
            t += 1.0;
        }

        i += 1;
    }

    let mut quad_count = 0;

    // Render particles
    let mut i = 0usize;
    while i < NUM_PARTICLES {
        let p = &particles[i];

        if p.life > 0.0 {
            // Color fades with life
            let alpha = (p.life * 255.0) as u32;
            let r = ((1.0 - p.life) * 255.0) as u32;
            let g = (p.life * 200.0) as u32;
            let b = 50u32;
            let color = (r << 24) | (g << 16) | (b << 8) | alpha;

            unsafe {
                emit_quad(p.x, p.y, PARTICLE_SIZE, PARTICLE_SIZE, color);
            }
            quad_count += 1;
        }

        i += 1;
    }

    quad_count
}
