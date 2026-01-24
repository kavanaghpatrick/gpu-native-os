//! SDF Text Rendering Diagnostic Tool
//!
//! Renders text to an offscreen texture and saves it as a PPM image for analysis.
//! Run with: cargo run --release --example sdf_render_test

use metal::*;
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer};
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 400;

fn main() {
    println!("=== SDF Text Rendering Diagnostic ===\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("GPU: {}", device.name());

    // Create bitmap font and text renderer
    let font = BitmapFont::new(&device);
    let mut text_renderer = TextRenderer::new(&device, 5000)
        .expect("Failed to create text renderer");

    // Create command queue
    let command_queue = device.new_command_queue();

    // Create offscreen render target
    let texture_desc = TextureDescriptor::new();
    texture_desc.set_width(WIDTH as u64);
    texture_desc.set_height(HEIGHT as u64);
    texture_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    texture_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    texture_desc.set_storage_mode(MTLStorageMode::Managed);
    let render_texture = device.new_texture(&texture_desc);

    println!("Render target: {}x{}", WIDTH, HEIGHT);

    // Clear the text renderer
    text_renderer.clear();

    // Color helper: RGBA to u32
    fn rgba(r: f32, g: f32, b: f32, a: f32) -> u32 {
        let r = (r * 255.0) as u32;
        let g = (g * 255.0) as u32;
        let b = (b * 255.0) as u32;
        let a = (a * 255.0) as u32;
        (r << 24) | (g << 16) | (b << 8) | a
    }

    let white = rgba(1.0, 1.0, 1.0, 1.0);
    let gray = rgba(0.7, 0.7, 0.7, 1.0);
    let yellow = rgba(1.0, 0.8, 0.2, 1.0);
    let green = rgba(0.2, 1.0, 0.5, 1.0);
    let red = rgba(1.0, 0.3, 0.3, 1.0);

    // Add test text at various sizes
    text_renderer.add_text_sized("GPU Filesystem Browser - Path|", 20.0, 30.0, white, 14.0);
    text_renderer.add_text_sized("Search: Type to search...", 20.0, 60.0, gray, 14.0);
    text_renderer.add_text_sized("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 20.0, 100.0, white, 16.0);
    text_renderer.add_text_sized("abcdefghijklmnopqrstuvwxyz", 20.0, 130.0, white, 16.0);
    text_renderer.add_text_sized("0123456789 !@#$%^&*()", 20.0, 160.0, white, 16.0);
    text_renderer.add_text_sized("Large Text Sample", 20.0, 210.0, yellow, 24.0);
    text_renderer.add_text_sized("Even Larger", 20.0, 260.0, green, 32.0);
    // Very large text for clear SDF quality check
    text_renderer.add_text_sized("O", 700.0, 320.0, white, 128.0);  // Single large 'O' for quality check
    text_renderer.add_text_sized("BIG", 20.0, 330.0, red, 48.0);

    println!("Added {} characters", text_renderer.char_count());

    // Create render pass
    let render_desc = RenderPassDescriptor::new();
    let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
    color_attachment.set_texture(Some(&render_texture));
    color_attachment.set_load_action(MTLLoadAction::Clear);
    color_attachment.set_store_action(MTLStoreAction::Store);
    color_attachment.set_clear_color(MTLClearColor::new(0.15, 0.1, 0.1, 1.0)); // Dark red background

    // Render
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_render_command_encoder(&render_desc);
    text_renderer.render(&encoder, &font, WIDTH as f32, HEIGHT as f32);
    encoder.end_encoding();

    // Synchronize managed texture
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.synchronize_resource(&render_texture);
    blit_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!("Rendering complete");

    // Read back pixels
    let bytes_per_row = WIDTH * 4;
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];

    render_texture.get_bytes(
        pixels.as_mut_ptr() as *mut _,
        bytes_per_row as u64,
        MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: WIDTH as u64, height: HEIGHT as u64, depth: 1 },
        },
        0,
    );

    // Save as PPM (simple format)
    let output_path = "sdf_render_test.ppm";
    let mut file = File::create(output_path).expect("Failed to create output file");

    // PPM header
    writeln!(file, "P6").unwrap();
    writeln!(file, "{} {}", WIDTH, HEIGHT).unwrap();
    writeln!(file, "255").unwrap();

    // Convert BGRA to RGB
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = ((y * WIDTH + x) * 4) as usize;
            let b = pixels[idx];
            let g = pixels[idx + 1];
            let r = pixels[idx + 2];
            file.write_all(&[r, g, b]).unwrap();
        }
    }

    println!("Saved to: {}", output_path);
    println!("\nTo view: open {} (or use any image viewer)", output_path);

    // Analyze the image
    analyze_render(&pixels, WIDTH, HEIGHT);
}

fn analyze_render(pixels: &[u8], width: u32, height: u32) {
    println!("\n=== Render Analysis ===");

    let mut non_background = 0u64;
    let mut total_brightness = 0u64;
    let mut white_pixels = 0u64;
    let mut dark_pixels = 0u64;

    // Background color is approximately (38, 25, 25) in RGB = 0.15, 0.1, 0.1
    let bg_r = 38u8;
    let bg_g = 25u8;
    let bg_b = 25u8;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let b = pixels[idx];
            let g = pixels[idx + 1];
            let r = pixels[idx + 2];

            let brightness = (r as u64 + g as u64 + b as u64) / 3;
            total_brightness += brightness;

            // Check if pixel is significantly different from background
            let diff_r = (r as i32 - bg_r as i32).abs();
            let diff_g = (g as i32 - bg_g as i32).abs();
            let diff_b = (b as i32 - bg_b as i32).abs();

            if diff_r > 20 || diff_g > 20 || diff_b > 20 {
                non_background += 1;
            }

            if r > 200 && g > 200 && b > 200 {
                white_pixels += 1;
            }

            if brightness < 50 {
                dark_pixels += 1;
            }
        }
    }

    let total_pixels = (width * height) as u64;
    let avg_brightness = total_brightness / total_pixels;

    println!("Total pixels: {}", total_pixels);
    println!("Non-background pixels: {} ({:.2}%)", non_background, non_background as f64 / total_pixels as f64 * 100.0);
    println!("White pixels (>200): {} ({:.2}%)", white_pixels, white_pixels as f64 / total_pixels as f64 * 100.0);
    println!("Dark pixels (<50): {} ({:.2}%)", dark_pixels, dark_pixels as f64 / total_pixels as f64 * 100.0);
    println!("Average brightness: {}", avg_brightness);

    // Sample specific regions where text should be
    println!("\n=== Text Region Samples ===");
    sample_region(pixels, width, 20, 20, 100, 40, "First line region");
    sample_region(pixels, width, 20, 90, 200, 50, "Uppercase region");
    sample_region(pixels, width, 20, 200, 200, 70, "Large text region");
    sample_region(pixels, width, 20, 300, 150, 80, "BIG text (48px)");
    sample_region(pixels, width, 680, 180, 120, 150, "Large O (128px)");

    // Analyze anti-aliasing quality (check for smooth gradients)
    println!("\n=== Anti-aliasing Analysis ===");
    analyze_aa_quality(pixels, width, height);
}

fn analyze_aa_quality(pixels: &[u8], width: u32, height: u32) {
    // Count pixels in brightness buckets
    let bg_brightness = 30;  // approximate background brightness
    let mut buckets = [0u64; 10]; // 0-25, 25-50, ..., 225-255

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let b = pixels[idx];
            let g = pixels[idx + 1];
            let r = pixels[idx + 2];
            let brightness = ((r as u32 + g as u32 + b as u32) / 3) as u8;

            // Skip near-background pixels
            if brightness > bg_brightness + 10 {
                let bucket = (brightness as usize / 25).min(9);
                buckets[bucket] += 1;
            }
        }
    }

    // Show distribution
    println!("Brightness distribution (excluding background):");
    let labels = ["  30-50", " 50-75", " 75-100", "100-125", "125-150",
                  "150-175", "175-200", "200-225", "225-255", "   255+"];
    let total: u64 = buckets.iter().sum();

    for (i, count) in buckets.iter().enumerate() {
        if *count > 0 {
            let pct = *count as f64 / total as f64 * 100.0;
            let bar_len = (pct / 2.0) as usize;
            let bar: String = "█".repeat(bar_len.min(30));
            println!("{}: {:5} ({:5.1}%) {}", labels[i], count, pct, bar);
        }
    }

    // Check for anti-aliasing (mid-range brightness = edge pixels)
    let edge_pixels: u64 = buckets[2..8].iter().sum();  // 75-200 brightness range
    let core_pixels: u64 = buckets[8..].iter().sum();   // 200-255 brightness (text cores)

    if total > 0 {
        let edge_ratio = edge_pixels as f64 / total as f64 * 100.0;
        let core_ratio = core_pixels as f64 / total as f64 * 100.0;
        println!("\nAnti-aliasing quality:");
        println!("  Edge pixels (AA): {:.1}%", edge_ratio);
        println!("  Core pixels: {:.1}%", core_ratio);

        if edge_ratio < 5.0 {
            println!("  ⚠ LOW AA - text may appear jagged");
        } else if edge_ratio > 40.0 {
            println!("  ⚠ EXCESSIVE AA - text may appear blurry");
        } else {
            println!("  ✓ Good anti-aliasing balance");
        }
    }
}

fn sample_region(pixels: &[u8], width: u32, x: u32, y: u32, w: u32, h: u32, name: &str) {
    let mut count = 0u64;
    let mut total_r = 0u64;
    let mut total_g = 0u64;
    let mut total_b = 0u64;
    let mut max_r = 0u8;
    let mut max_g = 0u8;
    let mut max_b = 0u8;
    let mut bright_pixels = 0u64;  // pixels brighter than background

    // Background color for comparison
    let bg_brightness = (38 + 25 + 25) / 3;

    for py in y..(y + h).min(400) {
        for px in x..(x + w).min(800) {
            let idx = ((py * width + px) * 4) as usize;
            let b = pixels[idx];
            let g = pixels[idx + 1];
            let r = pixels[idx + 2];

            total_r += r as u64;
            total_g += g as u64;
            total_b += b as u64;
            count += 1;

            max_r = max_r.max(r);
            max_g = max_g.max(g);
            max_b = max_b.max(b);

            let brightness = (r as u32 + g as u32 + b as u32) / 3;
            if brightness > bg_brightness + 20 {
                bright_pixels += 1;
            }
        }
    }

    if count > 0 {
        println!("{}: avg RGB = ({}, {}, {}), max RGB = ({}, {}, {}), bright pixels: {} ({:.1}%)",
            name,
            total_r / count,
            total_g / count,
            total_b / count,
            max_r, max_g, max_b,
            bright_pixels,
            bright_pixels as f64 / count as f64 * 100.0
        );
    }
}
