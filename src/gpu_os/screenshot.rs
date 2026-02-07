//! GPU Screenshot & Visual Testing Infrastructure
//!
//! Provides screenshot capture from GPU textures and visual testing utilities
//! for validating GPU rendering output.
//!
//! # Example
//! ```ignore
//! let screenshot = GpuScreenshot::new(&device);
//! screenshot.capture(&texture);
//! screenshot.save_ppm("output.ppm")?;
//!
//! // Visual testing
//! let baseline = VisualBaseline::load("baseline.ppm")?;
//! let diff = screenshot.diff(&baseline);
//! assert!(diff.max_difference < 5);
//! ```

use metal::*;
use std::fs::File;
use std::io::{Write, BufWriter, Read};
use std::path::Path;

/// Screenshot capture from GPU textures
pub struct GpuScreenshot {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,  // RGBA, 4 bytes per pixel
}

impl GpuScreenshot {
    /// Create an empty screenshot
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0u8; (width * height * 4) as usize],
        }
    }

    /// Capture pixels from a Metal texture
    ///
    /// The texture must be in Shared or Managed storage mode for CPU access.
    /// For Private storage textures, use capture_with_blit() instead.
    pub fn capture(&mut self, texture: &Texture) {
        let w = texture.width() as u32;
        let h = texture.height() as u32;

        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.pixels = vec![0u8; (w * h * 4) as usize];
        }

        let bytes_per_row = w * 4;
        let region = MTLRegion::new_2d(0, 0, w as u64, h as u64);

        texture.get_bytes(
            self.pixels.as_mut_ptr() as *mut _,
            bytes_per_row as u64,
            region,
            0,
        );
    }

    /// Capture from a texture using a blit command encoder
    ///
    /// This works for textures in any storage mode by first synchronizing
    /// to a staging buffer.
    pub fn capture_with_blit(
        &mut self,
        device: &Device,
        command_queue: &CommandQueue,
        texture: &Texture,
    ) {
        let w = texture.width() as u32;
        let h = texture.height() as u32;
        let bytes_per_row = w * 4;
        let buffer_size = (bytes_per_row * h) as u64;

        // Resize if needed
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.pixels = vec![0u8; buffer_size as usize];
        }

        // Create staging buffer
        let staging = device.new_buffer(
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Blit texture to staging buffer
        let cmd = command_queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();

        blit.copy_from_texture_to_buffer(
            texture,
            0,  // slice
            0,  // mip level
            MTLOrigin::default(),
            MTLSize::new(w as u64, h as u64, 1),
            &staging,
            0,  // destination offset
            bytes_per_row as u64,
            bytes_per_row as u64 * h as u64,
            MTLBlitOption::empty(),
        );

        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Copy from staging buffer
        let ptr = staging.contents() as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, self.pixels.as_mut_ptr(), buffer_size as usize);
        }
    }

    /// Save screenshot as PPM file (simple format, no external dependencies)
    pub fn save_ppm<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // PPM header
        writeln!(writer, "P6")?;
        writeln!(writer, "{} {}", self.width, self.height)?;
        writeln!(writer, "255")?;

        // Write RGB data (PPM doesn't support alpha)
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = ((y * self.width + x) * 4) as usize;
                writer.write_all(&[
                    self.pixels[idx],     // R
                    self.pixels[idx + 1], // G
                    self.pixels[idx + 2], // B
                ])?;
            }
        }

        Ok(())
    }

    /// Save screenshot as raw RGBA file
    pub fn save_raw<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        // Header: width (4 bytes) + height (4 bytes)
        file.write_all(&self.width.to_le_bytes())?;
        file.write_all(&self.height.to_le_bytes())?;
        // Pixel data
        file.write_all(&self.pixels)?;
        Ok(())
    }

    /// Load screenshot from raw RGBA file
    pub fn load_raw<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        let width = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let height = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        let mut pixels = vec![0u8; (width * height * 4) as usize];
        file.read_exact(&mut pixels)?;

        Ok(Self { width, height, pixels })
    }

    /// Get pixel at (x, y) as RGBA tuple
    pub fn get_pixel(&self, x: u32, y: u32) -> (u8, u8, u8, u8) {
        let idx = ((y * self.width + x) * 4) as usize;
        (
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        )
    }

    /// Set pixel at (x, y)
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
        let idx = ((y * self.width + x) * 4) as usize;
        self.pixels[idx] = r;
        self.pixels[idx + 1] = g;
        self.pixels[idx + 2] = b;
        self.pixels[idx + 3] = a;
    }
}

/// Result of comparing two screenshots
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Maximum per-channel difference found (0-255)
    pub max_difference: u8,
    /// Average per-channel difference across all pixels
    pub avg_difference: f64,
    /// Number of pixels that differ by more than the threshold
    pub pixels_different: u32,
    /// Total number of pixels compared
    pub total_pixels: u32,
    /// Structural similarity index (0.0-1.0, higher is more similar)
    pub similarity: f64,
}

impl DiffResult {
    /// Check if images are identical (zero difference)
    pub fn is_identical(&self) -> bool {
        self.max_difference == 0
    }

    /// Check if images are similar within a threshold
    pub fn is_similar(&self, threshold: u8) -> bool {
        self.max_difference <= threshold
    }

    /// Check if less than a percentage of pixels differ significantly
    pub fn within_tolerance(&self, threshold: u8, max_percent_different: f64) -> bool {
        let percent = (self.pixels_different as f64 / self.total_pixels as f64) * 100.0;
        self.max_difference <= threshold || percent <= max_percent_different
    }
}

/// Visual test baseline for comparison
pub struct VisualBaseline {
    pub screenshot: GpuScreenshot,
    pub name: String,
}

impl VisualBaseline {
    /// Create a new baseline from a screenshot
    pub fn new(name: &str, screenshot: GpuScreenshot) -> Self {
        Self {
            screenshot,
            name: name.to_string(),
        }
    }

    /// Load baseline from file
    pub fn load<P: AsRef<Path>>(name: &str, path: P) -> std::io::Result<Self> {
        let screenshot = GpuScreenshot::load_raw(path)?;
        Ok(Self {
            screenshot,
            name: name.to_string(),
        })
    }

    /// Save baseline to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.screenshot.save_raw(path)
    }

    /// Compare another screenshot against this baseline
    pub fn compare(&self, other: &GpuScreenshot, threshold: u8) -> DiffResult {
        diff_screenshots(&self.screenshot, other, threshold)
    }
}

/// Compare two screenshots and return difference metrics
pub fn diff_screenshots(a: &GpuScreenshot, b: &GpuScreenshot, threshold: u8) -> DiffResult {
    // Handle size mismatch
    if a.width != b.width || a.height != b.height {
        return DiffResult {
            max_difference: 255,
            avg_difference: 255.0,
            pixels_different: a.width * a.height,
            total_pixels: a.width * a.height,
            similarity: 0.0,
        };
    }

    let total = a.width * a.height;
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    let mut pixels_diff = 0u32;

    for i in 0..a.pixels.len() {
        let diff = (a.pixels[i] as i16 - b.pixels[i] as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
        sum_diff += diff as u64;

        // Count pixels with any channel exceeding threshold
        if i % 4 == 0 && diff > threshold {
            pixels_diff += 1;
        }
    }

    let avg_diff = sum_diff as f64 / (total * 4) as f64;
    let similarity = 1.0 - (avg_diff / 255.0);

    DiffResult {
        max_difference: max_diff,
        avg_difference: avg_diff,
        pixels_different: pixels_diff,
        total_pixels: total,
        similarity,
    }
}

/// Create a visual diff image highlighting differences
pub fn create_diff_image(a: &GpuScreenshot, b: &GpuScreenshot, scale: u8) -> GpuScreenshot {
    let w = a.width.max(b.width);
    let h = a.height.max(b.height);
    let mut result = GpuScreenshot::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let pa = if x < a.width && y < a.height {
                a.get_pixel(x, y)
            } else {
                (0, 0, 0, 255)
            };
            let pb = if x < b.width && y < b.height {
                b.get_pixel(x, y)
            } else {
                (0, 0, 0, 255)
            };

            // Calculate per-channel difference, scaled for visibility
            let dr = ((pa.0 as i16 - pb.0 as i16).unsigned_abs() as u16 * scale as u16).min(255) as u8;
            let dg = ((pa.1 as i16 - pb.1 as i16).unsigned_abs() as u16 * scale as u16).min(255) as u8;
            let db = ((pa.2 as i16 - pb.2 as i16).unsigned_abs() as u16 * scale as u16).min(255) as u8;

            result.set_pixel(x, y, dr, dg, db, 255);
        }
    }

    result
}

/// Visual test framework for automated visual testing
pub struct VisualTestRunner {
    baseline_dir: String,
    output_dir: String,
    threshold: u8,
}

impl VisualTestRunner {
    /// Create a new visual test runner
    pub fn new(baseline_dir: &str, output_dir: &str) -> Self {
        Self {
            baseline_dir: baseline_dir.to_string(),
            output_dir: output_dir.to_string(),
            threshold: 5,  // Default threshold
        }
    }

    /// Set the comparison threshold (0-255)
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }

    /// Run a visual test
    pub fn test(&self, name: &str, screenshot: &GpuScreenshot) -> VisualTestResult {
        let baseline_path = format!("{}/{}.raw", self.baseline_dir, name);

        // Try to load baseline
        match VisualBaseline::load(name, &baseline_path) {
            Ok(baseline) => {
                let diff = baseline.compare(screenshot, self.threshold);
                let passed = diff.is_similar(self.threshold);
                let max_diff = diff.max_difference;

                // Save actual output for comparison
                let actual_path = format!("{}/{}_actual.ppm", self.output_dir, name);
                let _ = screenshot.save_ppm(&actual_path);

                // If different, save diff image
                if !passed {
                    let diff_img = create_diff_image(&baseline.screenshot, screenshot, 4);
                    let diff_path = format!("{}/{}_diff.ppm", self.output_dir, name);
                    let _ = diff_img.save_ppm(&diff_path);
                }

                VisualTestResult {
                    name: name.to_string(),
                    passed,
                    diff: Some(diff),
                    message: if passed {
                        "Passed: images match".to_string()
                    } else {
                        format!("Failed: max difference {} exceeds threshold {}",
                            max_diff, self.threshold)
                    },
                }
            }
            Err(_) => {
                // No baseline exists - create one
                let _ = std::fs::create_dir_all(&self.baseline_dir);
                let _ = screenshot.save_raw(&baseline_path);
                let ppm_path = format!("{}/{}.ppm", self.baseline_dir, name);
                let _ = screenshot.save_ppm(&ppm_path);

                VisualTestResult {
                    name: name.to_string(),
                    passed: true,  // First run always passes
                    diff: None,
                    message: format!("Baseline created: {}", baseline_path),
                }
            }
        }
    }

    /// Update a baseline with a new screenshot
    pub fn update_baseline(&self, name: &str, screenshot: &GpuScreenshot) -> std::io::Result<()> {
        let _ = std::fs::create_dir_all(&self.baseline_dir);
        let baseline_path = format!("{}/{}.raw", self.baseline_dir, name);
        screenshot.save_raw(&baseline_path)?;
        let ppm_path = format!("{}/{}.ppm", self.baseline_dir, name);
        screenshot.save_ppm(&ppm_path)?;
        Ok(())
    }
}

/// Result of a visual test
#[derive(Debug)]
pub struct VisualTestResult {
    pub name: String,
    pub passed: bool,
    pub diff: Option<DiffResult>,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screenshot_basic() {
        let mut screenshot = GpuScreenshot::new(100, 100);
        screenshot.set_pixel(50, 50, 255, 0, 0, 255);
        let (r, g, b, a) = screenshot.get_pixel(50, 50);
        assert_eq!((r, g, b, a), (255, 0, 0, 255));
    }

    #[test]
    fn test_diff_identical() {
        let a = GpuScreenshot::new(10, 10);
        let b = GpuScreenshot::new(10, 10);
        let diff = diff_screenshots(&a, &b, 0);
        assert!(diff.is_identical());
        assert_eq!(diff.similarity, 1.0);
    }

    #[test]
    fn test_diff_different() {
        let mut a = GpuScreenshot::new(10, 10);
        let mut b = GpuScreenshot::new(10, 10);

        // Make one pixel very different
        a.set_pixel(5, 5, 255, 255, 255, 255);
        b.set_pixel(5, 5, 0, 0, 0, 255);

        let diff = diff_screenshots(&a, &b, 0);
        assert!(!diff.is_identical());
        assert_eq!(diff.max_difference, 255);
    }

    #[test]
    fn test_diff_threshold() {
        let mut a = GpuScreenshot::new(10, 10);
        let mut b = GpuScreenshot::new(10, 10);

        // Small difference
        a.set_pixel(5, 5, 100, 100, 100, 255);
        b.set_pixel(5, 5, 103, 103, 103, 255);

        let diff = diff_screenshots(&a, &b, 5);
        assert!(diff.is_similar(5));  // Within threshold
        assert!(!diff.is_similar(2)); // Outside threshold
    }
}
