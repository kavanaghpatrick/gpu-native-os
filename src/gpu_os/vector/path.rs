//! Path data structures and PathBuilder API

use std::mem;

/// Path command type
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PathCommand {
    MoveTo = 0,
    LineTo = 1,
    QuadTo = 2,
    CubicTo = 3,
    Close = 4,
}

/// A single path segment with command and control points
/// NOTE: Metal float2 requires 8-byte alignment, so we need 7 bytes padding after cmd
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PathSegment {
    /// Command type (MoveTo, LineTo, etc.)
    pub cmd: u8,
    /// Padding for 8-byte alignment (Metal float2 requirement)
    pub _pad: [u8; 7],
    /// Control points (interpretation depends on cmd)
    /// - MoveTo/LineTo: p0 = target point
    /// - QuadTo: p0 = control, p1 = target
    /// - CubicTo: p0 = control1, p1 = control2, p2 = target
    /// - Close: no points used
    pub points: [[f32; 2]; 3],
}

impl PathSegment {
    pub fn move_to(x: f32, y: f32) -> Self {
        Self {
            cmd: PathCommand::MoveTo as u8,
            _pad: [0; 7],
            points: [[x, y], [0.0, 0.0], [0.0, 0.0]],
        }
    }

    pub fn line_to(x: f32, y: f32) -> Self {
        Self {
            cmd: PathCommand::LineTo as u8,
            _pad: [0; 7],
            points: [[x, y], [0.0, 0.0], [0.0, 0.0]],
        }
    }

    pub fn quad_to(cx: f32, cy: f32, x: f32, y: f32) -> Self {
        Self {
            cmd: PathCommand::QuadTo as u8,
            _pad: [0; 7],
            points: [[cx, cy], [x, y], [0.0, 0.0]],
        }
    }

    pub fn cubic_to(c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) -> Self {
        Self {
            cmd: PathCommand::CubicTo as u8,
            _pad: [0; 7],
            points: [[c1x, c1y], [c2x, c2y], [x, y]],
        }
    }

    pub fn close() -> Self {
        Self {
            cmd: PathCommand::Close as u8,
            _pad: [0; 7],
            points: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        }
    }

    pub fn command(&self) -> PathCommand {
        match self.cmd {
            0 => PathCommand::MoveTo,
            1 => PathCommand::LineTo,
            2 => PathCommand::QuadTo,
            3 => PathCommand::CubicTo,
            4 => PathCommand::Close,
            _ => PathCommand::MoveTo,
        }
    }
}

/// A complete path with bounds
#[derive(Clone, Debug)]
pub struct Path {
    pub segments: Vec<PathSegment>,
    pub bounds: [f32; 4], // [min_x, min_y, max_x, max_y]
}

impl Path {
    pub fn new(segments: Vec<PathSegment>) -> Self {
        let bounds = Self::compute_bounds(&segments);
        Self { segments, bounds }
    }

    fn compute_bounds(segments: &[PathSegment]) -> [f32; 4] {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for seg in segments {
            match seg.command() {
                PathCommand::MoveTo | PathCommand::LineTo => {
                    let p = seg.points[0];
                    min_x = min_x.min(p[0]);
                    min_y = min_y.min(p[1]);
                    max_x = max_x.max(p[0]);
                    max_y = max_y.max(p[1]);
                }
                PathCommand::QuadTo => {
                    // Include control point and endpoint
                    for i in 0..2 {
                        let p = seg.points[i];
                        min_x = min_x.min(p[0]);
                        min_y = min_y.min(p[1]);
                        max_x = max_x.max(p[0]);
                        max_y = max_y.max(p[1]);
                    }
                }
                PathCommand::CubicTo => {
                    // Include all control points and endpoint
                    for i in 0..3 {
                        let p = seg.points[i];
                        min_x = min_x.min(p[0]);
                        min_y = min_y.min(p[1]);
                        max_x = max_x.max(p[0]);
                        max_y = max_y.max(p[1]);
                    }
                }
                PathCommand::Close => {}
            }
        }

        if min_x == f32::MAX {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            [min_x, min_y, max_x, max_y]
        }
    }

    /// Get byte slice for GPU upload
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.segments.as_ptr() as *const u8,
                self.segments.len() * mem::size_of::<PathSegment>(),
            )
        }
    }

    /// Number of segments
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Builder for constructing paths fluently
#[derive(Clone, Debug, Default)]
pub struct PathBuilder {
    segments: Vec<PathSegment>,
    current_pos: [f32; 2],
    start_pos: [f32; 2],
}

impl PathBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move to a new position (starts a new subpath)
    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.segments.push(PathSegment::move_to(x, y));
        self.current_pos = [x, y];
        self.start_pos = [x, y];
        self
    }

    /// Draw a line to a position
    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        // Implicit move_to if no current position
        if self.segments.is_empty() {
            self.move_to(x, y);
            return self;
        }
        self.segments.push(PathSegment::line_to(x, y));
        self.current_pos = [x, y];
        self
    }

    /// Draw a quadratic bezier curve
    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> &mut Self {
        if self.segments.is_empty() {
            self.move_to(cx, cy);
        }
        self.segments.push(PathSegment::quad_to(cx, cy, x, y));
        self.current_pos = [x, y];
        self
    }

    /// Draw a cubic bezier curve
    pub fn cubic_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) -> &mut Self {
        if self.segments.is_empty() {
            self.move_to(c1x, c1y);
        }
        self.segments.push(PathSegment::cubic_to(c1x, c1y, c2x, c2y, x, y));
        self.current_pos = [x, y];
        self
    }

    /// Close the current subpath (line back to start)
    pub fn close(&mut self) -> &mut Self {
        if !self.segments.is_empty() {
            self.segments.push(PathSegment::close());
            self.current_pos = self.start_pos;
        }
        self
    }

    /// Add a rectangle
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32) -> &mut Self {
        self.move_to(x, y)
            .line_to(x + w, y)
            .line_to(x + w, y + h)
            .line_to(x, y + h)
            .close()
    }

    /// Add a rounded rectangle
    pub fn rounded_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32) -> &mut Self {
        let r = r.min(w / 2.0).min(h / 2.0);
        // Start at top-left, after corner radius
        self.move_to(x + r, y)
            // Top edge
            .line_to(x + w - r, y)
            // Top-right corner (quadratic approximation)
            .quad_to(x + w, y, x + w, y + r)
            // Right edge
            .line_to(x + w, y + h - r)
            // Bottom-right corner
            .quad_to(x + w, y + h, x + w - r, y + h)
            // Bottom edge
            .line_to(x + r, y + h)
            // Bottom-left corner
            .quad_to(x, y + h, x, y + h - r)
            // Left edge
            .line_to(x, y + r)
            // Top-left corner
            .quad_to(x, y, x + r, y)
            .close()
    }

    /// Add a circle (approximated with 4 cubic beziers)
    pub fn circle(&mut self, cx: f32, cy: f32, r: f32) -> &mut Self {
        // Magic number for cubic bezier circle approximation
        let k = 0.5522847498;
        let kr = k * r;

        self.move_to(cx + r, cy)
            .cubic_to(cx + r, cy + kr, cx + kr, cy + r, cx, cy + r)
            .cubic_to(cx - kr, cy + r, cx - r, cy + kr, cx - r, cy)
            .cubic_to(cx - r, cy - kr, cx - kr, cy - r, cx, cy - r)
            .cubic_to(cx + kr, cy - r, cx + r, cy - kr, cx + r, cy)
            .close()
    }

    /// Add an ellipse
    pub fn ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32) -> &mut Self {
        let k = 0.5522847498;
        let kx = k * rx;
        let ky = k * ry;

        self.move_to(cx + rx, cy)
            .cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry)
            .cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy)
            .cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry)
            .cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy)
            .close()
    }

    /// Build the final path
    pub fn build(&self) -> Path {
        Path::new(self.segments.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_segment_size() {
        // PathSegment should be 32 bytes: 1 + 7 padding + 24 (3 * 2 * 4)
        // Metal float2 requires 8-byte alignment
        assert_eq!(mem::size_of::<PathSegment>(), 32);
    }

    #[test]
    fn test_path_builder() {
        let mut builder = PathBuilder::new();
        let path = builder
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .close()
            .build();

        assert_eq!(path.segments.len(), 4);
        assert_eq!(path.bounds, [0.0, 0.0, 100.0, 100.0]);
    }

    #[test]
    fn test_rounded_rect() {
        let mut builder = PathBuilder::new();
        let path = builder.rounded_rect(10.0, 10.0, 100.0, 50.0, 5.0).build();

        // Should have: move + 4 lines + 4 quads + close = 10 segments
        assert_eq!(path.segments.len(), 10);
    }

    #[test]
    fn test_circle() {
        let mut builder = PathBuilder::new();
        let path = builder.circle(50.0, 50.0, 25.0).build();

        // Should have: move + 4 cubics + close = 6 segments
        assert_eq!(path.segments.len(), 6);
    }
}
