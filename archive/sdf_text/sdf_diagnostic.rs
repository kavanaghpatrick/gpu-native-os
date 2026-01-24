use rust_experiment::gpu_os::sdf_text::{SdfFont, PathSegment};

fn main() {
    println!("=== SDF Font Diagnostic ===\n");
    
    let font = SdfFont::load_default().expect("Failed to load font");
    
    println!("Font metrics:");
    println!("  Units per EM: {}", font.units_per_em());
    println!("  Ascender: {}", font.ascender());
    println!("  Descender: {}", font.descender());
    println!("  Line height: {}", font.line_height());
    println!("  Cached glyphs: {}", font.cached_glyph_count());
    
    // Check a few sample glyphs
    for c in ['A', 'a', 'O', 'o', 'M', 'l', '!', '.'] {
        println!("\n--- Glyph '{}' ---", c);
        
        if let Some(metrics) = font.glyph_metrics(c) {
            println!("  Advance: {:.1}", metrics.advance_width);
            println!("  LSB: {:.1}", metrics.left_side_bearing);
            println!("  Bounds: x[{:.1}, {:.1}] y[{:.1}, {:.1}]",
                metrics.bounds.x_min, metrics.bounds.x_max,
                metrics.bounds.y_min, metrics.bounds.y_max);
            println!("  Bounds size: {:.1} x {:.1}", 
                metrics.bounds.width(), metrics.bounds.height());
        }
        
        if let Some(outline) = font.glyph_outline(c) {
            println!("  Segments: {}", outline.segments.len());
            println!("  Outline bounds: x[{:.1}, {:.1}] y[{:.1}, {:.1}]",
                outline.bounds.x_min, outline.bounds.x_max,
                outline.bounds.y_min, outline.bounds.y_max);
            
            // Count segment types
            let mut moves = 0;
            let mut lines = 0;
            let mut quads = 0;
            let mut cubics = 0;
            let mut closes = 0;
            
            for seg in &outline.segments {
                match seg {
                    PathSegment::MoveTo(_, _) => moves += 1,
                    PathSegment::LineTo(_, _) => lines += 1,
                    PathSegment::QuadTo(_, _, _, _) => quads += 1,
                    PathSegment::CubicTo(_, _, _, _, _, _) => cubics += 1,
                    PathSegment::Close => closes += 1,
                }
            }
            
            println!("  Segment breakdown: {} move, {} line, {} quad, {} cubic, {} close",
                moves, lines, quads, cubics, closes);
            
            // Show first few segments
            if outline.segments.len() > 0 {
                println!("  First segments:");
                for (i, seg) in outline.segments.iter().take(5).enumerate() {
                    match seg {
                        PathSegment::MoveTo(x, y) => 
                            println!("    [{}] MoveTo({:.1}, {:.1})", i, x, y),
                        PathSegment::LineTo(x, y) => 
                            println!("    [{}] LineTo({:.1}, {:.1})", i, x, y),
                        PathSegment::QuadTo(cx, cy, x, y) => 
                            println!("    [{}] QuadTo(c:{:.1},{:.1} e:{:.1},{:.1})", i, cx, cy, x, y),
                        PathSegment::CubicTo(c1x, c1y, c2x, c2y, x, y) => 
                            println!("    [{}] CubicTo(c1:{:.1},{:.1} c2:{:.1},{:.1} e:{:.1},{:.1})", 
                                i, c1x, c1y, c2x, c2y, x, y),
                        PathSegment::Close => 
                            println!("    [{}] Close", i),
                    }
                }
            }
        } else {
            println!("  (no outline)");
        }
    }
    
    // Check atlas data constants
    println!("\n=== Atlas Data Analysis ===");
    use rust_experiment::gpu_os::sdf_text::atlas_data::*;
    println!("Atlas size: {}x{}", ATLAS_WIDTH, ATLAS_HEIGHT);
    println!("SDF cell size: {}", SDF_SIZE);
    println!("Atlas columns: {}", ATLAS_COLS);
    println!("Units per EM: {}", UNITS_PER_EM);
    println!("Ascender: {}", ASCENDER);
    println!("Descender: {}", DESCENDER);
    
    println!("\n=== Sample Glyph Metrics from Atlas ===");
    for (i, c) in [' ', '!', 'A', 'a', 'O', 'M'].iter().enumerate() {
        let idx = (*c as u32 - 32) as usize;
        if idx < GLYPH_METRICS.len() {
            let m = &GLYPH_METRICS[idx];
            println!("'{}' (idx {}): advance={:.1}, bounds=[{:.1},{:.1},{:.1},{:.1}], atlas=({},{})",
                c, idx, m.advance, 
                m.bounds[0], m.bounds[1], m.bounds[2], m.bounds[3],
                m.atlas_x, m.atlas_y);
        }
    }
    
    // Check for potential issues
    println!("\n=== Potential Issues ===");
    
    // 1. Check if bounds look reasonable compared to font metrics
    let a_idx = ('A' as u32 - 32) as usize;
    let a_metric = &GLYPH_METRICS[a_idx];
    let expected_height = ASCENDER; // Capital should be close to ascender
    let actual_height = a_metric.bounds[3] - a_metric.bounds[1];
    println!("'A' height: {:.1} (expected ~{:.1})", actual_height, expected_height);
    
    // 2. Check coordinate system
    println!("Coordinate check:");
    println!("  'A' y_min={:.1}, y_max={:.1}", a_metric.bounds[1], a_metric.bounds[3]);
    if a_metric.bounds[1] >= 0.0 && a_metric.bounds[3] > 0.0 {
        println!("  Y-axis appears to be positive-up (baseline at y=0)");
    } else {
        println!("  WARNING: Y-axis might be inverted");
    }
    
    // 3. Check if atlas positions are correct
    let expected_cols = ATLAS_COLS as usize;
    let mut atlas_errors = 0;
    for (i, m) in GLYPH_METRICS.iter().enumerate() {
        let expected_x = (i % expected_cols) as u32 * (SDF_SIZE + PADDING);
        let expected_y = (i / expected_cols) as u32 * (SDF_SIZE + PADDING);
        if m.atlas_x != expected_x || m.atlas_y != expected_y {
            if atlas_errors < 3 {
                println!("  Atlas position mismatch at idx {}: got ({},{}), expected ({},{})",
                    i, m.atlas_x, m.atlas_y, expected_x, expected_y);
            }
            atlas_errors += 1;
        }
    }
    if atlas_errors == 0 {
        println!("  All atlas positions look correct");
    } else {
        println!("  Total atlas position errors: {}", atlas_errors);
    }
}
