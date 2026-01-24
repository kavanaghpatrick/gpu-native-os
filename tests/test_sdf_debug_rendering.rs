// Test: Debug SDF Rendering - Isolate geometry from atlas issues
//
// This test helps diagnose SDF text rendering problems by testing
// different fragment shader configurations:
//
// 1. SOLID_COLOR: Shows quads in solid color - tests geometry pipeline
// 2. UV_DEBUG: Visualizes UV coordinates as RGB - tests UV mapping
// 3. POSITION_DEBUG: Shows screen position as colors - tests transforms
// 4. RAW_SDF: Current debug mode - shows SDF grayscale
// 5. THRESHOLD_DEBUG: Shows SDF threshold bands
//
// To use: Modify the fragment shader in text_render.rs with these snippets

/// Fragment shader variants for debugging
pub mod debug_shaders {
    /// Test 1: Solid color quads (ignores atlas entirely)
    /// If this works but SDF doesn't, the problem is in the atlas
    pub const SOLID_COLOR: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    // Bypass atlas entirely - render solid colored quads
    return in.color;
}
"#;

    /// Test 2: Visualize UV coordinates as RGB
    /// Red = U (horizontal), Green = V (vertical), Blue = 0.5
    /// Good UVs should show smooth gradients across each glyph
    pub const UV_DEBUG: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    // UV coordinates as colors
    // U maps to Red (0-1 across glyph width)
    // V maps to Green (0-1 across glyph height)
    return float4(in.uv.x, in.uv.y, 0.5, 1.0);
}
"#;

    /// Test 3: Visualize screen position
    /// Shows where quads are in screen space
    pub const POSITION_DEBUG: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    // Use fragment position (in pixels after rasterization)
    // Normalize to 0-1 assuming ~1000 pixel screen
    float2 norm_pos = in.position.xy / 1000.0;
    return float4(fract(norm_pos.x * 10.0), fract(norm_pos.y * 10.0), 0.5, 1.0);
}
"#;

    /// Test 4: Raw SDF visualization (current debug mode)
    /// White = inside glyph (SDF > 0.5)
    /// Black = outside glyph (SDF < 0.5)
    /// Gray 50% = edge (SDF = 0.5)
    pub const RAW_SDF: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::linear, min_filter::linear);
    float d = atlas.sample(samp, in.uv).r;
    return float4(d, d, d, 1.0);
}
"#;

    /// Test 5: SDF threshold bands - shows discrete bands
    /// Helps identify if SDF values are in expected range
    pub const THRESHOLD_BANDS: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::linear, min_filter::linear);
    float d = atlas.sample(samp, in.uv).r;

    // Color bands to show SDF value distribution
    if (d < 0.3) return float4(1.0, 0.0, 0.0, 1.0);       // Red: far outside
    if (d < 0.4) return float4(1.0, 0.5, 0.0, 1.0);       // Orange: outside
    if (d < 0.5) return float4(1.0, 1.0, 0.0, 1.0);       // Yellow: near edge outside
    if (d < 0.6) return float4(0.0, 1.0, 0.0, 1.0);       // Green: near edge inside
    if (d < 0.7) return float4(0.0, 1.0, 1.0, 1.0);       // Cyan: inside
    return float4(0.0, 0.0, 1.0, 1.0);                     // Blue: far inside
}
"#;

    /// Test 6: Checkerboard pattern - tests UV precision
    /// Should show clean checkerboard if UVs are correct
    pub const CHECKERBOARD: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    // 8x8 checkerboard based on UV
    float2 uv_scaled = in.uv * 8.0;
    bool checker = (int(floor(uv_scaled.x)) + int(floor(uv_scaled.y))) % 2 == 0;
    float c = checker ? 1.0 : 0.3;
    return float4(in.color.rgb * c, 1.0);
}
"#;

    /// Test 7: Atlas bounds check - shows if UVs are sampling valid region
    pub const ATLAS_BOUNDS: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    // Check if UVs are in expected range for atlas
    // Each glyph is 48x48 in a 500x500 atlas
    // Valid U range per glyph: atlas_x/500 to (atlas_x+48)/500

    // Show UV bounds issues
    if (in.uv.x < 0.0 || in.uv.x > 1.0) return float4(1.0, 0.0, 1.0, 1.0); // Magenta: U out of bounds
    if (in.uv.y < 0.0 || in.uv.y > 1.0) return float4(1.0, 0.0, 1.0, 1.0); // Magenta: V out of bounds

    // Sample and show with color overlay indicating position in atlas
    constexpr sampler samp(mag_filter::linear, min_filter::linear);
    float d = atlas.sample(samp, in.uv).r;

    // Tint by atlas position
    return float4(d * 0.5 + in.uv.x * 0.5, d, d * 0.5 + in.uv.y * 0.5, 1.0);
}
"#;

    /// Production shader with proper SDF rendering
    pub const PRODUCTION: &str = r#"
fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::linear, min_filter::linear);
    float d = atlas.sample(samp, in.uv).r;

    // SDF threshold with anti-aliasing
    float edge = 0.5;
    float aa = fwidth(d) * 0.75;
    float alpha = smoothstep(edge - aa, edge + aa, d);

    if (alpha < 0.01) discard_fragment();

    return float4(in.color.rgb, in.color.a * alpha);
}
"#;
}

/// Diagnostic tests to run
pub mod diagnostics {
    /// What each test result indicates:
    ///
    /// 1. SOLID_COLOR test:
    ///    - If quads appear in correct positions: Geometry pipeline is working
    ///    - If nothing appears: Problem is in vertex shader or compute layout
    ///    - If quads are wrong size/position: Check compute kernel calculations
    ///
    /// 2. UV_DEBUG test:
    ///    - If gradients look correct (red left-to-right, green bottom-to-top): UVs correct
    ///    - If colors are uniform: UVs not varying across quad
    ///    - If colors are inverted: UV orientation is flipped
    ///
    /// 3. RAW_SDF test:
    ///    - If glyphs visible as grayscale shapes: Atlas is correct
    ///    - If all black/white: Atlas data is corrupt or all 0/255
    ///    - If gray noise: Atlas data not properly loaded
    ///    - If wrong characters: UV/atlas index mismatch
    ///
    /// 4. THRESHOLD_BANDS test:
    ///    - Mostly yellow/green: Good edge definition
    ///    - All red: SDF values too low (atlas issue)
    ///    - All blue: SDF values too high (atlas issue)
    ///    - Mixed colors on single glyph: Good SDF gradient

    pub fn print_diagnostic_guide() {
        println!("=== SDF Text Rendering Diagnostic Guide ===\n");

        println!("Step 1: Test SOLID_COLOR shader");
        println!("  Expected: Colored rectangles at text positions");
        println!("  If FAIL: Check vertex positions in compute kernel");
        println!();

        println!("Step 2: Test UV_DEBUG shader");
        println!("  Expected: Red-green gradient on each glyph quad");
        println!("  If FAIL: Check UV calculation in compute kernel");
        println!();

        println!("Step 3: Test RAW_SDF shader");
        println!("  Expected: Grayscale glyph shapes");
        println!("  If FAIL: Check atlas texture loading and UV mapping");
        println!();

        println!("Step 4: Test THRESHOLD_BANDS shader");
        println!("  Expected: Multi-colored bands showing SDF gradient");
        println!("  If FAIL: SDF values not in expected 0-1 range");
        println!();

        println!("Step 5: Test PRODUCTION shader");
        println!("  Expected: Clean, anti-aliased text");
        println!("  If FAIL: Check smoothstep parameters and alpha blending");
    }
}

/// Vertex buffer inspection
pub mod vertex_inspection {
    /// Checks to perform on vertex buffer output
    ///
    /// After compute kernel runs, verify:
    /// 1. Positions are non-zero for non-space characters
    /// 2. Positions are within screen bounds (0 to screen_width/height)
    /// 3. UVs are in valid atlas range
    /// 4. Colors match input segment colors
    /// 5. 6 vertices per character form valid quads

    pub fn expected_vertex_properties() {
        println!("=== Vertex Buffer Validation ===\n");

        println!("For each character, 6 vertices should form 2 triangles:");
        println!("  Triangle 1: v[0], v[1], v[2] (bottom-left, top-left, top-right)");
        println!("  Triangle 2: v[3], v[4], v[5] (bottom-left, top-right, bottom-right)");
        println!();

        println!("Position checks:");
        println!("  - x values should be >= segment.x");
        println!("  - y values should be near segment.y (baseline)");
        println!("  - Width = glyph_width * scale");
        println!("  - Height = glyph_height * scale");
        println!();

        println!("UV checks:");
        println!("  - u0 = atlas_x / 500");
        println!("  - v0 = atlas_y / 500");
        println!("  - u1 = (atlas_x + 48) / 500");
        println!("  - v1 = (atlas_y + 48) / 500");
        println!();

        println!("For 'A' (glyph index 33 = 'A' - 32):");
        println!("  - Check GLYPH_METRICS[33] for expected bounds");
        println!("  - Verify atlas_x, atlas_y point to correct glyph in atlas");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_diagnostic_info() {
        diagnostics::print_diagnostic_guide();
        println!("\n\n");
        vertex_inspection::expected_vertex_properties();
    }

    #[test]
    fn shader_snippets_are_valid() {
        // Just verify the shader strings are non-empty
        assert!(!debug_shaders::SOLID_COLOR.is_empty());
        assert!(!debug_shaders::UV_DEBUG.is_empty());
        assert!(!debug_shaders::POSITION_DEBUG.is_empty());
        assert!(!debug_shaders::RAW_SDF.is_empty());
        assert!(!debug_shaders::THRESHOLD_BANDS.is_empty());
        assert!(!debug_shaders::CHECKERBOARD.is_empty());
        assert!(!debug_shaders::ATLAS_BOUNDS.is_empty());
        assert!(!debug_shaders::PRODUCTION.is_empty());
    }
}
