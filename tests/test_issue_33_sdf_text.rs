// Tests for Issue #33: SDF Text Engine
//
// Run all tests:   cargo test --test test_issue_33_sdf_text
// Run Phase 1:     cargo test --test test_issue_33_sdf_text phase1
// Run Phase 2:     cargo test --test test_issue_33_sdf_text phase2
//
// Visual demo:     cargo run --release --example sdf_text_demo

use rust_experiment::gpu_os::sdf_text::*;

// ============================================================================
// Phase 1: Font Loading
// ============================================================================

mod phase1 {
    use super::*;

    #[test]
    fn test_ttf_parser_loads_system_font() {
        // Should be able to load at least one system font
        let result = SdfFont::load_default();
        assert!(result.is_ok(), "Should load a default system font");

        let font = result.unwrap();
        assert!(font.units_per_em() > 0, "Units per em should be positive");
    }

    #[test]
    fn test_font_metrics_valid() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // Basic metrics sanity checks
        assert!(font.units_per_em() >= 256, "Units per em should be at least 256");
        assert!(font.ascender() > 0, "Ascender should be positive");
        assert!(font.descender() < 0, "Descender should be negative");

        // Line height should be ascender - descender + gap
        let line_height = font.line_height();
        assert!(line_height > font.ascender(), "Line height should be greater than ascender");
    }

    #[test]
    fn test_glyph_outline_extraction() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // 'A' should have an outline
        let outline = font.glyph_outline('A');
        assert!(outline.is_some(), "Glyph 'A' should have an outline");

        let outline = outline.unwrap();
        assert!(!outline.is_empty(), "Outline should not be empty");
        assert!(!outline.segments.is_empty(), "Should have path segments");
        assert!(!outline.bounds.is_empty(), "Bounds should not be empty");
    }

    #[test]
    fn test_ascii_coverage() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // All printable ASCII (32-126) should have glyphs
        let mut missing = Vec::new();
        for codepoint in 32u8..=126 {
            let c = codepoint as char;
            if !font.has_glyph(c) {
                missing.push(c);
            }
        }

        assert!(
            missing.is_empty(),
            "Missing glyphs for ASCII characters: {:?}",
            missing
        );
    }

    #[test]
    fn test_glyph_advance_values() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // All ASCII should have positive advance
        for codepoint in 33u8..=126 {
            let c = codepoint as char;
            let advance = font.glyph_advance(c);
            assert!(
                advance.is_some() && advance.unwrap() > 0.0,
                "Glyph '{}' should have positive advance, got {:?}",
                c,
                advance
            );
        }

        // Space should have advance too
        let space_advance = font.glyph_advance(' ');
        assert!(
            space_advance.is_some() && space_advance.unwrap() > 0.0,
            "Space should have positive advance"
        );
    }

    #[test]
    fn test_path_segment_types() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // Check that we get actual path data
        let outline = font.glyph_outline('O').expect("'O' should have outline");

        let mut has_move = false;
        let mut has_close = false;
        let mut has_curve = false;

        for segment in &outline.segments {
            match segment {
                PathSegment::MoveTo(_, _) => has_move = true,
                PathSegment::Close => has_close = true,
                PathSegment::QuadTo(_, _, _, _) | PathSegment::CubicTo(_, _, _, _, _, _) => {
                    has_curve = true
                }
                _ => {}
            }
        }

        assert!(has_move, "Outline should have MoveTo");
        assert!(has_close, "Outline should have Close");
        // 'O' should have curves (it's round)
        assert!(has_curve, "'O' outline should have curves");
    }

    #[test]
    fn test_scale_to_pixels() {
        let font = SdfFont::load_default().expect("Need system font for test");

        let units = 1000.0;
        let font_size = 16.0;
        let pixels = font.scale_to_pixels(units, font_size);

        let expected = units * font_size / font.units_per_em() as f32;
        assert!(
            (pixels - expected).abs() < 0.001,
            "Scale calculation should match"
        );
    }

    #[test]
    fn test_cached_glyph_count() {
        let font = SdfFont::load_default().expect("Need system font for test");

        // Should have cached ASCII 32-126 = 95 characters
        assert!(
            font.cached_glyph_count() >= 95,
            "Should have at least 95 cached glyphs, got {}",
            font.cached_glyph_count()
        );
    }
}

// ============================================================================
// Phase 2: CPU SDF Generation
// ============================================================================

mod phase2 {
    use super::*;

    #[test]
    fn test_sdf_generator_creates_bitmap() {
        let generator = SdfGenerator::new(64, 4);
        let outline = GlyphOutline::default();
        let sdf = generator.generate(&outline);

        assert_eq!(sdf.width, 64);
        assert_eq!(sdf.height, 64);
        assert_eq!(sdf.data.len(), 64 * 64);
    }

    #[test]
    fn test_sdf_bitmap_to_u8_conversion() {
        let mut sdf = SdfBitmap::new(4, 4);
        sdf.set(0, 0, -1.0); // Inside
        sdf.set(1, 0, 0.0); // Edge
        sdf.set(2, 0, 1.0); // Outside

        let bytes = sdf.to_u8(1.0);

        // -1.0 should map to 0 (inside)
        assert_eq!(bytes[0], 0);
        // 0.0 should map to ~127 (edge)
        assert!((bytes[1] as i32 - 127).abs() <= 1);
        // 1.0 should map to 255 (outside)
        assert_eq!(bytes[2], 255);
    }

    // TODO Phase 2: Add more tests when SDF generation is implemented
    // - test_sdf_distance_at_edge_is_zero
    // - test_sdf_inside_is_negative
    // - test_sdf_outside_is_positive
    // - test_sdf_simple_square_path
}

// ============================================================================
// Phase 3: SDF Atlas Packing
// ============================================================================

mod phase3 {
    use super::*;
    use metal::Device;

    #[test]
    fn test_atlas_creation() {
        let device = Device::system_default().expect("No Metal device");
        let atlas = SdfAtlas::new(&device, 512, 512);

        assert_eq!(atlas.dimensions(), (512, 512));
        assert_eq!(atlas.glyph_count(), 0);
    }

    #[test]
    fn test_atlas_glyph_storage() {
        let device = Device::system_default().expect("No Metal device");
        let mut atlas = SdfAtlas::new(&device, 512, 512);

        let sdf = SdfBitmap::new(32, 32);
        atlas
            .add_glyph('A', &sdf, [10.0, 16.0], [0.0, 14.0], 10.0)
            .expect("Should add glyph");

        assert_eq!(atlas.glyph_count(), 1);
        assert!(atlas.glyph_info('A').is_some());
    }

    // TODO Phase 3: Add more tests when atlas packing is implemented
    // - test_atlas_packs_multiple_glyphs
    // - test_atlas_uvs_are_valid
    // - test_atlas_no_overlap
    // - test_atlas_full_ascii_fits
}

// ============================================================================
// Phase 4: GPU SDF Rendering
// ============================================================================

mod phase4 {
    use super::*;
    use metal::Device;

    #[test]
    fn test_renderer_creation() {
        let device = Device::system_default().expect("No Metal device");
        let result = SdfTextRenderer::new(&device, 1024);

        assert!(result.is_ok(), "Renderer should be created");
    }

    // TODO Phase 4: Add more tests when rendering is implemented
    // - test_sdf_render_creates_vertices
    // - test_sdf_render_different_sizes
    // - test_sdf_render_performance_10k_glyphs
}

// ============================================================================
// Phase 5: Text Shaping & Measurement
// ============================================================================

mod phase5 {
    use super::*;
    use metal::Device;

    #[test]
    fn test_text_measurement() {
        let device = Device::system_default().expect("No Metal device");
        let font = SdfFont::load_default().expect("Need system font");
        let renderer = SdfTextRenderer::new(&device, 1024).expect("Need renderer");

        let metrics = renderer.measure_text(&font, "Hello", 16.0);

        assert!(metrics.width > 0.0, "Text should have positive width");
        assert!(metrics.height > 0.0, "Text should have positive height");
        assert!(metrics.ascent > 0.0, "Should have positive ascent");
        assert!(metrics.descent < 0.0, "Should have negative descent");
    }

    #[test]
    fn test_text_layout_line() {
        let device = Device::system_default().expect("No Metal device");
        let font = SdfFont::load_default().expect("Need system font");
        let renderer = SdfTextRenderer::new(&device, 1024).expect("Need renderer");

        let instances = renderer.layout_line(
            &font,
            "Hello",
            0.0,
            100.0,
            16.0,
            [1.0, 1.0, 1.0, 1.0],
        );

        assert_eq!(instances.len(), 5, "Should have 5 glyph instances");

        // First glyph should be at origin
        assert_eq!(instances[0].x, 0.0);
        assert_eq!(instances[0].codepoint, 'H');

        // Subsequent glyphs should advance
        for i in 1..instances.len() {
            assert!(
                instances[i].x > instances[i - 1].x,
                "Glyphs should advance left to right"
            );
        }
    }

    #[test]
    fn test_empty_text_measurement() {
        let device = Device::system_default().expect("No Metal device");
        let font = SdfFont::load_default().expect("Need system font");
        let renderer = SdfTextRenderer::new(&device, 1024).expect("Need renderer");

        let metrics = renderer.measure_text(&font, "", 16.0);
        assert_eq!(metrics.width, 0.0, "Empty text should have zero width");

        let instances = renderer.layout_line(&font, "", 0.0, 0.0, 16.0, [1.0; 4]);
        assert!(instances.is_empty(), "Empty text should produce no instances");
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_font_to_measurement() {
    let font = SdfFont::load_default().expect("Should load default font");

    // Verify we can go from font -> metrics -> measurement
    let text = "The quick brown fox jumps over the lazy dog";

    let mut total_advance = 0.0;
    for c in text.chars() {
        if let Some(advance) = font.glyph_advance(c) {
            total_advance += advance;
        }
    }

    // At 16px, measure scaled advance
    let scale = 16.0 / font.units_per_em() as f32;
    let expected_width = total_advance * scale;

    // Use renderer for comparison
    let device = metal::Device::system_default().expect("No Metal device");
    let renderer = SdfTextRenderer::new(&device, 1024).expect("Need renderer");
    let metrics = renderer.measure_text(&font, text, 16.0);

    assert!(
        (metrics.width - expected_width).abs() < 0.1,
        "Measured width ({}) should match calculated ({})",
        metrics.width,
        expected_width
    );
}
