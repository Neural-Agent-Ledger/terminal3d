use super::*;
use re_core::oklab::{linear_to_oklab, oklab_chroma, unpack_linear};
use re_core::SurfaceKind;

fn floor_cell(hit: Vec2, fg: u32, bg: u32) -> Cell {
    Cell {
        glyph: u32::from('#'),
        fg,
        bg,
        hit,
        surface: SurfaceKind::Floor,
        ..Default::default()
    }
}

fn hue_deg(packed: u32) -> f32 {
    oklab_to_oklch(linear_to_oklab(unpack_linear(packed)))
        .h
        .to_degrees()
}

fn lightness(packed: u32) -> f32 {
    linear_to_oklab(unpack_linear(packed)).l
}

#[test]
fn empty_sources_is_a_true_noop() {
    let sources = DriftSources::default();
    let mut fb = Framebuffer::new(2, 2);
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(3.0, 3.0), 0x0000_FF00, 0x0010_1010),
    );
    let before = fb.clone();
    apply_drift(&mut fb, &sources);
    for row in 0..2 {
        for col in 0..2 {
            assert_eq!(fb.get(col, row), before.get(col, row));
        }
    }
}

#[test]
fn a_world_source_shifts_hue_near_its_centre_and_fades_with_distance() {
    let src = DriftSource {
        world_pos: Vec2::new(5.0, 5.0),
        radius: 4.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 18.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };

    let mut fb = Framebuffer::new(2, 1);
    // Cell 0: right at the source centre. Cell 1: far outside the radius.
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(5.0, 5.0), 0x0040_C0C0, 0x0010_1010),
    );
    fb.set(
        1,
        0,
        floor_cell(Vec2::new(50.0, 50.0), 0x0040_C0C0, 0x0010_1010),
    );

    let before_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    apply_drift(&mut fb, &sources);
    let near_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let far_fg = fb.get(1, 0).unwrap().fg;

    let moved = wrap_hue_delta((near_hue - before_hue).to_radians()).to_degrees();
    assert!(
        moved.abs() > 5.0,
        "a cell at the source centre must shift hue noticeably (Δ={moved}°)"
    );
    assert_eq!(
        far_fg, 0x0040_C0C0,
        "a cell far outside the radius must be untouched"
    );
}

#[test]
fn semantic_cells_get_zero_hue_shift_and_a_tiny_lightness_cap() {
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0, // huge, so weight ~1 everywhere in this tiny test grid
        delta_l: 0.5,  // deliberately way over the semantic cap
        delta_c: 0.2,
        delta_h_deg: 40.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };

    let mut fb = Framebuffer::new(1, 1);
    let mut c = floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010);
    c.material = 7; // is_semantic_default: material != 0
    fb.set(0, 0, c);

    let before_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let before_l = lightness(fb.get(0, 0).unwrap().fg);
    apply_drift(&mut fb, &sources);
    let after_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let after_l = lightness(fb.get(0, 0).unwrap().fg);

    let hue_moved = wrap_hue_delta((after_hue - before_hue).to_radians()).to_degrees();
    assert!(
        hue_moved.abs() < 1.0,
        "R1: semantic surfaces must get Δh≈0, got {hue_moved}°"
    );
    assert!(
        (after_l - before_l).abs() <= SEMANTIC_MAX_DELTA_L + 1e-3,
        "R1: semantic surfaces must cap |ΔL| at {SEMANTIC_MAX_DELTA_L}, got {}",
        after_l - before_l
    );
}

#[test]
fn non_semantic_cells_get_the_full_authored_hue_shift_within_the_near_cap() {
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 40.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010),
    ); // material == 0

    let before_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    apply_drift(&mut fb, &sources);
    let after_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let moved = wrap_hue_delta((after_hue - before_hue).to_radians()).to_degrees();
    assert!(
        moved.abs() > 20.0,
        "a non-semantic cell must receive most of the authored 40° shift, got {moved}°"
    );
}

#[test]
fn r2_caps_a_source_that_authors_an_extreme_hue_delta() {
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 500.0, // absurd — must be clamped, not wrapped-and-trusted
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010),
    );

    let before_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    apply_drift(&mut fb, &sources);
    let after_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let moved = wrap_hue_delta((after_hue - before_hue).to_radians())
        .to_degrees()
        .abs();
    assert!(
        moved <= MAX_SOURCE_DELTA_H_NEAR_DEG + 2.0,
        "a single source's hue delta must be capped near {MAX_SOURCE_DELTA_H_NEAR_DEG}°, got {moved}°"
    );
}

#[test]
fn gaze_shifts_the_screen_centre_cell_but_not_the_corner() {
    let gaze = GazeSource {
        screen_center: (0.5, 0.5),
        radius: 0.15,
        delta_l: 0.0,
        delta_c: -0.06,
        delta_h_deg: 0.0,
    };
    let sources = DriftSources {
        world: vec![],
        gaze: Some(gaze),
    };

    let mut fb = Framebuffer::new(5, 5);
    for row in 0..5 {
        for col in 0..5 {
            fb.set(
                col,
                row,
                floor_cell(Vec2::new(col as f32, row as f32), 0x0080_2040, 0x0010_1010),
            );
        }
    }
    let before_chroma = oklab_chroma(linear_to_oklab(unpack_linear(fb.get(2, 2).unwrap().fg)));
    apply_drift(&mut fb, &sources);
    let center_chroma = oklab_chroma(linear_to_oklab(unpack_linear(fb.get(2, 2).unwrap().fg)));
    let corner = fb.get(0, 0).unwrap().fg;

    assert!(
        center_chroma < before_chroma,
        "the screen-centre cell must desaturate under gaze"
    );
    assert_eq!(
        corner, 0x0080_2040,
        "a screen corner (far from centre) must be untouched by gaze"
    );
}

#[test]
fn contrast_floor_is_restored_on_a_near_collapsing_pair() {
    // Craft an fg/bg pair whose ΔE is already right at the edge, then apply a
    // drift that would otherwise push them together (e.g. both toward the
    // same darker lightness) and confirm the safety net keeps ΔE at/above the
    // floor.
    let fg = 0x0030_3030u32;
    let bg = 0x0028_2828u32; // deliberately close to fg — low starting ΔE
    let low_de = re_core::oklab::delta_e(
        linear_to_oklab(unpack_linear(fg)),
        linear_to_oklab(unpack_linear(bg)),
    );
    assert!(
        low_de < CONTRAST_FLOOR_DELTA_E,
        "fixture must start BELOW the floor (got {low_de})"
    );

    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        // A literal all-zero offset would trip `apply_drift_with`'s
        // zero-delta early-exit (a true no-op fast path) BEFORE the R5 safety
        // net ever runs. An imperceptible non-zero chroma nudge keeps the
        // cell on the "something changed this frame" path so the floor check
        // actually executes, without materially perturbing the fixture's
        // already-low starting ΔE.
        delta_l: 0.0,
        delta_c: 0.0001,
        delta_h_deg: 0.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    fb.set(0, 0, floor_cell(Vec2::new(1.0, 1.0), fg, bg));
    apply_drift(&mut fb, &sources);

    let out = fb.get(0, 0).unwrap();
    let after_de = re_core::oklab::delta_e(
        linear_to_oklab(unpack_linear(out.fg)),
        linear_to_oklab(unpack_linear(out.bg)),
    );
    assert!(
        after_de >= CONTRAST_FLOOR_DELTA_E - 1e-3,
        "R5: contrast floor must be restored, got ΔE={after_de}"
    );
}

#[test]
fn blank_cells_are_exempt_from_the_contrast_floor() {
    // A space glyph has κ≈0 — R5 explicitly only protects κ≥0.1 cells, so a
    // blank cell's fg/bg must NOT be nudged even if they nearly collapse.
    let fg = 0x0030_3030u32;
    let bg = 0x0028_2828u32;
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 5.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    let mut c = floor_cell(Vec2::new(1.0, 1.0), fg, bg);
    c.glyph = u32::from(' ');
    fb.set(0, 0, c);
    apply_drift(&mut fb, &sources);
    let out = fb.get(0, 0).unwrap();
    // fg/bg should still have moved by the drift itself, but NOT been pulled
    // apart further by the contrast-floor nudge specifically — check this by
    // comparing against a non-blank cell run through the SAME drift alone
    // (no floor kick-in expected to differ only by blankness)... simpler and
    // robust: just confirm the safety net's `glyph == ' '` early-out is hit by
    // asserting the result matches `drift_packed` directly with no nudge.
    let expected_fg = drift_packed(fg, 0.0, 0.0, 5.0);
    let expected_bg = drift_packed(bg, 0.0, 0.0, 5.0);
    assert_eq!(
        out.fg, expected_fg,
        "blank cell fg must be exactly the drifted colour, no floor nudge"
    );
    assert_eq!(
        out.bg, expected_bg,
        "blank cell bg must be exactly the drifted colour, no floor nudge"
    );
}

#[test]
fn underline_colour_drifts_too_when_present() {
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 30.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    let mut c = floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010);
    c.style.underline = 0x0080_4020;
    fb.set(0, 0, c);
    apply_drift(&mut fb, &sources);
    let out = fb.get(0, 0).unwrap();
    assert_ne!(
        out.style.underline, 0x0080_4020,
        "underline colour must drift when present"
    );
}

#[test]
fn no_underline_sentinel_is_never_run_through_the_field() {
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 30.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010),
    ); // underline == NO_UNDERLINE
    apply_drift(&mut fb, &sources);
    assert_eq!(fb.get(0, 0).unwrap().style.underline, re_core::NO_UNDERLINE);
}

#[test]
fn is_semantic_default_follows_the_material_tag() {
    let untagged = Cell::default();
    let tagged = Cell {
        material: 3,
        ..Default::default()
    };
    assert!(!is_semantic_default(&untagged));
    assert!(is_semantic_default(&tagged));
}

#[test]
fn apply_drift_with_honours_a_custom_semantic_predicate() {
    // A caller-supplied predicate that treats EVERYTHING as semantic must
    // zero every hue shift, even for a material==0 cell that the default
    // predicate would treat as non-semantic.
    let src = DriftSource {
        world_pos: Vec2::new(0.0, 0.0),
        radius: 100.0,
        delta_l: 0.0,
        delta_c: 0.0,
        delta_h_deg: 40.0,
    };
    let sources = DriftSources {
        world: vec![src],
        gaze: None,
    };
    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        floor_cell(Vec2::new(1.0, 1.0), 0x0040_C0C0, 0x0010_1010),
    );

    let before_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    apply_drift_with(&mut fb, &sources, |_| true);
    let after_hue = hue_deg(fb.get(0, 0).unwrap().fg);
    let moved = wrap_hue_delta((after_hue - before_hue).to_radians()).to_degrees();
    assert!(
        moved.abs() < 1.0,
        "a fully-semantic predicate must suppress hue drift everywhere"
    );
}

#[test]
fn radial_falloff_is_one_at_centre_and_zero_past_radius() {
    assert!((radial_falloff(0.0, 4.0) - 1.0).abs() < 1e-6);
    assert_eq!(radial_falloff(4.0, 4.0), 0.0);
    assert_eq!(radial_falloff(10.0, 4.0), 0.0);
    assert_eq!(
        radial_falloff(1.0, 0.0),
        0.0,
        "a zero-radius source must never divide by zero"
    );
}
