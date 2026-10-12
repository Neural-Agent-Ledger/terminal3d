//! Integration tests for the pure glitch core.
//!
//! Proves the acceptance contract:
//! * determinism — same `(fb, cfg, time, seed)` → byte-identical frame;
//! * off / intensity-0 = identity — the frame is provably unchanged;
//! * each effect actually CHANGES the frame when enabled alone;
//! * dropout/corruption counts SCALE with master intensity;
//! * the snapshot approach does NOT compound — re-deriving from the clean frame
//!   each tick is bounded, unlike glitching the live buffer in place.

use re_core::{Cell, Framebuffer, SurfaceKind};
use re_glitch::{apply_glitch, GlitchConfig};

/// A deterministic, content-rich test frame: a gradient of glyphs + colours so
/// every effect (glyph, fg, bg, geometry) has something to move/corrupt. No RNG.
fn make_frame(cols: usize, rows: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(cols, rows);
    for row in 0..rows {
        for col in 0..cols {
            let glyph = b'!' as u32 + ((col * 7 + row * 13) % 90) as u32;
            let fg = (((col * 17) & 0xFF) << 16 | ((row * 23) & 0xFF) << 8 | ((col * row) & 0xFF))
                as u32;
            let bg = ((((col + 5) * 11) & 0xFF) << 16 | ((row * 3) & 0xFF) << 8 | 0x20) as u32;
            fb.set(
                col,
                row,
                Cell {
                    glyph,
                    fg,
                    bg,
                    depth: 1.0 + col as f32,
                    surface: SurfaceKind::WallSolid,
                    ..Cell::default()
                },
            );
        }
    }
    fb
}

/// Snapshot every cell's (glyph, fg, bg) for comparison.
fn capture(fb: &Framebuffer) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::with_capacity(fb.cols() * fb.rows());
    for row in 0..fb.rows() {
        for col in 0..fb.cols() {
            let c = fb.get(col, row).unwrap();
            v.push((c.glyph, c.fg, c.bg));
        }
    }
    v
}

/// Count cells that differ between two captures.
fn diff_count(a: &[(u32, u32, u32)], b: &[(u32, u32, u32)]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn determinism_same_inputs_same_frame() {
    let cfg = GlitchConfig::heavy();
    let mut a = make_frame(80, 32);
    let mut b = make_frame(80, 32);
    apply_glitch(&mut a, &cfg, 2.5, 0xABCD);
    apply_glitch(&mut b, &cfg, 2.5, 0xABCD);
    assert_eq!(capture(&a), capture(&b), "same inputs must give same frame");

    // A different time OR seed must (almost surely) differ.
    let mut c = make_frame(80, 32);
    apply_glitch(&mut c, &cfg, 9.99, 0xABCD);
    assert_ne!(
        capture(&a),
        capture(&c),
        "different time should change the look"
    );
    let mut d = make_frame(80, 32);
    apply_glitch(&mut d, &cfg, 2.5, 0x1234);
    assert_ne!(
        capture(&a),
        capture(&d),
        "different seed should change the look"
    );
}

#[test]
fn off_is_identity() {
    let base = make_frame(64, 24);
    let mut f = base.clone();
    apply_glitch(&mut f, &GlitchConfig::off(), 5.0, 42);
    assert_eq!(
        capture(&base),
        capture(&f),
        "off config must leave frame unchanged"
    );
}

#[test]
fn intensity_zero_is_identity_even_with_effects_enabled() {
    // All effects enabled but master intensity 0 → still identity.
    let mut cfg = GlitchConfig::heavy();
    cfg.intensity = 0.0;
    let base = make_frame(64, 24);
    let mut f = base.clone();
    apply_glitch(&mut f, &cfg, 5.0, 42);
    assert_eq!(capture(&base), capture(&f), "master intensity 0 = identity");
}

/// Helper: a config with ONLY one named effect enabled at full master.
fn only(name: &str) -> GlitchConfig {
    let mut c = GlitchConfig::off();
    c.intensity = 1.0;
    match name {
        "scanlines" => c.scanlines.enabled = true,
        "bitrot" => c.bitrot.enabled = true,
        "tear" => c.tear.enabled = true,
        "aberration" => c.aberration.enabled = true,
        "block" => c.block.enabled = true,
        "dropout" => c.dropout.enabled = true,
        "warp" => c.warp.enabled = true,
        "invert" => {
            c.invert.enabled = true;
            c.invert.rate = 0.0; // held ON so the single-time test reliably flashes
        }
        _ => unreachable!(),
    }
    c
}

#[test]
fn each_effect_changes_the_frame() {
    let effects = [
        "scanlines",
        "bitrot",
        "tear",
        "aberration",
        "block",
        "dropout",
        "warp",
        "invert",
    ];
    let base = make_frame(96, 40);
    let cap0 = capture(&base);
    for name in effects {
        let mut f = base.clone();
        apply_glitch(&mut f, &only(name), 1.7, 7);
        let changed = diff_count(&cap0, &capture(&f));
        assert!(
            changed > 0,
            "effect '{name}' must change at least one cell (changed {changed})"
        );
    }
}

#[test]
fn dropout_count_scales_with_intensity() {
    let base = make_frame(100, 40);
    let count_black = |intensity: f32| -> usize {
        let mut c = GlitchConfig::off();
        c.intensity = intensity;
        c.dropout.enabled = true;
        c.dropout.density = 0.5; // generous so the scaling is clearly visible
        let mut f = base.clone();
        apply_glitch(&mut f, &c, 0.0, 99);
        // Count cells blanked to black (fg==0 && bg==0) that weren't before.
        let mut n = 0;
        for row in 0..f.rows() {
            for col in 0..f.cols() {
                let cell = f.get(col, row).unwrap();
                let orig = base.get(col, row).unwrap();
                let was_black = orig.fg == 0 && orig.bg == 0;
                if cell.fg == 0 && cell.bg == 0 && !was_black {
                    n += 1;
                }
            }
        }
        n
    };
    let low = count_black(0.2);
    let high = count_black(0.9);
    assert!(
        high > low,
        "dropout count must scale with intensity (low={low}, high={high})"
    );
    assert_eq!(count_black(0.0), 0, "zero intensity drops nothing");
}

#[test]
fn bitrot_count_scales_with_intensity() {
    let base = make_frame(100, 40);
    let cap0 = capture(&base);
    let corrupt_count = |intensity: f32| -> usize {
        let mut c = GlitchConfig::off();
        c.intensity = intensity;
        c.bitrot.enabled = true;
        c.bitrot.density = 0.3;
        c.bitrot.spread = 0.0; // isolate the base density scaling
        let mut f = base.clone();
        apply_glitch(&mut f, &c, 0.0, 5);
        diff_count(&cap0, &capture(&f))
    };
    let low = corrupt_count(0.2);
    let high = corrupt_count(0.9);
    assert!(
        high > low,
        "bit-rot count must scale with intensity (low={low}, high={high})"
    );
}

#[test]
fn bitrot_spread_increases_corruption() {
    // With spread > 0, contagion should corrupt MORE cells than density alone.
    let base = make_frame(100, 40);
    let cap0 = capture(&base);
    let with_spread = |spread: f32| -> usize {
        let mut c = GlitchConfig::off();
        c.intensity = 1.0;
        c.bitrot.enabled = true;
        c.bitrot.density = 0.15;
        c.bitrot.spread = spread;
        let mut f = base.clone();
        apply_glitch(&mut f, &c, 0.0, 11);
        diff_count(&cap0, &capture(&f))
    };
    assert!(
        with_spread(1.0) > with_spread(0.0),
        "spreading bit-rot must corrupt more than non-spreading"
    );
}

#[test]
fn snapshot_approach_does_not_compound() {
    // Re-deriving the glitch from the CLEAN frame each tick (the plugin's model)
    // is bounded: the diff vs clean stays comparable tick-to-tick. Whereas
    // glitching the LIVE buffer in place compounds — strictly more cells diverge.
    let clean = make_frame(80, 32);
    let cap_clean = capture(&clean);
    let cfg = GlitchConfig::heavy();

    // The SNAPSHOT model: always start from clean, glitch once, at advancing time.
    let mut snapshot_diffs = Vec::new();
    for tick in 0..10 {
        let mut f = clean.clone(); // re-derive from clean every tick
        apply_glitch(&mut f, &cfg, tick as f32 * 0.1, 1);
        snapshot_diffs.push(diff_count(&cap_clean, &capture(&f)));
    }

    // To measure compounding cleanly we use a GEOMETRY config (tearing): a slice
    // shifted by N cells once differs from clean by that slice; shifted again
    // in-place it travels 2N — strictly FURTHER from clean. A frozen pattern (rate
    // 0) so any divergence growth is purely the compounding, not time evolution.
    let mut geo = GlitchConfig::off();
    geo.intensity = 1.0;
    geo.tear.enabled = true;
    geo.tear.tear_frac = 1.0; // every slice tears, maximizing the signal
    geo.tear.max_shift = 5;
    geo.tear.rate = 0.0; // frozen pattern: divergence growth == compounding
    geo.block.enabled = true;
    geo.block.jump_frac = 1.0;
    geo.block.rate = 0.0;

    // Snapshot model: re-derive from clean each tick → IDENTICAL frame each time
    // (same pattern over the same clean source). Bounded, never compounds.
    let mut snap_frames = Vec::new();
    for _ in 0..10 {
        let mut f = clean.clone();
        apply_glitch(&mut f, &geo, 0.0, 1);
        snap_frames.push(capture(&f));
    }
    let snap_diff = diff_count(&cap_clean, snap_frames.last().unwrap());

    // In-place model: keep glitching the SAME buffer. Each pass shifts the
    // already-shifted content FURTHER, so successive frames keep CHANGING — the
    // glitch compounds. We capture every tick to compare against the previous.
    let mut live = clean.clone();
    let mut inplace_frames = Vec::new();
    for _ in 0..10 {
        apply_glitch(&mut live, &geo, 0.0, 1);
        inplace_frames.push(capture(&live));
    }

    // CORE GUARANTEE: the snapshot model reproduces the IDENTICAL glitched frame
    // every tick — re-deriving from the clean source never compounds.
    assert!(
        snap_frames.iter().all(|f| f == &snap_frames[0]),
        "snapshot model must reproduce the SAME glitched frame every tick (no compounding)"
    );

    // CONTRAST: the in-place model keeps producing DIFFERENT frames as the
    // displacement accretes — i.e. it compounds. At least one tick-to-tick step
    // must differ (in practice every early one does).
    let inplace_keeps_changing = inplace_frames.windows(2).any(|w| w[0] != w[1]);
    assert!(
        inplace_keeps_changing,
        "in-place glitch must compound: successive in-place frames must differ"
    );

    // And the in-place result is NOT the snapshot result — the two models truly
    // diverge (proves the plugin's snapshot choice is meaningful, not a no-op).
    assert_ne!(
        inplace_frames.last().unwrap(),
        snap_frames.last().unwrap(),
        "compounded in-place frame must differ from the bounded snapshot frame"
    );

    // Sanity: the snapshot glitch actually changed the frame, and the heavy run
    // (above) stayed bounded without crashing.
    assert!(snap_diff > 0, "snapshot glitch should change the frame");
    let snap_max = *snapshot_diffs.iter().max().unwrap();
    assert!(snap_max > 0, "heavy snapshot run should change the frame");
}

#[test]
fn handles_empty_and_tiny_frames() {
    // No panic on degenerate sizes.
    let mut z = Framebuffer::new(0, 0);
    apply_glitch(&mut z, &GlitchConfig::heavy(), 1.0, 1);
    let mut tiny = make_frame(1, 1);
    apply_glitch(&mut tiny, &GlitchConfig::heavy(), 1.0, 1);
}
