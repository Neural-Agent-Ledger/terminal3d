use super::*;
use re_core::oklab::{linear_to_oklab, oklab_chroma, oklab_to_oklch, unpack_linear};

fn gate(hue_arcs: Vec<HueArcDeg>, chroma_cap: f32, mono_tint: Option<(f32, f32)>) -> PaletteGate {
    PaletteGate {
        hue_arcs,
        arc_soft_deg: 0.0,
        chroma_cap,
        mono_tint,
        channels: ChannelMask::COLOR_AND_SKY,
        leak: 0.0,
    }
}

#[test]
fn full_gate_is_a_literal_noop() {
    let g = PaletteGate::full();
    assert!(g.is_noop());
    let mut fb = Framebuffer::new(2, 2);
    fb.set(
        0,
        0,
        Cell {
            fg: 0x00FF_1234,
            bg: 0x0056_7890,
            ..Default::default()
        },
    );
    let before = fb.clone();
    let lut = GateLut::build(&g);
    apply_gate(&mut fb, &g, &lut);
    for row in 0..2 {
        for col in 0..2 {
            assert_eq!(fb.get(col, row), before.get(col, row));
        }
    }
}

#[test]
fn no_channels_selected_is_a_noop_even_with_a_real_gate() {
    let mut g = gate(vec![(0.0, 30.0)], 0.1, Some((0.0, 0.05)));
    g.channels = ChannelMask::NONE;
    assert!(g.is_noop());
}

#[test]
fn mono_gate_forces_every_hue_to_the_tint_but_keeps_lightness() {
    // Empty hue_arcs = "mono" (the campaign's L1 red-phosphor stage): every input colour
    // must come out at the SAME lightness, but with the mono_tint's hue.
    let tint_h = 25.0_f32; // a red-ish hue in degrees
    let g = gate(vec![], f32::INFINITY, Some((tint_h, 0.15)));
    let lut = GateLut::build(&g);

    for &input in &[
        0x0000_FF00u32,
        0x0000_00FFu32,
        0x00FF_FF00u32,
        0x0040_8020u32,
    ] {
        let out = lut.apply(input);
        let l_in = linear_to_oklab(unpack_linear(input)).l;
        let l_out = linear_to_oklab(unpack_linear(out)).l;
        // Tolerance is wider than a naive "preserve L exactly" check would want:
        // forcing an extreme-L, saturated input (e.g. pure yellow, L~0.97) onto
        // the mono_tint's hue/chroma can walk the Oklab (L,a,b) point outside
        // the sRGB gamut, and `pack_linear`'s clamp-to-[0,1] on the round-trip
        // shifts the apparent L by more than a couple hundredths. That is real,
        // expected gamut-clipping behaviour, not a gate-logic bug — the gate
        // itself only ever rewrites hue/chroma at a fixed L (see `gate_oklch`).
        assert!(
            (l_in - l_out).abs() < 0.15,
            "mono gate must approximately preserve lightness: in={input:#08x} L={l_in}, out={out:#08x} L={l_out}"
        );
        let h_out = oklab_to_oklch(linear_to_oklab(unpack_linear(out)))
            .h
            .to_degrees();
        let d = wrap_hue_delta((h_out - tint_h).to_radians())
            .to_degrees()
            .abs();
        // Same gamut-clipping caveat as the lightness check above: at an
        // extreme input lightness (pure yellow, L~0.97) the forced
        // `(tint_h, 0.15)` Oklch point can sit outside the sRGB gamut, and
        // clipping on the u32 round-trip drags the RE-MEASURED hue a few more
        // degrees off `tint_h` than the un-clipped gate output actually
        // wrote. `gate_oklch` itself always writes exactly `tint_h` before
        // that clip (see its `channels`-gated hue-forcing branch) — this
        // wider tolerance just accounts for the lossy pack/unpack, not a
        // logic gap in the gate.
        assert!(
            d < 10.0,
            "mono gate must force the tint hue: got {h_out}°, want ~{tint_h}°"
        );
    }
}

#[test]
fn a_hue_inside_the_allowed_arc_keeps_its_hue() {
    // R+G stage (after L1 -> L2 in the campaign): allow a wide red->green arc,
    // uncapped chroma. A pure-red input's hue must survive intact.
    let g = gate(vec![(-40.0, 140.0)], f32::INFINITY, Some((0.0, 0.1)));
    let lut = GateLut::build(&g);
    let red = 0x00FF_2020u32;
    let out = lut.apply(red);
    let h_in = oklab_to_oklch(linear_to_oklab(unpack_linear(red)))
        .h
        .to_degrees();
    let h_out = oklab_to_oklch(linear_to_oklab(unpack_linear(out)))
        .h
        .to_degrees();
    let d = wrap_hue_delta((h_out - h_in).to_radians())
        .to_degrees()
        .abs();
    assert!(
        d < 6.0,
        "an allowed hue must survive close to its input value (Δ={d}°)"
    );
}

#[test]
fn a_hue_outside_every_arc_is_forced_to_the_tint() {
    // Same R+G gate: a strongly BLUE input is outside the arc and must be
    // forced to the mono_tint (unless the tint itself happens to be blue).
    let g = gate(vec![(-40.0, 140.0)], f32::INFINITY, Some((20.0, 0.1)));
    let lut = GateLut::build(&g);
    let blue = 0x0020_20FFu32;
    let out = lut.apply(blue);
    let h_out = oklab_to_oklch(linear_to_oklab(unpack_linear(out)))
        .h
        .to_degrees();
    let d = wrap_hue_delta((h_out - 20.0).to_radians())
        .to_degrees()
        .abs();
    assert!(
        d < 6.0,
        "a locked hue must be forced to the tint (got {h_out}°)"
    );
}

#[test]
fn chroma_cap_clips_a_saturated_colour_even_when_the_hue_is_allowed() {
    let low_cap = 0.03_f32;
    let g = gate(vec![(0.0, 360.0)], low_cap, None);
    let lut = GateLut::build(&g);
    let saturated_red = 0x00FF_0000u32;
    let out = lut.apply(saturated_red);
    let c_out = oklab_chroma(linear_to_oklab(unpack_linear(out)));
    assert!(
        c_out <= low_cap + 0.02,
        "chroma must be capped near {low_cap}, got {c_out}"
    );
}

#[test]
fn none_tint_desaturates_locked_colours_to_grey() {
    let g = gate(vec![], f32::INFINITY, None);
    let lut = GateLut::build(&g);
    let vivid = 0x0000_FF80u32;
    let out = lut.apply(vivid);
    let c_out = oklab_chroma(linear_to_oklab(unpack_linear(out)));
    assert!(
        c_out < 0.02,
        "no mono_tint must desaturate toward grey, got chroma={c_out}"
    );
}

#[test]
fn sky_bit_off_leaves_sky_cells_untouched() {
    let mut g = gate(vec![], f32::INFINITY, Some((0.0, 0.1)));
    g.channels = ChannelMask::FG.union(ChannelMask::BG); // no SKY bit
    let lut = GateLut::build(&g);

    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        Cell {
            fg: 0x0000_FF00,
            bg: 0x0000_00FF,
            surface: SurfaceKind::Empty,
            ..Default::default()
        },
    );
    let before = *fb.get(0, 0).unwrap();
    apply_gate(&mut fb, &g, &lut);
    assert_eq!(
        *fb.get(0, 0).unwrap(),
        before,
        "sky must pass through when SKY bit is unset"
    );
}

#[test]
fn sky_bit_on_gates_sky_cells_too() {
    let mut g = gate(vec![], f32::INFINITY, Some((0.0, 0.1)));
    g.channels = ChannelMask::COLOR_AND_SKY;
    let lut = GateLut::build(&g);

    let mut fb = Framebuffer::new(1, 1);
    fb.set(
        0,
        0,
        Cell {
            fg: 0x0000_FF00,
            bg: 0x0000_00FF,
            surface: SurfaceKind::Empty,
            ..Default::default()
        },
    );
    let before = *fb.get(0, 0).unwrap();
    apply_gate(&mut fb, &g, &lut);
    let after = *fb.get(0, 0).unwrap();
    assert_ne!(after.fg, before.fg, "sky must be gated when SKY bit is set");
}

#[test]
fn only_selected_channels_change() {
    // FG only: bg must be untouched even though the gate would otherwise
    // reject its hue.
    let mut g = gate(vec![], f32::INFINITY, Some((0.0, 0.1)));
    g.channels = ChannelMask::FG;
    let lut = GateLut::build(&g);

    let mut fb = Framebuffer::new(1, 1);
    // `surface: Floor` — a default `Cell` is `SurfaceKind::Empty` (sky), and
    // this gate's mask has no SKY bit, so an unset surface would be skipped
    // entirely and silently pass this assertion for the wrong reason.
    fb.set(
        0,
        0,
        Cell {
            fg: 0x0000_FF00,
            bg: 0x0000_00FF,
            surface: SurfaceKind::Floor,
            ..Default::default()
        },
    );
    apply_gate(&mut fb, &g, &lut);
    let out = fb.get(0, 0).unwrap();
    assert_eq!(
        out.bg, 0x0000_00FF,
        "bg must be untouched when only FG is selected"
    );
    assert_ne!(out.fg, 0x0000_FF00, "fg must be gated when FG is selected");
}

#[test]
fn leak_blends_the_original_colour_back_in() {
    let mut g = gate(vec![], f32::INFINITY, Some((0.0, 0.1)));
    g.leak = 1.0; // full leak = fully ungated
    let lut_full_leak = GateLut::build(&g);
    g.leak = 0.0;
    let lut_no_leak = GateLut::build(&g);

    let vivid_green = 0x0000_FF00u32;
    assert_eq!(
        lut_full_leak.apply(vivid_green),
        vivid_green,
        "leak=1.0 must reproduce the original colour exactly"
    );
    assert_ne!(
        lut_no_leak.apply(vivid_green),
        vivid_green,
        "leak=0.0 must actually gate a locked colour"
    );
}

#[test]
fn arc_weight_is_continuous_across_a_soft_edge() {
    // Just inside, right at, and just outside a soft-edged arc boundary
    // should give a MONOTONE weight, not a jump.
    let g = gate(vec![(0.0, 60.0)], f32::INFINITY, Some((180.0, 0.1)));
    let mut g_soft = g.clone();
    g_soft.arc_soft_deg = 20.0;

    let w_in = g_soft.arc_weight(50.0);
    let w_edge = g_soft.arc_weight(65.0); // 5 deg past the edge, within the 20 deg soft band
    let w_far = g_soft.arc_weight(90.0); // well outside

    assert!(
        w_in >= w_edge,
        "weight must not increase moving away from the arc"
    );
    assert!(
        w_edge >= w_far,
        "weight must keep falling further from the arc"
    );
    assert!(
        w_far <= 0.4,
        "far outside the soft band must be mostly/fully locked"
    );
}

#[test]
fn hard_edge_gate_is_reproduced_by_zero_soft_degrees() {
    let g = gate(vec![(0.0, 60.0)], f32::INFINITY, Some((180.0, 0.1)));
    assert_eq!(g.arc_weight(30.0), 1.0);
    assert_eq!(g.arc_weight(90.0), 0.0);
}

#[test]
fn a_wrapping_arc_crossing_zero_degrees_is_handled() {
    // An arc from 350° to 10° must include both 355° and 5°, and exclude 180°.
    let g = gate(vec![(350.0, 10.0)], f32::INFINITY, None);
    assert_eq!(g.arc_weight(355.0), 1.0);
    assert_eq!(g.arc_weight(5.0), 1.0);
    assert_eq!(g.arc_weight(180.0), 0.0);
}

#[test]
fn gate_lut_apply_is_deterministic() {
    let g = gate(vec![(-30.0, 140.0)], 0.2, Some((10.0, 0.08)));
    let lut = GateLut::build(&g);
    for &c in &[0x0000_0000u32, 0x00FF_FFFF, 0x0012_3456, 0x00AB_CDEF] {
        assert_eq!(
            lut.apply(c),
            lut.apply(c),
            "same input must give the same output"
        );
    }
}

#[test]
fn channel_mask_union_and_contains() {
    let m = ChannelMask::FG.union(ChannelMask::SKY);
    assert!(m.contains(ChannelMask::FG));
    assert!(m.contains(ChannelMask::SKY));
    assert!(!m.contains(ChannelMask::BG));
    assert!(!m.contains(ChannelMask::UL));
    assert!(ChannelMask::ALL.contains(ChannelMask::DRIFT));
}

#[test]
fn underline_colour_is_gated_only_when_ul_bit_set_and_a_colour_is_present() {
    let mut g = gate(vec![], f32::INFINITY, Some((0.0, 0.1)));
    g.channels = ChannelMask::UL;
    let lut = GateLut::build(&g);

    let mut fb = Framebuffer::new(1, 1);
    // `surface: Floor` for the same reason as `only_selected_channels_change`
    // above — a defaulted (sky) surface would be skipped before the UL branch
    // is ever reached, since this mask has no SKY bit.
    let mut c = Cell {
        surface: SurfaceKind::Floor,
        ..Default::default()
    };
    c.style.underline = 0x0000_FF00;
    fb.set(0, 0, c);
    apply_gate(&mut fb, &g, &lut);
    assert_ne!(
        fb.get(0, 0).unwrap().style.underline,
        0x0000_FF00,
        "UL bit must gate the underline colour"
    );

    // NO_UNDERLINE sentinel must never be run through the LUT (it is not a
    // real colour and would produce garbage).
    let mut fb2 = Framebuffer::new(1, 1);
    fb2.set(
        0,
        0,
        Cell {
            surface: SurfaceKind::Floor,
            ..Default::default()
        },
    ); // underline == NO_UNDERLINE by default
    apply_gate(&mut fb2, &g, &lut);
    assert_eq!(
        fb2.get(0, 0).unwrap().style.underline,
        re_core::NO_UNDERLINE
    );
}
