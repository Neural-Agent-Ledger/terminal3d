//! `re_render::palette_gate` — the engine-side "palette gate": a progress
//! lock on which hues a frame is allowed to show, so a game can make colour
//! itself something the player unlocks over time.
//!
//! Pipeline stage: a post-shading colour remap. Inputs: a shaded
//! [`Framebuffer`] + a [`PaletteGate`] (compiled into a [`GateLut`]). Output:
//! the same framebuffer with every selected channel's hue/chroma remapped
//! in place via [`apply_gate`]. Knobs: [`PaletteGate`]'s hue arcs, soft-edge
//! width, chroma cap, mono tint, channel mask, and leak.
//!
//! This module is PURE ENGINE DATA + MATH: it has no dependency on `re_game`
//! (no save/progress/state-machine knowledge). A game-side driver turns
//! campaign progress into a [`PaletteGate`] value, the same split
//! `mirror_ramp.rs` uses for [`crate::MirrorClarity`].
//!
//! # What the gate does
//!
//! A [`PaletteGate`] describes which Oklch hues are "unlocked" (one or more
//! allowed arcs, in degrees) and how saturated they may be (a chroma cap).
//! [`apply_gate`] remaps every gated channel of every cell through a
//! precomputed [`GateLut`]:
//!
//! * A colour whose hue falls inside an allowed arc keeps its hue, with its
//!   chroma clipped to [`PaletteGate::chroma_cap`].
//! * A colour outside every arc (or the WHOLE frame, when `hue_arcs` is empty
//!   — "mono") is forced to [`PaletteGate::mono_tint`] (or desaturated to
//!   neutral grey when `mono_tint` is `None`) at its OWN lightness — a
//!   red-on-black VALUE image: shape and depth still read because L carries
//!   them, only hue/chroma are locked.
//! * [`PaletteGate::arc_soft_deg`] blends the two outcomes smoothly across the
//!   arc boundary so newly-unlocked hues don't pop in at a hard edge.
//! * Lightness (`L`) is NEVER touched by the gate itself — only hue and
//!   chroma. That is what keeps the gated image legible as a value image.
//!
//! # Where it runs
//!
//! Intended to run AFTER `re_postfx`, BEFORE `re_glitch` — the caller (the
//! game's render pipeline) owns the actual call order; this module only
//! provides the pure `apply_gate` function + the `GateLut` it is applied
//! through. [`PaletteGate::leak`] is what lets glitch show a few frames of a
//! locked colour "from the future": the caller raises `leak` transiently and
//! rebuilds (or reuses, since leak is a per-call blend, not baked into the
//! LUT) the gate application for those frames.
//!
//! # Cost
//!
//! [`GateLut::build`] pre-computes a 33-level-per-channel cube once (`33³`
//! packed colours, ~108 KiB `Box<[u32]>`), rebuilt only when the gate itself
//! changes. [`GateLut::apply`] is then a single quantized array index per
//! colour — no per-cell Oklab math. Dumps and goldens use
//! [`PaletteGate::full`] (a permanently no-op gate), so they stay `Full` and
//! byte-identical to today unless a test opts in.

use re_core::oklab::{packed_to_oklch, wrap_hue_delta, Oklch};
use re_core::{Cell, Framebuffer, SurfaceKind};

/// Which of a cell's colour channels the gate remaps. Bit-set over a `u8`,
/// matching the hand-rolled style [`re_core::CellStyle`] already uses in this
/// codebase (no external bitflags dependency).
///
/// `DRIFT` and `SKY` are not literal `Cell` channels — they gate which UPSTREAM
/// SOURCES the palette constraint extends to, since [`apply_gate`] only sees
/// the resolved `fg`/`bg`/underline triple, not where a colour came from:
/// * `SKY` — whether sky cells (`SurfaceKind::Empty`/`Ceiling`, the same "is
///   sky" predicate used throughout `re_render`) are gated at all. Unset ⇒ sky
///   passes through untouched (e.g. a starfield that should stay full-colour
///   even while the ground floor is phosphor-red).
/// * `DRIFT` — whether the colour-drift field (`color_drift.rs`) is
///   itself constrained to stay within this gate's arcs, versus being allowed
///   to author a transient excursion outside them that the NEXT `apply_gate`
///   call then re-clamps anyway. This is a hook for `color_drift.rs` to read;
///   `apply_gate` does not need it (it always re-clamps every colour it
///   sees) — it exists so a caller building a `ChannelMask` for both passes
///   has one flag to share instead of inventing a second type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelMask(pub u8);

impl ChannelMask {
    /// The foreground colour channel.
    pub const FG: ChannelMask = ChannelMask(1 << 0);
    /// The background colour channel.
    pub const BG: ChannelMask = ChannelMask(1 << 1);
    /// The underline colour channel.
    pub const UL: ChannelMask = ChannelMask(1 << 2);
    /// Whether the colour-drift field is itself gated (see the struct-level
    /// doc above for why this is a mask bit rather than an implicit always).
    pub const DRIFT: ChannelMask = ChannelMask(1 << 3);
    /// Whether sky cells are gated at all.
    pub const SKY: ChannelMask = ChannelMask(1 << 4);

    /// No channels — a gate with this mask is a literal no-op (matches
    /// [`PaletteGate::full`]'s intent when combined with an all-pass gate).
    pub const NONE: ChannelMask = ChannelMask(0);

    /// Every channel + every upstream source: the gate applies everywhere.
    pub const ALL: ChannelMask =
        ChannelMask(Self::FG.0 | Self::BG.0 | Self::UL.0 | Self::DRIFT.0 | Self::SKY.0);

    /// The common case: colour + sky, but the drift field is left alone (drift
    /// authors its own chroma caps in `color_drift.rs` instead of deferring
    /// to this mask).
    pub const COLOR_AND_SKY: ChannelMask =
        ChannelMask(Self::FG.0 | Self::BG.0 | Self::UL.0 | Self::SKY.0);

    /// Whether every bit set in `bit` is also set in `self`.
    #[inline]
    #[must_use]
    pub const fn contains(self, bit: ChannelMask) -> bool {
        self.0 & bit.0 == bit.0
    }

    /// The mask with every bit set in either `self` or `other`.
    #[inline]
    #[must_use]
    pub const fn union(self, other: ChannelMask) -> ChannelMask {
        ChannelMask(self.0 | other.0)
    }
}

/// One allowed Oklch hue arc, in DEGREES (`0..360`, wrapping). `lo`/`hi` are
/// authored in the natural reading direction (e.g. red→yellow `(-10.0,
/// 60.0)`); the arc is evaluated on the circle, so `lo > hi` (an arc crossing
/// 0°/360°) is valid and means "wrap through 0".
pub type HueArcDeg = (f32, f32);

/// The progress-gated colour palette. Pure engine data: no dependency on
/// save/progress — a game-side driver turns campaign progress into one of
/// these.
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteGate {
    /// Allowed Oklch hue arcs, in degrees. EMPTY means "mono": every hue in
    /// the frame is forced to [`Self::mono_tint`] (or grey) — the campaign's
    /// L1 "red phosphor only" stage.
    pub hue_arcs: Vec<HueArcDeg>,
    /// Soft-edge width in degrees on EITHER side of each arc boundary, so a
    /// hue crossing the edge fades between "allowed" and "locked" instead of
    /// popping. `0.0` = a hard edge.
    pub arc_soft_deg: f32,
    /// Maximum Oklch chroma any output colour may carry, in `[0, ~0.37]`
    /// (Oklch chroma for sRGB gamut colours tops out well under 0.4). Applied
    /// AFTER the hue-arc decision, to both allowed and locked colours.
    pub chroma_cap: f32,
    /// `(hue_deg, chroma)` used for a LOCKED colour (outside every arc, or
    /// every colour when `hue_arcs` is empty). `None` desaturates to neutral
    /// grey (chroma 0) instead.
    pub mono_tint: Option<(f32, f32)>,
    /// Which channels / upstream sources this gate extends to (see
    /// [`ChannelMask`]'s doc for the non-obvious `DRIFT`/`SKY` bits).
    pub channels: ChannelMask,
    /// `0..1`: how much of the ORIGINAL (ungated) colour blends back in over
    /// the gated result. `0.0` (default) = fully gated. Raised transiently by
    /// glitch so a locked colour can flicker through for a few frames — "a
    /// memory intrusion from the future".
    pub leak: f32,
}

impl PaletteGate {
    /// The permanently no-op gate: every hue allowed, chroma uncapped, no
    /// channels selected. Dumps and goldens use this so they stay `Full` and
    /// byte-identical unless a test explicitly opts into a gated stage.
    #[must_use]
    pub fn full() -> Self {
        PaletteGate {
            hue_arcs: vec![(0.0, 360.0)],
            arc_soft_deg: 0.0,
            chroma_cap: f32::INFINITY,
            mono_tint: None,
            channels: ChannelMask::NONE,
            leak: 0.0,
        }
    }

    /// `true` when this gate cannot change any pixel — either no channels are
    /// selected, or every hue is allowed at an uncapped chroma. `apply_gate`
    /// uses this as a fast, exact identity short-circuit (matches the
    /// `PostfxConfig::off()` / `PointLights::NONE` "no-op is a single branch"
    /// convention already used throughout `re_render`).
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.channels == ChannelMask::NONE
            || (self.leak >= 1.0)
            || (self.chroma_cap.is_infinite()
                && self
                    .hue_arcs
                    .iter()
                    .any(|&(lo, hi)| (hi - lo).abs() >= 360.0))
    }

    /// Does hue `h_deg` (any real, will be normalised) fall inside ANY
    /// authored arc, INCLUDING the soft margin? Returns a `[0,1]` weight: `1`
    /// = fully inside an arc, `0` = fully locked, in between = inside the soft
    /// edge band. `hue_arcs` empty ⇒ always `0` (mono).
    fn arc_weight(&self, h_deg: f32) -> f32 {
        if self.hue_arcs.is_empty() {
            return 0.0;
        }
        let h = norm_deg(h_deg);
        let mut best = 0.0_f32;
        for &(lo, hi) in &self.hue_arcs {
            let w = single_arc_weight(h, lo, hi, self.arc_soft_deg);
            if w > best {
                best = w;
            }
        }
        best
    }

    /// Gate one Oklch colour (lightness untouched, hue/chroma remapped).
    /// This is the function [`GateLut::build`] samples 33³ times; direct
    /// per-colour callers (e.g. a future settings preview) can call it too.
    #[must_use]
    pub fn gate_oklch(&self, c: Oklch) -> Oklch {
        let h_deg = c.h.to_degrees();
        let w = self.arc_weight(h_deg);

        // The "unlocked" branch: same hue, chroma clipped to the cap.
        let unlocked = Oklch {
            l: c.l,
            c: c.c.min(self.chroma_cap),
            h: c.h,
        };

        // The "locked" branch: mono_tint's hue/chroma (or neutral grey), same L.
        let locked = match self.mono_tint {
            Some((tint_h_deg, tint_c)) => Oklch {
                l: c.l,
                c: tint_c.min(self.chroma_cap),
                h: tint_h_deg.to_radians(),
            },
            None => Oklch {
                l: c.l,
                c: 0.0,
                h: c.h,
            },
        };

        if w >= 1.0 {
            unlocked
        } else if w <= 0.0 {
            locked
        } else {
            // Soft-edge blend. Hue must blend along the SHORT way around the
            // circle (wrap_hue_delta), never the long way, or a colour near
            // the boundary would visibly spin through unrelated hues.
            let dh = wrap_hue_delta(unlocked.h - locked.h);
            Oklch {
                l: c.l,
                c: locked.c + (unlocked.c - locked.c) * w,
                h: locked.h + dh * w,
            }
        }
    }

    /// Blend `gated` back toward `original` by [`Self::leak`]. Both are
    /// packed `0x00RRGGBB`; a straight linear-RGB lerp is enough for a
    /// transient glitch flicker (this is not a colour-managed operation).
    fn apply_leak(&self, original: u32, gated: u32) -> u32 {
        if self.leak <= 0.0 {
            return gated;
        }
        let leak = self.leak.clamp(0.0, 1.0);
        let o = re_core::oklab::unpack_linear(original);
        let g = re_core::oklab::unpack_linear(gated);
        re_core::oklab::pack_linear(re_core::oklab::lerp_linear(g, o, leak))
    }
}

/// Normalise a degree value into `[0, 360)`.
#[inline]
fn norm_deg(mut d: f32) -> f32 {
    d %= 360.0;
    if d < 0.0 {
        d += 360.0;
    }
    d
}

/// Angular distance (degrees, `[0,180]`) from hue `h` to the nearest point of
/// arc `(lo,hi)` — `0` when `h` is inside the arc. Handles a wrapping arc
/// (`lo > hi`, crossing 0°/360°) by testing the wrapped membership directly.
fn single_arc_weight(h: f32, lo: f32, hi: f32, soft_deg: f32) -> f32 {
    let lo = norm_deg(lo);
    let hi = norm_deg(hi);
    let inside = if lo <= hi {
        h >= lo && h <= hi
    } else {
        h >= lo || h <= hi
    };
    if inside {
        return 1.0;
    }
    if soft_deg <= 0.0 {
        return 0.0;
    }
    // Outside: distance to whichever edge is nearer, the short way round.
    let d_lo = wrap_hue_delta((h - lo).to_radians()).to_degrees().abs();
    let d_hi = wrap_hue_delta((h - hi).to_radians()).to_degrees().abs();
    let d = d_lo.min(d_hi);
    (1.0 - d / soft_deg).clamp(0.0, 1.0)
}

/// LUT resolution per channel: a 33³ RGB LUT. `32` steps between
/// samples ⇒ each sample is `255.0 / 32.0` sRGB units apart.
const LUT_N: usize = 33;

/// A pre-computed [`PaletteGate`] lookup table over quantized sRGB space:
/// `33³` packed output colours, built once per gate change. [`Self::apply`] is
/// then a single array index — no Oklab math on the hot per-cell path.
///
/// Deliberately NEAREST-sample, not trilinear: at 33 levels per channel the
/// gate's own soft edge (`arc_soft_deg`) already smooths the one place banding
/// would otherwise show (the arc boundary in hue), so the extra interpolation
/// cost is not spent. A future tuning pass can add it without changing this
/// type's public shape.
#[derive(Clone, Debug)]
pub struct GateLut {
    table: Box<[u32]>,
}

impl GateLut {
    /// Build the LUT for `gate`. Cheap-ish (33³ = 35,937 Oklab round-trips)
    /// but meant to be rebuilt only when the gate itself changes, not per
    /// frame — callers cache it (e.g. keyed on the gate's `PartialEq`).
    #[must_use]
    pub fn build(gate: &PaletteGate) -> Self {
        let mut table = vec![0u32; LUT_N * LUT_N * LUT_N].into_boxed_slice();
        for ri in 0..LUT_N {
            for gi in 0..LUT_N {
                for bi in 0..LUT_N {
                    let packed = quantized_index_to_packed(ri, gi, bi);
                    let gated_oklch = gate.gate_oklch(packed_to_oklch(packed));
                    let gated_packed = re_core::oklab::oklch_to_packed(gated_oklch);
                    table[lut_index(ri, gi, bi)] = gate.apply_leak(packed, gated_packed);
                }
            }
        }
        GateLut { table }
    }

    /// Apply the LUT to one packed `0x00RRGGBB` colour: quantize to the
    /// nearest sample per channel, then a single array index.
    #[inline]
    #[must_use]
    pub fn apply(&self, packed: u32) -> u32 {
        let r = ((packed >> 16) & 0xFF) as u8;
        let g = ((packed >> 8) & 0xFF) as u8;
        let b = (packed & 0xFF) as u8;
        let ri = byte_to_lut_index(r);
        let gi = byte_to_lut_index(g);
        let bi = byte_to_lut_index(b);
        self.table[lut_index(ri, gi, bi)]
    }
}

#[inline]
fn byte_to_lut_index(v: u8) -> usize {
    ((v as f32 / 255.0) * (LUT_N as f32 - 1.0) + 0.5) as usize
}

#[inline]
fn lut_sample_byte(i: usize) -> u8 {
    ((i as f32 / (LUT_N as f32 - 1.0)) * 255.0 + 0.5) as u8
}

#[inline]
fn quantized_index_to_packed(ri: usize, gi: usize, bi: usize) -> u32 {
    ((lut_sample_byte(ri) as u32) << 16)
        | ((lut_sample_byte(gi) as u32) << 8)
        | lut_sample_byte(bi) as u32
}

#[inline]
fn lut_index(ri: usize, gi: usize, bi: usize) -> usize {
    (ri * LUT_N + gi) * LUT_N + bi
}

/// `true` for the cells [`ChannelMask::SKY`] gates — the same "is sky"
/// predicate used throughout `re_render` (`SurfaceKind::Ceiling | Empty`).
#[inline]
fn is_sky(surface: SurfaceKind) -> bool {
    matches!(surface, SurfaceKind::Ceiling | SurfaceKind::Empty)
}

/// Apply `gate` (through its precomputed `lut`) to every selected channel of
/// every cell of `fb`. A literal no-op (`gate.is_noop()`) returns immediately
/// without touching a single cell — dumps/goldens with [`PaletteGate::full`]
/// are therefore byte-identical to the pre-gate build.
///
/// Runs AFTER `re_postfx`, BEFORE `re_glitch` — this function only
/// does the per-cell remap; pipeline ordering is the caller's (`re_game`'s)
/// responsibility.
pub fn apply_gate(fb: &mut Framebuffer, gate: &PaletteGate, lut: &GateLut) {
    if gate.is_noop() {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            if !gate.channels.contains(ChannelMask::SKY) && is_sky(s.surface) {
                continue;
            }
            let mut c: Cell = *s;
            if gate.channels.contains(ChannelMask::FG) {
                c.fg = lut.apply(c.fg);
            }
            if gate.channels.contains(ChannelMask::BG) {
                c.bg = lut.apply(c.bg);
            }
            if gate.channels.contains(ChannelMask::UL) && c.style.underline != re_core::NO_UNDERLINE
            {
                c.style.underline = lut.apply(c.style.underline);
            }
            fb.set(col, row, c);
        }
    }
}

#[cfg(test)]
mod tests;
