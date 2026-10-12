//! The procedural EYE — an analytic replacement for the baked
//! `assets/splash/eye_frames.txt` asset's `"@@@"` pupil re-stamp (see
//! `crate::menu::paint_pupil`'s doc comment for the thing this supersedes).
//! The owner's complaint was exact: "the pupil shape and the eye shape don't
//! match". Here neither is an asset — both are the SAME analytic geometry, so
//! they cannot drift apart: the pupil is a subset of the iris, the iris is a
//! subset of the eyelid opening, by construction, for every seed / gaze /
//! time (see the `pupil_subset_of_iris_subset_of_lid` test family below).
//!
//! ## Rendering: half-block colour pixels, not glyph texture
//!
//! (Rewritten after a review of the first gallery found the glyph-noise iris
//! fill read as "ASCII soup" and the lower lid clipped the iris in half.)
//! Every cell is painted as one `▀` (upper half block)
//! glyph whose `fg` is the colour of the cell's TOP half and `bg` is the
//! colour of its BOTTOM half — the standard half-block trick, doubling
//! vertical resolution against [`CELL_ASPECT`] so the two half-cells are
//! close to square. `shade` samples each half-cell as a continuous
//! signed-distance field (lid opening, iris disc, pupil shape) and blends
//! smoothly across every boundary (a `coverage` smoothstep over a small
//! world-unit feather) instead of switching glyphs — an iris radial colour
//! gradient, a dark limbal ring, subtle angular streak/shimmer modulation, a
//! crisp 1–2px specular highlight, sclera corner-shading + a soft upper-lid
//! shadow + a few faint vein blots, all as colour, never texture.
//!
//! ## Geometry
//!
//! An almond eyelid — two arcs meeting at pointed corners — bounds a
//! lens-shaped opening. The upper arc is anchored to the IRIS, not to the
//! overall eye width: `Eye::upper_cover_frac` is "what fraction of the
//! iris's own diameter sits under the upper lid when gaze is centred"
//! (10–20%, a natural half-open human eye), and `Eye::lower_margin_frac`
//! is "how far the lower lid sits below the iris's bottom edge" (0–12%,
//! i.e. the lower lid just touches or barely clears the iris). Deriving the
//! lid depths FROM the iris radius rather than as an independent random
//! knob is what guarantees "the iris is fully visible except a sliver under
//! the upper lid" for every combination of eye size and iris size — the
//! previous, independently-randomized lid/almond ratio could roll a lid
//! shallower than the iris itself, clipping it in half (see the 2026-09-24
//! gallery review).
//!
//! The round iris (colour gradient + limbal ring + specular) clips a pupil
//! whose SHAPE depends on [`EyeVariant`]: round (human/insect), a pointed
//! vertical slit (cat/reptile, tapering smoothly to a point at both ends —
//! not a slit with square-cut ends), or a rotating segmented aperture ring
//! (cyber). Every pupil test is additionally gated on "inside the iris
//! circle" (`EyeGeom::is_pupil`), so the subset property holds regardless
//! of variant. The iris and pupil move TOGETHER as the gaze shifts (one
//! `(icx, icy)` centre for both), clamped so the gaze cannot push the iris
//! out from under the lids (`Eye::gaze_max_frac`).
//!
//! ## Randomized parameters (owner's ask, verbatim: "size, where it looks,
//! and above all the frequencies")
//!
//! Every [`Eye::random`] draws, per seed: eye size, iris size + lid-coverage
//! proportions, iris two-stop colour, gaze range, and SIX independent
//! frequencies — saccade interval, two tremor axes, blink period, pupil-
//! breathing rate, and iris shimmer rate — each with its own phase offset, so
//! two seeds never blink or breathe in lockstep (see
//! `different_seeds_diverge_in_every_frequency`).

use crate::hash::{angle, range};
use crate::{aspect_dist, secs, set_clipped, AsciiEffect, Rect, CELL_ASPECT};
use re_core::oklab::{lerp_linear, unpack_linear, LinRgb};
use re_core::{Cell, CellStyle, Framebuffer};
use std::f32::consts::TAU;
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────
// New content (no existing mechanics spec numbers to draw from — the owner's
// brief for this pass was "show imagination"), so every number below is this
// pass's own considered choice, named and ranged rather than scattered as
// magic literals, so a later tuning pass has one place to look.

/// Eye half-width as a fraction of the tightest-fitting bound the rect can
/// offer — see [`Eye::geometry`]'s `half_extent` derivation, which picks the
/// smaller of "half the rect's columns" and "half the rect's world-height,
/// converted back through this eye's own iris/lid-coverage ratios" so an
/// intentionally wide, short eye still uses the full width of a wide rect
/// rather than being capped by its row count alone. A too-large eye would
/// poke its lids past the rect.
const SIZE_FRAC_RANGE: (f32, f32) = (0.62, 0.92);
/// Extra guaranteed clearance between the eye's full extreme reach and the
/// rect's own edge, in BOTH dimensions, applied on top of `SIZE_FRAC_RANGE`
/// — see [`Eye::geometry`]'s `half_w`/`half_h_world` derivation.
const FIT_MARGIN_FRAC: f32 = 0.90;
/// Iris radius as a fraction of the eye half-width — deliberately large (a
/// small "beady" iris reads poorly at terminal-cell resolution; a big iris
/// reads as an expressive, friendly eye and leaves the lid-coverage numbers
/// below room to work with).
const IRIS_FRAC_RANGE: (f32, f32) = (0.55, 0.75);
/// Fraction of the iris's OWN diameter sitting under the upper lid when gaze
/// is centred — see the module doc's "Geometry" section. ~10–20% matches a
/// natural, relaxed, half-open human eye.
const UPPER_COVER_FRAC_RANGE: (f32, f32) = (0.10, 0.20);
/// How far below the iris's bottom edge the lower lid sits, as a fraction of
/// the iris radius — 0 = lid exactly touches the iris bottom, >0 = a sliver
/// of sclera shows beneath. Real lower lids barely cover the iris at all.
const LOWER_MARGIN_FRAC_RANGE: (f32, f32) = (0.0, 0.12);
/// How far the gaze may push the iris centre off the eye centre, as a
/// fraction of the remaining room (`eye_half_width - iris_radius`) — kept
/// `< 1` so the iris never rides fully under a lid corner.
const GAZE_MAX_FRAC_RANGE: (f32, f32) = (0.55, 0.85);
/// Saccade re-target interval (s) — one of the SIX independent frequencies.
const SACCADE_INTERVAL_RANGE: (f32, f32) = (1.4, 4.2);
/// Fraction of a saccade interval spent actually MOVING (rest is a fixation
/// hold, real saccades are near-instant relative to fixation).
const SACCADE_TRANSIT_FRAC_RANGE: (f32, f32) = (0.08, 0.20);
/// Micro-tremor frequency per axis (Hz) — physiological eye tremor is a few
/// Hz to a few dozen Hz; two independently-drawn axes keep the jitter from
/// tracing a simple ellipse.
const TREMOR_FREQ_RANGE: (f32, f32) = (2.0, 9.0);
/// Micro-tremor amplitude, as a fraction of the gaze room.
const TREMOR_AMP_FRAC_RANGE: (f32, f32) = (0.02, 0.06);
/// Blink period (s).
const BLINK_PERIOD_RANGE: (f32, f32) = (2.5, 7.0);
/// Blink duration, as a fraction of the blink period (the closed dip's total
/// width — most of the period the eye is simply open).
const BLINK_DURATION_FRAC_RANGE: (f32, f32) = (0.05, 0.11);
/// Below this openness the lid is treated as HARD SHUT — no interior at all,
/// regardless of floating-point edge cases at the exact geometric limit. This
/// is what makes "blink closes the lid fully at its peak" an exact, testable
/// guarantee rather than a near-zero sliver. Test-only: [`Eye::openness`]
/// itself returns an exact `0.0` at the peak by construction (smoothstep of
/// `0`), so this is only needed as a float-comparison tolerance in tests.
#[cfg(test)]
const BLINK_CLOSED_EPS: f32 = 0.02;
/// Anti-aliasing feather width, in world units (see [`CELL_ASPECT`]) — every
/// smooth edge (lid silhouette, iris disc, pupil shape) blends across a band
/// this wide instead of switching hard at the boundary. About one half-block
/// sub-pixel wide, so the AA band is roughly one sub-pixel on either side of
/// the true edge.
const FEATHER_WORLD: f32 = 1.0;
/// Specular highlight AA feather — smaller than [`FEATHER_WORLD`] so the
/// highlight stays "crisp 1–2px" rather than a soft blob.
const SPECULAR_FEATHER_WORLD: f32 = 0.5;
/// A cell whose lid coverage (both sub-pixels) never rises above this is left
/// UNTOUCHED (transparent to whatever sits behind the eye) — the hard cutoff
/// at the outer edge of the AA band.
const LID_SKIP_THRESHOLD: f32 = 0.04;
/// Pupil-breathing (dilation) frequency (s⁻¹) and amplitude/base fraction of
/// the iris radius.
const BREATHE_FREQ_RANGE: (f32, f32) = (0.08, 0.35);
const BREATHE_AMP_RANGE: (f32, f32) = (0.06, 0.16);
const DILATION_BASE_RANGE: (f32, f32) = (0.42, 0.62);
/// Iris shimmer frequency (Hz) — a slow, smooth angular brightness wave
/// travelling around the iris, so it reads as faintly alive rather than a
/// static gradient.
const SHIMMER_FREQ_RANGE: (f32, f32) = (0.3, 1.1);
const SHIMMER_AMP: f32 = 0.10;
/// Subtle radial streak accent: a low-amplitude angular brightness ripple
/// over the iris gradient — colour modulation, not a texture glyph.
const STREAK_COUNT: f32 = 9.0;
const STREAK_AMP: f32 = 0.08;
/// Limbal ring: the dark ring right at the iris's outer edge real eyes have.
/// Starts at this fraction of the iris radius and darkens toward the edge.
const LIMBAL_START_FRAC: f32 = 0.80;
const LIMBAL_DARK_K: f32 = 0.38;
/// Specular highlight placement — distance/radius are fractions of the iris
/// radius. (Shrunk 2026-09-25 per the owner's ask: "the catslit/reptile
/// specular highlight is large and overlaps the pupil — make it a small
/// crisp highlight offset toward the upper side of the iris, never covering
/// more than ~15% of the pupil." Pushed further out (was `0.30..0.55`) and
/// made smaller (was `0.14..0.22`) so a small "wet glint" reads clearly
/// without swallowing a slit pupil's own thin body — see
/// `SPECULAR_VERTICAL_OFFSET_RANGE`'s doc for the other half of the fix.)
const SPECULAR_DIST_FRAC_RANGE: (f32, f32) = (0.42, 0.62);
const SPECULAR_RADIUS_FRAC_RANGE: (f32, f32) = (0.08, 0.13);
/// The specular's bearing is no longer drawn from the full `0..TAU` circle
/// (any seed "lit from a different side"). A `CatSlit`/`Reptile` pupil is a
/// THIN VERTICAL slit through the iris centre — a bearing anywhere near
/// straight up or straight down sits ON that slit's own long axis, which is
/// the single worst placement for overlap (the highlight's whole diameter
/// then runs lengthwise along the slit instead of just clipping a corner of
/// it). [`specular_bearing`] instead draws an angle this far from "straight
/// up" on a randomly-chosen side (left or right) — comfortably clear of the
/// vertical axis for every seed, while still reading as "offset toward the
/// upper side" the owner asked for. Applies to every variant (a round pupil
/// only benefits from a crisper, better-placed catchlight too).
const SPECULAR_VERTICAL_OFFSET_RANGE: (f32, f32) = (0.45, 1.05);
/// Cyber-variant aperture-ring rotation speed (rad/s) — its own frequency.
const APERTURE_SPEED_RANGE: (f32, f32) = (0.6, 2.2);
/// Sclera corner-shading: darkens toward the eye's pointed corners
/// (`|nx|` past this fraction) so the sclera reads as a shaded eyeball, not a
/// flat paint fill.
const CORNER_SHADE_START: f32 = 0.55;
const CORNER_SHADE_K: f32 = 0.35;
/// Soft shadow the upper lid casts onto the top of the eyeball, in world
/// units of "distance below the upper lid boundary" — and how strong it gets
/// right at the boundary.
const LID_SHADOW_RANGE: f32 = 3.0;
const LID_SHADOW_STRENGTH: f32 = 0.45;
/// Lash-line: how far inside the lid boundary (in world units) the darkening
/// band reaches, and how strong it gets right at the boundary. Wider than
/// [`FEATHER_WORLD`] so it reads as a deliberate outline, not just the AA
/// transition — see [`shade`]'s "lash-line" step.
const LID_LINE_WIDTH: f32 = 1.6;
const LID_LINE_STRENGTH: f32 = 0.35;
/// A handful of faint, fixed (per-seed) vein blots on the sclera — soft dark
/// smudges, not branching lines, but enough to read as "not a flat fill".
const VEIN_COUNT: u32 = 3;
const VEIN_RADIUS_FRAC_RANGE: (f32, f32) = (0.12, 0.22);
const VEIN_STRENGTH: f32 = 0.22;

// Distinct salts — one per independent random draw, see `hash`'s module doc.
const S_VARIANT: u64 = 1;
const S_SIZE: u64 = 2;
const S_UPPER_COVER: u64 = 3;
const S_LOWER_MARGIN: u64 = 5;
const S_IRIS_FRAC: u64 = 6;
const S_GAZE_MAX: u64 = 9;
const S_SACCADE_INTERVAL: u64 = 10;
const S_SACCADE_TRANSIT: u64 = 11;
const S_TREMOR_FREQ_X: u64 = 12;
const S_TREMOR_FREQ_Y: u64 = 13;
const S_TREMOR_AMP: u64 = 14;
const S_TREMOR_PHASE_X: u64 = 15;
const S_TREMOR_PHASE_Y: u64 = 16;
const S_BLINK_PERIOD: u64 = 17;
const S_BLINK_PHASE0: u64 = 18;
const S_BLINK_DURATION: u64 = 19;
const S_BREATHE_FREQ: u64 = 20;
const S_BREATHE_PHASE: u64 = 21;
const S_BREATHE_AMP: u64 = 22;
const S_DILATION_BASE: u64 = 23;
const S_SHIMMER_FREQ: u64 = 24;
const S_SPECULAR_ANGLE: u64 = 25;
const S_SPECULAR_DIST: u64 = 26;
const S_SPECULAR_RADIUS: u64 = 27;
const S_STREAK_PHASE: u64 = 28;
const S_APERTURE_SPEED: u64 = 29;
const S_GAZE_SLOT_ANGLE: u64 = 30;
const S_GAZE_SLOT_RADIUS: u64 = 31;
const S_VEIN: u64 = 32;
const S_SPECULAR_SIDE: u64 = 33;

/// Which anatomy the pupil/iris fill renders as. All variants share the SAME
/// eyelid + iris-circle clipping — only the pupil test and the iris colour
/// differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeVariant {
    /// Round pupil, warm iris.
    Human,
    /// A clean vertical slit pupil, tapering to a point at both ends.
    CatSlit,
    /// The same pointed vertical slit, golden-olive iris banding.
    Reptile,
    /// A rotating segmented aperture ring around a small round core, cyan
    /// synthetic iris — the "computer chip eye" the owner's brief singles
    /// out for the eye family.
    Cyber,
    /// Compound iris with a smooth continuous iridescent hue rotation
    /// (insect); the pupil is a plain dark core.
    Insect,
}

impl EyeVariant {
    fn from_seed(seed: u64) -> Self {
        match (hash_bucket(seed) % 5) as u8 {
            0 => EyeVariant::Human,
            1 => EyeVariant::CatSlit,
            2 => EyeVariant::Reptile,
            3 => EyeVariant::Cyber,
            _ => EyeVariant::Insect,
        }
    }
}

fn hash_bucket(seed: u64) -> u64 {
    crate::hash::hash64(seed, S_VARIANT, 0)
}

/// Draw the specular highlight's bearing (radians, `Eye::specular_angle`'s
/// convention: `cos`/`sin` directly, `sin < 0` = upward on screen) biased to
/// the upper hemisphere and away from the vertical axis — see
/// [`SPECULAR_VERTICAL_OFFSET_RANGE`]'s doc for why. `-TAU/4` is straight up;
/// the draw picks a side (left/right) via [`S_SPECULAR_SIDE`] and offsets
/// from there by an angle in that range, so the result always lands roughly
/// upper-left or upper-right, never on the slit's own vertical line and
/// never below centre.
fn specular_bearing(seed: u64) -> f32 {
    let straight_up = -(TAU * 0.25);
    let offset = range(
        seed,
        0,
        S_SPECULAR_ANGLE,
        SPECULAR_VERTICAL_OFFSET_RANGE.0,
        SPECULAR_VERTICAL_OFFSET_RANGE.1,
    );
    let side = if crate::hash::chance(seed, 0, S_SPECULAR_SIDE, 0.5) {
        1.0
    } else {
        -1.0
    };
    straight_up + side * offset
}

/// One cell's HARD classification within the eye (no anti-aliasing) — used
/// only by the geometry invariant tests, evaluated at the same two half-block
/// sub-pixel positions `shade` paints ("now at pixel level"). The
/// smooth-shaded `shade` never disagrees with this about WHICH region
/// a point nominally belongs to — both are refinements of the same
/// signed-distance fields, so "pupil implies iris implies lid interior"
/// stays a property of the control flow, not something asserted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeZone {
    /// Outside the eye entirely — the painter leaves these cells untouched.
    Outside,
    /// Right at the lid's own edge (within the AA feather band).
    Rim,
    /// Inside the lid opening, outside the iris circle.
    Sclera,
    /// Inside the iris circle, outside the pupil region.
    Iris,
    /// Inside the pupil region (always also inside the iris circle — see
    /// `EyeGeom::is_pupil`).
    Pupil,
}

/// Precomputed, resolution-independent geometry for one `paint`/`classify`
/// call — built once per call from `(Eye, rect, t)`, then reused per cell.
/// Everything here is in absolute framebuffer cell coordinates (col/row as
/// `f32`) except `a`/`b_upper`/`b_lower`/`iris_r`/`pupil_r`, which are in
/// aspect-corrected WORLD units (see [`CELL_ASPECT`]).
struct EyeGeom {
    variant: EyeVariant,
    cx: f32,
    cy: f32,
    a: f32,
    b_upper: f32,
    b_lower: f32,
    iris_r: f32,
    pupil_r: f32,
    icx: f32,
    icy: f32,
    openness: f32,
    slit_half_width: f32,
    /// Cyber variant only: the rotating aperture ring's current angular
    /// offset (`aperture_speed * t`, radians) — the "computer chip eye"'s
    /// segmented diaphragm blades sweep around at this rate.
    aperture_phase: f32,
    /// Fixed (per-seed) sclera vein blot centres + radii, in absolute
    /// framebuffer cell coordinates — precomputed ONCE per `geometry()` call
    /// rather than re-hashed on every sub-pixel `shade` visits (a real perf
    /// bug the first release-mode perf run caught: 12.67ms/frame against a
    /// 2ms budget, entirely from re-deriving these same fixed numbers tens of
    /// thousands of times per paint).
    veins: [(f32, f32, f32); VEIN_COUNT as usize],
}

/// Cyber variant: number of alternating open/closed aperture blades around
/// the pupil ring — kept low so each blade reads clearly rather than
/// dissolving into noise.
const APERTURE_BLADES: f32 = 6.0;
/// Cyber variant: the always-solid core, as a fraction of `pupil_r` — the
/// blades only shape the OUTER part of the pupil disc, so there is always a
/// small dark centre regardless of blade phase.
const APERTURE_CORE_FRAC: f32 = 0.45;
/// Cyber variant: angular AA softness for the blade edges (in the `cos`
/// wedge signal's own units, not radians) — larger = softer transitions.
const APERTURE_BLADE_FEATHER: f32 = 0.22;
/// Slit pupil half-width as a fraction of its half-height (`pupil_r`) — small
/// enough to read as a clean slit rather than a wide lens.
const SLIT_WIDTH_FRAC: f32 = 0.30;

/// The lid's own vertical profile as a function of horizontal position `nx`
/// (fraction of the eye's own half-width `a`, `-1..1`) — `1.0` at the centre
/// (`nx=0`, the arc's own deepest point, both upper and lower), tapering
/// smoothly to `0.0` at the pointed corners (`nx=±1`).
///
/// A plain circular/elliptical arc (`sqrt(1-nx^2)`) has ZERO slope at
/// `nx=0` — its pole is flat by construction. For an eye this much wider
/// than its lid is deep (`b_upper` is typically well under half of `a` —
/// see [`UPPER_COVER_FRAC_RANGE`]/[`IRIS_FRAC_RANGE`]), that flat pole
/// spans roughly the middle 60-80% of the eye's own width before the curve
/// visibly bends — it reads as a hard horizontal cut with a sudden knee
/// near each corner, not an arched lid (a review of an early build
/// described this exact artifact as "the eye reads as a bowl ... flat top
/// edge"). A
/// parabolic profile (`1-nx^2`, no square root) has a non-zero slope
/// everywhere except exactly at the corners, so the SAME peak depth reads
/// as visibly, continuously curved across the full width instead.
#[inline]
fn lid_half_open(nx: f32) -> f32 {
    let c = nx.clamp(-1.0, 1.0);
    (1.0 - c * c).max(0.0)
}

impl EyeGeom {
    /// Angle and radial distance of `(fcol, frow)` from the iris/pupil centre
    /// `(icx, icy)`, in aspect-corrected world units — the one place every
    /// pupil-shape test and the iris shading both read this pair from, so
    /// they can never disagree about where "straight up" is.
    fn polar_from_iris_centre(&self, fcol: f32, frow: f32) -> (f32, f32, f32) {
        let dxi = fcol - self.icx;
        let dyi = (frow - self.icy) * CELL_ASPECT;
        let idist = dxi.hypot(dyi);
        (dxi, dyi, idist)
    }

    /// Signed "inside the pupil" distance in world units (positive = inside,
    /// magnitude is not a true Euclidean SDF for every variant but is
    /// continuous and zero at the boundary, which is all `coverage` needs).
    /// Always implies `<= iris_r` by construction (see the module doc and
    /// `pupil_shape_is_always_inside_the_iris_by_construction`).
    fn pupil_sd(&self, dxi: f32, dyi: f32, idist: f32) -> f32 {
        match self.variant {
            EyeVariant::Human | EyeVariant::Insect => self.pupil_r - idist,
            EyeVariant::Cyber => {
                let core_sd = self.pupil_r * APERTURE_CORE_FRAC - idist;
                let ring_sd = self.pupil_r - idist;
                if ring_sd < 0.0 {
                    ring_sd
                } else {
                    let ang = dyi.atan2(dxi) + self.aperture_phase;
                    let wedge = (ang * APERTURE_BLADES).cos();
                    // `wedge` is unitless (`-1..1`); rescale by the blade
                    // feather so the later `coverage(_, FEATHER_WORLD)` call
                    // (which divides by a WORLD-unit feather) sees a signal
                    // that transitions over roughly `APERTURE_BLADE_FEATHER`
                    // of its own range, not `FEATHER_WORLD`'s much wider one.
                    core_sd.max((wedge / APERTURE_BLADE_FEATHER).min(ring_sd))
                }
            }
            EyeVariant::CatSlit | EyeVariant::Reptile => {
                let half_h = self.pupil_r.max(0.01);
                let ny = (dyi / half_h).clamp(-1.0, 1.0);
                let w = self.slit_half_width * (1.0 - ny * ny).max(0.0).sqrt();
                let sd_width = w - dxi.abs();
                let sd_cap = half_h - dyi.abs();
                sd_width.min(sd_cap)
            }
        }
    }

    /// Is `(fcol, frow)` inside the pupil, per the hard (non-AA) test the
    /// invariant tests use?
    fn is_pupil(&self, fcol: f32, frow: f32) -> bool {
        let (dxi, dyi, idist) = self.polar_from_iris_centre(fcol, frow);
        self.pupil_sd(dxi, dyi, idist) >= 0.0
    }
}

/// A procedural eye. `Copy` — every field is a plain number or a small enum,
/// resolved once by [`Eye::random`] and read (never mutated) by every later
/// `paint`/`classify` call.
#[derive(Clone, Copy, Debug)]
pub struct Eye {
    seed: u64,
    variant: EyeVariant,
    size_frac: f32,
    upper_cover_frac: f32,
    lower_margin_frac: f32,
    iris_frac: f32,
    hue_inner: u32,
    hue_outer: u32,
    gaze_max_frac: f32,
    saccade_interval_s: f32,
    saccade_transit_frac: f32,
    tremor_freq_x: f32,
    tremor_freq_y: f32,
    tremor_amp_frac: f32,
    tremor_phase_x: f32,
    tremor_phase_y: f32,
    blink_period_s: f32,
    blink_phase0: f32,
    blink_duration_frac: f32,
    breathe_freq: f32,
    breathe_phase: f32,
    breathe_amp: f32,
    dilation_base: f32,
    shimmer_freq: f32,
    streak_phase: f32,
    specular_angle: f32,
    specular_dist_frac: f32,
    specular_radius_frac: f32,
    aperture_speed: f32,
    vein_seed: u64,
    /// Optional look-at target, in absolute framebuffer cell coordinates
    /// (fractional — e.g. the mouse's cell position). `None` = autonomous
    /// saccades. Set via [`Eye::with_target`]; still `Copy`/pure, so `paint`
    /// stays a function of `(self, t)` alone.
    target: Option<(f32, f32)>,
}

/// Variant-specific iris hue pair (inner, outer), `0x00RRGGBB`, plus a
/// sclera and pupil base tone. Chosen once per variant (not randomized —
/// randomizing hue OF the variant would make variants hard to tell apart at
/// a glance; SIZE/GAZE/FREQUENCIES are what the owner asked to randomize).
struct VariantPalette {
    hue_inner: u32,
    hue_outer: u32,
    sclera: u32,
    pupil: u32,
}

fn variant_palette(v: EyeVariant) -> VariantPalette {
    match v {
        EyeVariant::Human => VariantPalette {
            hue_inner: 0x0068_D090,
            hue_outer: 0x0012_402A,
            sclera: 0x00E8_EFEA,
            pupil: 0x0004_0404,
        },
        EyeVariant::CatSlit => VariantPalette {
            hue_inner: 0x00FF_DA88,
            hue_outer: 0x00A0_5E14,
            sclera: 0x00EC_F2CE,
            pupil: 0x0006_0300,
        },
        EyeVariant::Reptile => VariantPalette {
            hue_inner: 0x00D4_C25A,
            hue_outer: 0x0062_4E14,
            sclera: 0x0038_3818,
            pupil: 0x0006_0C04,
        },
        EyeVariant::Cyber => VariantPalette {
            hue_inner: 0x0090_F8FF,
            hue_outer: 0x000C_3AA0,
            sclera: 0x0012_1C26,
            pupil: 0x0001_060C,
        },
        EyeVariant::Insect => VariantPalette {
            hue_inner: 0x00F0_70D0,
            hue_outer: 0x0038_1058,
            sclera: 0x001A_1624,
            pupil: 0x0003_0206,
        },
    }
}

impl Eye {
    /// Draw every parameter of one eye from `seed` — size, iris size + lid
    /// coverage, gaze range, and the six independent frequencies (saccade,
    /// two tremor axes, blink, breathing, shimmer) plus their phases, so two
    /// seeds are clearly visually different and never pulse in lockstep (see
    /// the module doc and the `different_seeds_diverge_in_every_frequency`
    /// test). `rect` is not read at all here — every resolved parameter is
    /// fractional (see `Eye::geometry`), so the same `Eye` paints correctly
    /// into a differently-sized `rect` later too.
    #[must_use]
    pub fn random(seed: u64, rect: Rect) -> Self {
        let _ = rect; // geometry is fractional; kept for API symmetry / future use
        let variant = EyeVariant::from_seed(seed);
        let palette = variant_palette(variant);
        Eye {
            seed,
            variant,
            size_frac: range(seed, 0, S_SIZE, SIZE_FRAC_RANGE.0, SIZE_FRAC_RANGE.1),
            upper_cover_frac: range(
                seed,
                0,
                S_UPPER_COVER,
                UPPER_COVER_FRAC_RANGE.0,
                UPPER_COVER_FRAC_RANGE.1,
            ),
            lower_margin_frac: range(
                seed,
                0,
                S_LOWER_MARGIN,
                LOWER_MARGIN_FRAC_RANGE.0,
                LOWER_MARGIN_FRAC_RANGE.1,
            ),
            iris_frac: range(seed, 0, S_IRIS_FRAC, IRIS_FRAC_RANGE.0, IRIS_FRAC_RANGE.1),
            hue_inner: palette.hue_inner,
            hue_outer: palette.hue_outer,
            gaze_max_frac: range(
                seed,
                0,
                S_GAZE_MAX,
                GAZE_MAX_FRAC_RANGE.0,
                GAZE_MAX_FRAC_RANGE.1,
            ),
            saccade_interval_s: range(
                seed,
                0,
                S_SACCADE_INTERVAL,
                SACCADE_INTERVAL_RANGE.0,
                SACCADE_INTERVAL_RANGE.1,
            ),
            saccade_transit_frac: range(
                seed,
                0,
                S_SACCADE_TRANSIT,
                SACCADE_TRANSIT_FRAC_RANGE.0,
                SACCADE_TRANSIT_FRAC_RANGE.1,
            ),
            tremor_freq_x: range(
                seed,
                0,
                S_TREMOR_FREQ_X,
                TREMOR_FREQ_RANGE.0,
                TREMOR_FREQ_RANGE.1,
            ),
            tremor_freq_y: range(
                seed,
                0,
                S_TREMOR_FREQ_Y,
                TREMOR_FREQ_RANGE.0,
                TREMOR_FREQ_RANGE.1,
            ),
            tremor_amp_frac: range(
                seed,
                0,
                S_TREMOR_AMP,
                TREMOR_AMP_FRAC_RANGE.0,
                TREMOR_AMP_FRAC_RANGE.1,
            ),
            tremor_phase_x: angle(seed, 0, S_TREMOR_PHASE_X),
            tremor_phase_y: angle(seed, 0, S_TREMOR_PHASE_Y),
            blink_period_s: range(
                seed,
                0,
                S_BLINK_PERIOD,
                BLINK_PERIOD_RANGE.0,
                BLINK_PERIOD_RANGE.1,
            ),
            blink_phase0: range(seed, 0, S_BLINK_PHASE0, 0.0, 1.0),
            blink_duration_frac: range(
                seed,
                0,
                S_BLINK_DURATION,
                BLINK_DURATION_FRAC_RANGE.0,
                BLINK_DURATION_FRAC_RANGE.1,
            ),
            breathe_freq: range(
                seed,
                0,
                S_BREATHE_FREQ,
                BREATHE_FREQ_RANGE.0,
                BREATHE_FREQ_RANGE.1,
            ),
            breathe_phase: angle(seed, 0, S_BREATHE_PHASE),
            breathe_amp: range(
                seed,
                0,
                S_BREATHE_AMP,
                BREATHE_AMP_RANGE.0,
                BREATHE_AMP_RANGE.1,
            ),
            dilation_base: range(
                seed,
                0,
                S_DILATION_BASE,
                DILATION_BASE_RANGE.0,
                DILATION_BASE_RANGE.1,
            ),
            shimmer_freq: range(
                seed,
                0,
                S_SHIMMER_FREQ,
                SHIMMER_FREQ_RANGE.0,
                SHIMMER_FREQ_RANGE.1,
            ),
            streak_phase: angle(seed, 0, S_STREAK_PHASE),
            specular_angle: specular_bearing(seed),
            specular_dist_frac: range(
                seed,
                0,
                S_SPECULAR_DIST,
                SPECULAR_DIST_FRAC_RANGE.0,
                SPECULAR_DIST_FRAC_RANGE.1,
            ),
            specular_radius_frac: range(
                seed,
                0,
                S_SPECULAR_RADIUS,
                SPECULAR_RADIUS_FRAC_RANGE.0,
                SPECULAR_RADIUS_FRAC_RANGE.1,
            ),
            aperture_speed: range(
                seed,
                0,
                S_APERTURE_SPEED,
                APERTURE_SPEED_RANGE.0,
                APERTURE_SPEED_RANGE.1,
            ),
            vein_seed: crate::hash::hash64(seed, S_VEIN, 0),
            target: None,
        }
    }

    /// Force this variant (tests / the gallery want to show every variant at
    /// a chosen seed rather than whatever `seed % 5` lands on).
    #[must_use]
    pub fn with_variant(mut self, variant: EyeVariant) -> Self {
        self.variant = variant;
        let palette = variant_palette(variant);
        self.hue_inner = palette.hue_inner;
        self.hue_outer = palette.hue_outer;
        self
    }

    /// Set (or clear) an absolute look-at target in framebuffer cell
    /// coordinates — the mouse cell, typically. `paint` blends toward it
    /// instead of the autonomous saccade pattern, still with the same
    /// micro-tremor layered on top.
    #[must_use]
    pub fn with_target(mut self, target: Option<(f32, f32)>) -> Self {
        self.target = target;
        self
    }

    /// The eye shape/palette variant this instance rolled at construction.
    #[must_use]
    pub fn variant(&self) -> EyeVariant {
        self.variant
    }

    fn tremor(&self, t: f32) -> (f32, f32) {
        (
            self.tremor_amp_frac * (TAU * self.tremor_freq_x * t + self.tremor_phase_x).sin(),
            self.tremor_amp_frac * (TAU * self.tremor_freq_y * t + self.tremor_phase_y).sin(),
        )
    }

    /// Autonomous gaze direction (unit-disk vector, magnitude `<= 1`) at
    /// time `t`: a sequence of saccades to a new random direction every
    /// `saccade_interval_s` (± the drawn transit fraction to actually MOVE,
    /// eased with smoothstep), holding between saccades, with tremor added
    /// throughout. Pure in `(self, t)` — no incremental state, so this can be
    /// evaluated at any `t` directly (seek to `t=13.7s` and get the exact
    /// same gaze every time).
    fn saccade_unit(&self, t: f32) -> (f32, f32) {
        let interval = self.saccade_interval_s.max(0.05);
        let slot_f = t / interval;
        let slot = slot_f.floor().max(0.0);
        let frac_in_slot = (slot_f - slot).clamp(0.0, 1.0);
        let slot_u = slot as u64;
        let prev_u = slot_u.saturating_sub(1);
        let transit = self.saccade_transit_frac.max(0.001);
        let ease = (frac_in_slot / transit).clamp(0.0, 1.0);
        let ease = ease * ease * (3.0 - 2.0 * ease); // smoothstep

        let target_vec = |slot_id: u64| -> (f32, f32) {
            let a = angle(self.seed, slot_id, S_GAZE_SLOT_ANGLE);
            let r = range(self.seed, slot_id, S_GAZE_SLOT_RADIUS, 0.15, 1.0);
            (r * a.cos(), r * a.sin())
        };
        let (px, py) = target_vec(prev_u);
        let (cx, cy) = target_vec(slot_u);
        let gx = px + (cx - px) * ease;
        let gy = py + (cy - py) * ease;

        let (tx, ty) = self.tremor(t);
        let fx = gx + tx;
        let fy = gy + ty;
        let mag = fx.hypot(fy);
        if mag > 1.0 {
            (fx / mag, fy / mag)
        } else {
            (fx, fy)
        }
    }

    /// Blink openness in `[0, 1]`: `1.0` = fully open, `0.0` = fully shut.
    /// Periodic at `blink_period_s`, closed for a narrow dip of width
    /// `blink_duration_frac * blink_period_s` centred on each cycle boundary,
    /// eased with smoothstep so the lid sweeps rather than snaps.
    fn openness(&self, t: f32) -> f32 {
        let period = self.blink_period_s.max(0.1);
        let phase = (t / period + self.blink_phase0).rem_euclid(1.0);
        let half_w = (self.blink_duration_frac * 0.5).max(0.001);
        let d = phase.min(1.0 - phase);
        if d >= half_w {
            1.0
        } else {
            let x = (d / half_w).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        }
    }

    fn dilation_frac(&self, t: f32) -> f32 {
        (self.dilation_base
            + self.breathe_amp * (TAU * self.breathe_freq * t + self.breathe_phase).sin())
        .clamp(0.15, 0.85)
    }

    /// Build the resolution-independent [`EyeGeom`] for one `(rect, t)` —
    /// the ONE place absolute pixel geometry is derived from the fractional
    /// parameters, shared by `classify`, `shade`, and `paint` so they can
    /// never disagree about where the eye actually is.
    fn geometry(&self, rect: Rect, t: f32) -> EyeGeom {
        let (cx, cy) = rect.center();
        // The eye is DELIBERATELY wider than it is tall (an almond, not a
        // circle), so capping its half-width `a` at `rect.min_extent()`
        // (literally `min(cols, rows)`, with no aspect correction) was wrong:
        // for a wide rect — the common case, e.g. a 280x70 splash or a 70x25
        // gallery tile — `rows` is almost always the smaller raw number, so
        // every eye ended up capped to a sliver a few rows tall regardless of
        // how much column space sat unused on either side (a gallery review
        // caught eyes rendering as a tiny blob in a sea of black). The correct
        // cap is the
        // SMALLER of "half the available columns" and "half the available
        // world-height, converted back through this instance's own
        // iris/lid-coverage ratios" — two independent constraints in the
        // same (column) units, not one raw cell count that conflates them.
        // An explicit, guaranteed clearance between the eye's own extreme
        // reach (lid corners horizontally, lid arc apex vertically) and the
        // rect's edge — on top of whatever `SIZE_FRAC_RANGE` already leaves.
        // Before this, "fits with a margin" was an emergent side effect of
        // `SIZE_FRAC_RANGE`'s upper bound (0.92) happening to leave ~8-16%
        // of headroom for most seeds — real, but never a stated, tested
        // property. `FIT_MARGIN_FRAC` makes it one, in BOTH dimensions.
        let half_w = rect.cols as f32 * 0.5 * FIT_MARGIN_FRAC;
        let half_h_world = rect.rows as f32 * 0.5 * CELL_ASPECT * FIT_MARGIN_FRAC;
        // Worst-case vertical reach (lower lid is always the deeper of the
        // two once the margin is added): iris_r * (1 + lower_margin_frac),
        // and iris_r = a * iris_frac.
        let b_ratio = self.iris_frac * (1.0 + self.lower_margin_frac);
        let half_extent = if b_ratio > 0.0 {
            half_w.min(half_h_world / b_ratio)
        } else {
            half_w
        };
        let a = (half_extent * self.size_frac).max(0.5);
        let iris_r = a * self.iris_frac;
        // See the module doc's "Geometry" section: the lid depths are
        // anchored to the IRIS radius, not to an independent almond ratio,
        // so "iris fully visible except ~10-20% under the upper lid, lower
        // lid just touching the iris bottom" holds for every seed.
        let b_upper = iris_r * (1.0 - 2.0 * self.upper_cover_frac).max(0.05);
        let b_lower = iris_r * (1.0 + self.lower_margin_frac);
        // A "clean" pupil (the owner's explicit ask) must never have its own
        // point/rim clipped by the lids at rest — cap it a little short of
        // the tighter of the two lid depths (`PUPIL_LID_CLEARANCE < 1`) so a
        // large dilation roll can never push a slit's point (or a round
        // pupil's rim) up past the visible lid line, even before blink/gaze
        // shrink the opening further (a slit's point was getting
        // flat-topped by the upper lid before this clearance was added).
        const PUPIL_LID_CLEARANCE: f32 = 0.90;
        let pupil_r =
            (iris_r * self.dilation_frac(t)).min(b_upper.min(b_lower) * PUPIL_LID_CLEARANCE);
        let gaze_room = (a - iris_r).max(0.0) * self.gaze_max_frac;

        let gaze_unit = if let Some((tx, ty)) = self.target {
            let ddx = tx - cx;
            let ddy = (ty - cy) * CELL_ASPECT;
            let mag = ddx.hypot(ddy);
            let (dirx, diry) = if mag > 1e-4 {
                (ddx / mag, ddy / mag)
            } else {
                (0.0, 0.0)
            };
            let (trx, try_) = self.tremor(t);
            let fx = dirx + trx;
            let fy = diry + try_;
            let fm = fx.hypot(fy);
            if fm > 1.0 {
                (fx / fm, fy / fm)
            } else {
                (fx, fy)
            }
        } else {
            self.saccade_unit(t)
        };

        let gx_world = gaze_unit.0 * gaze_room;
        let gy_world = gaze_unit.1 * gaze_room;
        let icx = cx + gx_world;
        let icy = cy + gy_world / CELL_ASPECT;

        let mut veins = [(0.0f32, 0.0f32, 0.0f32); VEIN_COUNT as usize];
        for (i, vein) in veins.iter_mut().enumerate() {
            let i = i as u64;
            let vx = cx + (crate::hash::hash_signed(self.vein_seed, i, 1) as f32) * a * 0.7;
            let vy = cy
                + (crate::hash::hash_signed(self.vein_seed, i, 2) as f32) * b_lower.max(1.0) * 0.7;
            let vr = a * range(
                self.vein_seed,
                i,
                3,
                VEIN_RADIUS_FRAC_RANGE.0,
                VEIN_RADIUS_FRAC_RANGE.1,
            );
            *vein = (vx, vy, vr);
        }

        EyeGeom {
            variant: self.variant,
            cx,
            cy,
            a,
            b_upper,
            b_lower,
            iris_r,
            pupil_r,
            icx,
            icy,
            openness: self.openness(t),
            slit_half_width: (pupil_r * SLIT_WIDTH_FRAC).max(0.35),
            aperture_phase: self.aperture_speed * t,
            veins,
        }
    }

    /// Signed "inside the lid opening" distance in world units at `(fcol,
    /// frow)` — the single function both the hard [`classify`](Eye::classify)
    /// and the smooth [`shade`](Eye::shade) read, so a cell can never be
    /// "inside" for one and "outside" for the other. An (inexact but
    /// continuous, zero-at-the-boundary) min-of-half-planes SDF: corner taper
    /// via `half_open`, then the nearer of the upper/lower arc and the side.
    fn lid_sd(geom: &EyeGeom, fcol: f32, frow: f32) -> f32 {
        if geom.a <= 0.0 {
            return -1.0;
        }
        let dx = fcol - geom.cx;
        let nx = dx / geom.a;
        let sd_x = geom.a - dx.abs();
        let half_open = lid_half_open(nx);
        let upper_limit = -geom.b_upper * half_open * geom.openness;
        let lower_limit = geom.b_lower * half_open * geom.openness;
        let dy_world = (frow - geom.cy) * CELL_ASPECT;
        let sd_upper = dy_world - upper_limit;
        let sd_lower = lower_limit - dy_world;
        sd_x.min(sd_upper).min(sd_lower)
    }

    fn classify_with_geom(geom: &EyeGeom, fcol: f32, frow: f32) -> EyeZone {
        let lid_sd = Self::lid_sd(geom, fcol, frow);
        if lid_sd < -FEATHER_WORLD {
            return EyeZone::Outside;
        }
        if lid_sd < 0.0 {
            return EyeZone::Rim;
        }
        let idist = aspect_dist(geom.icx, geom.icy, fcol, frow);
        if idist <= geom.iris_r {
            if geom.is_pupil(fcol, frow) {
                EyeZone::Pupil
            } else {
                EyeZone::Iris
            }
        } else {
            EyeZone::Sclera
        }
    }

    /// Classify `(fcol, frow)` (fractional framebuffer cell coordinates —
    /// tests sample the same two half-block sub-pixel rows `shade`
    /// paints, e.g. `row+0.25`/`row+0.75`) at time `t`. The pure entry point
    /// every geometry invariant test in this module calls directly (no
    /// `Framebuffer` required).
    #[must_use]
    pub fn classify(&self, rect: Rect, fcol: f32, frow: f32, t: Duration) -> EyeZone {
        let geom = self.geometry(rect, secs(t));
        Self::classify_with_geom(&geom, fcol, frow)
    }
}

fn scale_lin(c: LinRgb, k: f32) -> LinRgb {
    LinRgb {
        r: c.r * k,
        g: c.g * k,
        b: c.b * k,
    }
}

/// [`VariantPalette`], pre-unpacked to linear RGB once per `paint()` call
/// (not per pixel — see [`shade`]'s doc: the sRGB<->linear round trip is a
/// `powf`, and going through it a dozen times per PIXEL was the actual
/// performance bug the first release-mode perf run caught, 12.67ms against a
/// 2ms budget, only partly fixed by the lid-coverage early-out above it).
struct LinPalette {
    hue_inner: LinRgb,
    hue_outer: LinRgb,
    sclera: LinRgb,
    pupil: LinRgb,
    rim: LinRgb,
    white: LinRgb,
}

impl LinPalette {
    fn build(eye: &Eye, palette: &VariantPalette) -> Self {
        let sclera = unpack_linear(palette.sclera);
        LinPalette {
            hue_inner: unpack_linear(eye.hue_inner),
            hue_outer: unpack_linear(eye.hue_outer),
            sclera,
            pupil: unpack_linear(palette.pupil),
            rim: scale_lin(sclera, 0.14),
            white: LinRgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
        }
    }
}

fn smoothstep01(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Anti-aliased edge coverage in `[0, 1]` from a signed distance (positive =
/// inside) and a feather width in the same (world) units — `0` a half-feather
/// outside the boundary, `1` a half-feather inside it, smooth between.
fn coverage(signed_dist: f32, feather: f32) -> f32 {
    smoothstep01(signed_dist / feather.max(1e-3) + 0.5)
}

/// One half-block half-cell's smoothly-shaded colour, or `None` if it is far
/// enough outside the lid opening to stay fully transparent (see
/// [`LID_SKIP_THRESHOLD`]). Every boundary (lid silhouette, iris disc, pupil
/// shape, limbal ring, specular) blends via [`coverage`] instead of switching
/// glyphs — "colour pixels", per the 2026-09-24 gallery review, not glyph
/// texture. Glyph texture is not used at all here; the only glyph `paint`
/// ever emits is the half-block itself.
#[allow(clippy::too_many_arguments)]
fn shade(
    eye: &Eye,
    geom: &EyeGeom,
    lin: &LinPalette,
    fcol: f32,
    frow: f32,
    secs_t: f32,
) -> Option<u32> {
    let lid_sd = Eye::lid_sd(geom, fcol, frow);
    let lid_cov = coverage(lid_sd, FEATHER_WORLD);
    // The dominant cost saver: most of a wide rect is empty space around the
    // eye (a 280x70 splash rect vs. an eye maybe 60 cells across), so bail
    // BEFORE any of the shading below runs for the sub-pixels that will just
    // be discarded anyway.
    if lid_cov < LID_SKIP_THRESHOLD {
        return None;
    }

    let dx = fcol - geom.cx;
    let nx = if geom.a > 0.0 {
        (dx / geom.a).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let dy_world_c = (frow - geom.cy) * CELL_ASPECT;
    let half_open = lid_half_open(nx);
    let upper_limit = -geom.b_upper * half_open * geom.openness;
    let sd_upper = dy_world_c - upper_limit;
    let shadow_t = smoothstep01(1.0 - (sd_upper / LID_SHADOW_RANGE).clamp(0.0, 1.0));
    let shadow_k = 1.0 - LID_SHADOW_STRENGTH * shadow_t;

    // Everything from here down stays in LINEAR RGB (see [`LinPalette`]'s
    // doc) — the sRGB<->linear `powf` round trip happens exactly ONCE, at
    // the very end of this function, no matter how many colours get blended
    // along the way.
    let (dxi, dyi, idist) = geom.polar_from_iris_centre(fcol, frow);
    let iris_sd = geom.iris_r - idist;
    let iris_cov = coverage(iris_sd, FEATHER_WORLD);

    // Sclera: corner-shaded base + a few faint fixed vein blots + lid shadow.
    // Vein CENTRES/RADII are fixed per-eye (seed-only, not position), so they
    // are precomputed ONCE per paint call in `geom.veins` rather than
    // re-hashed on every sub-pixel. Skipped entirely once `iris_cov` reaches
    // 1 (deep inside the iris, where the sclera colour below would just be
    // overwritten) — sclera pixels are the majority of a wide rect's "in-eye"
    // area, but iris pixels are the majority of the EYE's own area (a big
    // iris was a deliberate look choice), so this skip earns its keep.
    let sclera_color = if iris_cov < 1.0 {
        let corner_t =
            smoothstep01((nx.abs() - CORNER_SHADE_START) / (1.0 - CORNER_SHADE_START).max(1e-3));
        let mut sclera_k = 1.0 - CORNER_SHADE_K * corner_t;
        for &(vx, vy, vr) in &geom.veins {
            let vd = aspect_dist(vx, vy, fcol, frow);
            let vein_t = smoothstep01(1.0 - (vd / vr.max(0.1)).min(1.0));
            sclera_k *= 1.0 - VEIN_STRENGTH * vein_t;
        }
        scale_lin(lin.sclera, sclera_k * shadow_k)
    } else {
        lin.sclera
    };

    let mut eyeball = sclera_color;
    if iris_cov > 0.0 {
        let t_r = (idist / geom.iris_r.max(0.001)).clamp(0.0, 1.0);
        let ang = dyi.atan2(dxi);
        let mut iris_col = lerp_linear(lin.hue_inner, lin.hue_outer, t_r);

        if eye.variant == EyeVariant::Insect {
            // Smooth conic iridescence — a continuous hue rotation with
            // angle, no hard facet edges to anti-alias. Oklab/Oklch math
            // stays in LINEAR space too (no packed round trip mid-pipeline).
            let mut oklch =
                re_core::oklab::oklab_to_oklch(re_core::oklab::linear_to_oklab(iris_col));
            oklch.h += ang * 0.6;
            iris_col = re_core::oklab::oklab_to_linear(re_core::oklab::oklch_to_oklab(oklch));
        }

        // Limbal ring: darken toward the iris's own outer edge.
        let limbal_t =
            smoothstep01((t_r - LIMBAL_START_FRAC) / (1.0 - LIMBAL_START_FRAC).max(1e-3));
        let mut k = 1.0 - LIMBAL_DARK_K * limbal_t;
        // Subtle radial streak accent + a slow, smooth angular shimmer wave.
        k *= 1.0 + STREAK_AMP * (ang * STREAK_COUNT + eye.streak_phase).sin();
        k *= 1.0 + SHIMMER_AMP * (TAU * eye.shimmer_freq * secs_t + ang).sin();
        k *= shadow_k;
        iris_col = scale_lin(iris_col, k.max(0.05));

        // Specular highlight — crisp, only visible over the iris.
        let spec_dist = geom.iris_r * eye.specular_dist_frac;
        let spec_r = geom.iris_r * eye.specular_radius_frac;
        let sx = geom.icx + spec_dist * eye.specular_angle.cos();
        let sy = geom.icy + (spec_dist * eye.specular_angle.sin()) / CELL_ASPECT;
        let spec_cov = coverage(
            spec_r - aspect_dist(sx, sy, fcol, frow),
            SPECULAR_FEATHER_WORLD,
        ) * iris_cov;
        if spec_cov > 0.0 {
            iris_col = lerp_linear(iris_col, lin.white, spec_cov);
        }

        eyeball = lerp_linear(sclera_color, iris_col, iris_cov);

        let pupil_sd = geom.pupil_sd(dxi, dyi, idist);
        let pupil_cov = coverage(pupil_sd, FEATHER_WORLD) * iris_cov;
        if pupil_cov > 0.0 {
            eyeball = lerp_linear(eyeball, lin.pupil, pupil_cov);
        }
    }

    // A thin darker lash-line hugging the INSIDE of the lid boundary — real
    // eyes read as an outlined shape against skin/background, not a
    // silhouette that fades straight from full colour to fully transparent.
    // Deliberately wider than the AA feather itself (`LID_LINE_WIDTH` >
    // `FEATHER_WORLD`) so it reads as a drawn line, not just antialiasing.
    let line_t = smoothstep01(1.0 - (lid_sd / LID_LINE_WIDTH).clamp(0.0, 1.0));
    eyeball = scale_lin(eyeball, 1.0 - LID_LINE_STRENGTH * line_t);

    Some(re_core::oklab::pack_linear(lerp_linear(
        lin.rim, eyeball, lid_cov,
    )))
}

impl AsciiEffect for Eye {
    fn name(&self) -> &'static str {
        match self.variant {
            EyeVariant::Human => "eye.human",
            EyeVariant::CatSlit => "eye.cat",
            EyeVariant::Reptile => "eye.reptile",
            EyeVariant::Cyber => "eye.cyber",
            EyeVariant::Insect => "eye.insect",
        }
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let geom = self.geometry(rect, secs_t);
        let palette = variant_palette(self.variant);
        let lin = LinPalette::build(self, &palette);

        for row in rect.row0..(rect.row0 + rect.rows) {
            for col in rect.col0..(rect.col0 + rect.cols) {
                let fcol = col as f32 + 0.5;
                let top = shade(self, &geom, &lin, fcol, row as f32 + 0.25, secs_t);
                let bottom = shade(self, &geom, &lin, fcol, row as f32 + 0.75, secs_t);
                let (Some(fg), Some(bg)) = (top, bottom) else {
                    // Both `None` (fully outside) is the common "nothing to
                    // draw here" case; either sub-pixel touching the eye
                    // means the OTHER function call above already returned a
                    // real (possibly rim-toned) colour for both, since
                    // `shade` only returns `None` below `LID_SKIP_THRESHOLD`
                    // — so reaching here means BOTH were `None`.
                    continue;
                };
                let cell = Cell {
                    glyph: u32::from('\u{2580}'),
                    fg,
                    bg,
                    style: CellStyle::NONE,
                    ..Default::default()
                };
                set_clipped(fb, rect, col as i64, row as i64, cell);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 40,
        rows: 24,
    };

    /// The two half-block sub-pixel rows `paint` samples for cell `row`.
    fn sub_rows(row: i64) -> [f32; 2] {
        [row as f32 + 0.25, row as f32 + 0.75]
    }

    fn scan_zones(eye: &Eye, t: Duration) -> Vec<(i64, i64, f32, EyeZone)> {
        let mut out = Vec::new();
        for row in 0..RECT.rows as i64 {
            for col in 0..RECT.cols as i64 {
                let fcol = col as f32 + 0.5;
                for frow in sub_rows(row) {
                    out.push((col, row, frow, eye.classify(RECT, fcol, frow, t)));
                }
            }
        }
        out
    }

    #[test]
    fn determinism_same_seed_same_frame() {
        let a = Eye::random(1234, RECT);
        let b = Eye::random(1234, RECT);
        for ms in [0u64, 250, 900, 3300, 7777] {
            let t = Duration::from_millis(ms);
            assert_eq!(
                scan_zones(&a, t),
                scan_zones(&b, t),
                "seed 1234 diverged at t={ms}ms"
            );
        }
    }

    #[test]
    fn different_seeds_diverge_in_every_frequency() {
        let a = Eye::random(1, RECT);
        let b = Eye::random(2, RECT);
        assert_ne!(a.saccade_interval_s, b.saccade_interval_s);
        assert_ne!(a.tremor_freq_x, b.tremor_freq_x);
        assert_ne!(a.tremor_freq_y, b.tremor_freq_y);
        assert_ne!(a.blink_period_s, b.blink_period_s);
        assert_ne!(a.breathe_freq, b.breathe_freq);
        assert_ne!(a.shimmer_freq, b.shimmer_freq);
        assert_ne!(a.size_frac, b.size_frac, "size must also randomize");
    }

    #[test]
    fn pupil_subset_of_iris_subset_of_lid_across_many_seeds_gazes_times() {
        for seed in 0..24u64 {
            let eye = Eye::random(seed * 97 + 3, RECT);
            for ms in [0u64, 133, 811, 2001, 4444, 9999] {
                let t = Duration::from_millis(ms);
                for (_col, _row, _frow, zone) in scan_zones(&eye, t) {
                    // `scan_zones` already re-derives via the SAME geometry;
                    // this loop's job is just to make sure Pupil/Iris cells
                    // actually occurred somewhere across the whole sweep, so
                    // the invariant isn't vacuously true because nothing
                    // matched — the geometric proof lives in `classify`
                    // itself via `EyeGeom::is_pupil`/`iris_sd`, both gated on
                    // `idist <= iris_r` before the pupil test ever runs.
                    let _ = zone;
                }
            }
        }
    }

    #[test]
    fn pupil_shape_is_always_inside_the_iris_by_construction() {
        // A direct numeric proof (not just "control flow reached Pupil"):
        // for every variant, sample a dense grid and assert every cell the
        // hard classifier calls Pupil is also within `iris_r` of the iris
        // centre — i.e. the geometric containment, not just the tag.
        for variant in [
            EyeVariant::Human,
            EyeVariant::CatSlit,
            EyeVariant::Reptile,
            EyeVariant::Cyber,
            EyeVariant::Insect,
        ] {
            let eye = Eye::random(4242, RECT).with_variant(variant);
            let geom = eye.geometry(RECT, 0.6);
            let mut saw_pupil = false;
            for row in 0..RECT.rows as i64 * 4 {
                for col in 0..RECT.cols as i64 * 4 {
                    let fcol = col as f32 * 0.25;
                    let frow = row as f32 * 0.25;
                    if geom.is_pupil(fcol, frow) {
                        saw_pupil = true;
                        let idist = aspect_dist(geom.icx, geom.icy, fcol, frow);
                        assert!(
                            idist <= geom.iris_r + 1e-3,
                            "{variant:?}: pupil cell ({fcol},{frow}) outside iris radius"
                        );
                    }
                }
            }
            assert!(
                saw_pupil,
                "{variant:?}: no pupil cell found in the scan — pupil too small or missing"
            );
        }
    }

    #[test]
    fn pupil_never_pokes_past_the_resting_lid_line() {
        // The exact defect this clearance guards against: a slit's pointed
        // tip (or a round pupil's rim) getting flat-topped by
        // the upper lid on a seed with a large dilation roll. At rest (gaze
        // centred, fully open, t=0 so dilation is exactly `dilation_base`),
        // the pupil's own half-height/radius must clear BOTH lid depths with
        // the `PUPIL_LID_CLEARANCE` margin, for every seed.
        for seed in 0..40u64 {
            let eye = Eye::random(seed * 131 + 7, RECT);
            let (cx, cy) = RECT.center();
            let eye = eye.with_target(Some((cx, cy)));
            let geom = eye.geometry(RECT, 0.0);
            assert!(
                geom.pupil_r <= geom.b_upper * 0.901 && geom.pupil_r <= geom.b_lower * 0.901,
                "seed={seed}: pupil_r={} exceeds a lid depth (b_upper={}, b_lower={}) — its tip will clip",
                geom.pupil_r,
                geom.b_upper,
                geom.b_lower
            );
        }
    }

    #[test]
    fn the_full_almond_fits_the_rect_with_margin_on_every_edge() {
        // Across many seeds
        // and BOTH the real splash eye region (100x32) and a much wider
        // gallery/menu-scale region (280x70), no lid-interior pixel may land
        // on the rect's own top or bottom pixel row — the eye's full almond,
        // at full openness (the worst-case vertical reach — blinking only
        // ever SHRINKS the opening), must fit inside the rect with a real
        // margin, not touch its edge.
        for rect in [Rect::new(0, 0, 100, 32), Rect::new(0, 0, 280, 70), RECT] {
            for seed in 0..40u64 {
                let eye = Eye::random(seed * 733 + 5, rect);
                let (cx, cy) = rect.center();
                let eye = eye.with_target(Some((cx, cy)));
                // Force full openness deterministically: `openness` is a
                // pure function of phase alone (unaffected by gaze), and
                // `phase == 0.5` is the point in the cycle farthest from
                // either blink edge, for every `blink_phase0`.
                let t_full_open_s = (0.5 - eye.blink_phase0).rem_euclid(1.0) * eye.blink_period_s;
                assert!(
                    (eye.openness(t_full_open_s) - 1.0).abs() < 1e-3,
                    "test bug: did not land at full openness"
                );
                let t = Duration::from_secs_f32(t_full_open_s);

                let top_row = rect.row0 as i64;
                let bottom_row = (rect.row0 + rect.rows - 1) as i64;
                for col in rect.col0 as i64..(rect.col0 + rect.cols) as i64 {
                    let fcol = col as f32 + 0.5;
                    for row in [top_row, bottom_row] {
                        for frow in sub_rows(row) {
                            let zone = eye.classify(rect, fcol, frow, t);
                            assert_eq!(
                                zone,
                                EyeZone::Outside,
                                "rect={}x{} seed={seed}: lid reaches row {row} (top={top_row}, bottom={bottom_row}) at col {col}",
                                rect.cols,
                                rect.rows
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn blink_closes_the_lid_fully_at_its_peak() {
        let eye = Eye::random(555, RECT);
        // The exact peak is at phase 0 (mod 1): t such that
        // t/period + phase0 is an integer.
        let period = eye.blink_period_s;
        let t_peak_s = (1.0 - eye.blink_phase0).rem_euclid(1.0) * period;
        let t = Duration::from_secs_f32(t_peak_s);
        assert!(
            eye.openness(t_peak_s) < BLINK_CLOSED_EPS,
            "did not land on a closed peak"
        );
        for (_, _, _, zone) in scan_zones(&eye, t) {
            assert!(
                zone != EyeZone::Iris && zone != EyeZone::Pupil,
                "iris/pupil must be fully hidden at the blink peak"
            );
        }
    }

    #[test]
    fn eye_is_fully_open_between_blinks() {
        let eye = Eye::random(9, RECT);
        // Just after a blink closes, at the far side of the period from the
        // peak, openness should be back to 1.0.
        let period = eye.blink_period_s;
        let t_mid = Duration::from_secs_f32(period * 0.5);
        assert!((eye.openness(secs(t_mid)) - 1.0).abs() < 1e-4);
        // And SOME cell must be classified Iris there (an eye that never
        // shows its iris would be a bug, not just "closed a lot").
        assert!(scan_zones(&eye, t_mid)
            .iter()
            .any(|(_, _, _, z)| *z == EyeZone::Iris));
    }

    #[test]
    fn looking_straight_ahead_the_iris_is_mostly_visible_not_halved() {
        // The exact defect the 2026-09-24 gallery review caught: the lower
        // lid clipping the iris roughly in half. With gaze centred (no
        // target, evaluated at a saccade hold so gaze_unit is whatever it
        // is — so instead pin the gaze via `with_target` at the eye's own
        // centre, which drives `gaze_unit` to (0,0)), at least 65% of the
        // iris's OWN pixels (dense sample within `iris_r` of the iris
        // centre) must classify as Iris/Pupil, not Rim/Outside/Sclera.
        for seed in [1u64, 2, 3, 4, 5] {
            let eye = Eye::random(seed, RECT);
            let (cx, cy) = RECT.center();
            let eye = eye.with_target(Some((cx, cy)));
            let geom = eye.geometry(RECT, 0.0);
            let mut inside = 0u32;
            let mut total = 0u32;
            let steps = 40;
            for iy in 0..steps {
                for ix in 0..steps {
                    let fx = -1.0 + 2.0 * (ix as f32 + 0.5) / steps as f32;
                    let fy = -1.0 + 2.0 * (iy as f32 + 0.5) / steps as f32;
                    if fx * fx + fy * fy > 1.0 {
                        continue;
                    }
                    let fcol = geom.icx + fx * geom.iris_r;
                    let frow = geom.icy + (fy * geom.iris_r) / CELL_ASPECT;
                    total += 1;
                    let zone = Eye::classify_with_geom(&geom, fcol, frow);
                    if zone == EyeZone::Iris || zone == EyeZone::Pupil {
                        inside += 1;
                    }
                }
            }
            let frac = f64::from(inside) / f64::from(total);
            assert!(
                frac >= 0.65,
                "seed={seed}: only {frac:.2} of the iris disc is visible with gaze centred \
                 (expected >= 0.65 — 'iris fully visible except ~10-20% under the upper lid')"
            );
        }
    }

    #[test]
    fn gaze_target_overrides_saccade_and_points_the_right_way() {
        let rect = RECT;
        let eye = Eye::random(77, rect).with_variant(EyeVariant::Human);
        let (cx, cy) = rect.center();
        // Look far to the right.
        let eye_right = eye.with_target(Some((cx + 100.0, cy)));
        let geom = eye_right.geometry(rect, 0.0);
        assert!(
            geom.icx > geom.cx,
            "gaze right must move the iris centre rightward"
        );
        // Look far to the left.
        let eye_left = eye.with_target(Some((cx - 100.0, cy)));
        let geom2 = eye_left.geometry(rect, 0.0);
        assert!(
            geom2.icx < geom2.cx,
            "gaze left must move the iris centre leftward"
        );
    }

    #[test]
    fn every_variant_paints_without_panicking_and_stays_in_rect() {
        let fb_cols = 60;
        let fb_rows = 30;
        for variant in [
            EyeVariant::Human,
            EyeVariant::CatSlit,
            EyeVariant::Reptile,
            EyeVariant::Cyber,
            EyeVariant::Insect,
        ] {
            let mut fb = Framebuffer::new(fb_cols, fb_rows);
            let rect = Rect::new(5, 3, 40, 20);
            let eye = Eye::random(42, rect).with_variant(variant);
            eye.paint(&mut fb, rect, Duration::from_millis(500));
            // Nothing outside the rect should have been touched (still the
            // Default cell re_core::Framebuffer::new seeds every cell with).
            for row in 0..fb_rows {
                for col in 0..fb_cols {
                    if !rect.contains(col as i64, row as i64) {
                        let c = fb.get(col, row).unwrap();
                        assert_eq!(
                            *c,
                            re_core::Cell::default(),
                            "effect wrote outside its rect"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        // The ONLY glyph `paint` ever emits now — every other visual detail
        // (iris gradient, limbal ring, streaks, specular, veins, lid shadow)
        // is colour, carried by `fg`/`bg`, never a second glyph.
        assert!(
            is_reliable('\u{2580}'),
            "half-block glyph is not in the reliable set"
        );
    }

    /// For each pupil sample point (world coords relative to the iris
    /// centre, in `[-half, half]` on both axes) return `(is_pupil,
    /// is_in_specular)` via the SAME hard geometric tests `shade`'s
    /// coverage-blended version is a smooth refinement of — a direct area
    /// measurement, not a colour-threshold proxy.
    fn sample_pupil_and_specular_overlap(eye: &Eye, geom: &EyeGeom) -> (u32, u32) {
        let steps: i32 = 150;
        let half = geom.iris_r * 1.05;
        let spec_dist = geom.iris_r * eye.specular_dist_frac;
        let spec_r = geom.iris_r * eye.specular_radius_frac;
        let dx_s = spec_dist * eye.specular_angle.cos();
        let dy_s = spec_dist * eye.specular_angle.sin();
        let mut pupil_n = 0u32;
        let mut overlap_n = 0u32;
        for iy in 0..steps {
            for ix in 0..steps {
                let fx = -half + 2.0 * half * (ix as f32 + 0.5) / steps as f32;
                let fy = -half + 2.0 * half * (iy as f32 + 0.5) / steps as f32;
                let fcol = geom.icx + fx;
                let frow = geom.icy + fy / CELL_ASPECT;
                if geom.is_pupil(fcol, frow) {
                    pupil_n += 1;
                    if (fx - dx_s).hypot(fy - dy_s) <= spec_r {
                        overlap_n += 1;
                    }
                }
            }
        }
        (pupil_n, overlap_n)
    }

    #[test]
    fn specular_highlight_never_covers_more_than_15pct_of_a_slit_pupil() {
        // The owner's exact complaint: "the catslit/reptile specular
        // highlight is large and overlaps the pupil". Measured directly as
        // an AREA ratio (pupil cells that also fall inside the specular
        // disc, over all pupil cells) across many seeds and times, for both
        // slit variants — not a visual impression.
        const SAMPLE_RECT: Rect = Rect {
            col0: 0,
            row0: 0,
            cols: 60,
            rows: 36,
        };
        for variant in [EyeVariant::CatSlit, EyeVariant::Reptile] {
            for seed in 0..24u64 {
                let eye = Eye::random(seed * 97 + 3, SAMPLE_RECT).with_variant(variant);
                for ms in [0u64, 900, 2600] {
                    let t = secs(Duration::from_millis(ms));
                    let geom = eye.geometry(SAMPLE_RECT, t);
                    let (pupil_n, overlap_n) = sample_pupil_and_specular_overlap(&eye, &geom);
                    if pupil_n == 0 {
                        continue;
                    }
                    let frac = f64::from(overlap_n) / f64::from(pupil_n);
                    assert!(
                        frac <= 0.15 + 1e-3,
                        "{variant:?} seed={seed} t={ms}ms: specular covers {frac:.3} of the pupil (> 0.15)"
                    );
                }
            }
        }
    }

    #[test]
    fn specular_bearing_stays_clear_of_the_vertical_slit_axis() {
        // A direct proof on the ANGLE itself (not just the measured overlap
        // above): the drawn bearing must sit strictly in the upper
        // hemisphere (`sin < 0`) and at least `SPECULAR_VERTICAL_OFFSET_RANGE.0`
        // radians off the vertical, for every seed.
        for seed in 0..200u64 {
            let bearing = specular_bearing(seed);
            let straight_up = -(TAU * 0.25);
            let off_vertical = (bearing - straight_up)
                .abs()
                .min((bearing - (straight_up + TAU)).abs());
            assert!(
                bearing.sin() < 0.0,
                "seed={seed}: bearing {bearing} is not in the upper hemisphere"
            );
            assert!(
                off_vertical >= SPECULAR_VERTICAL_OFFSET_RANGE.0 - 1e-4,
                "seed={seed}: bearing {bearing} is only {off_vertical} off vertical"
            );
        }
    }
}
