//! `re_render::ramp_steps` — make a 45° `RampZ` run READ as a flight of STEPS.
//!
//! ## Why this module exists
//!
//! The DDA hits a [`re_core::Shape::RampZ`] cell on its tilted 45° plane and the
//! shading path lights that plane with its real slope normal ([`crate::shape_hit`]).
//! That is geometrically correct, but a continuous lit plane reads as a SMOOTH,
//! blank diagonal face — the eye gets no "these are stairs" cue. The silhouette
//! contour pass ([`crate::contour`] / [`crate::slope`]) only draws the OUTER edge of
//! the run (a `◣`/`◢` chamfer at the corner), so the interior of the ramp face is
//! empty fill.
//!
//! This module supplies the INTERIOR detail per the project's contour(outer) +
//! symbols(inner) rule: it turns the smooth ramp face into visible TREADS and RISERS.
//! For a ramp cell hit, given the world-height the column sees on the slope and the
//! cell's vertical face coordinate, it returns:
//!
//! * a **STEP-EDGE line** (`_`/`‾`/`/`/`\`) at the lip of each tread — a bright,
//!   high-contrast horizontal/diagonal stroke so each step's nose is legible, AND
//! * an **INNER lean line** on the tread body so the surface is not blank — a faint
//!   diagonal hatch leaning the SAME way as the slope, reinforcing the climb
//!   direction, AND
//! * a per-band **brightness/colour bias** (`StepShade`) so the edge line is brighter
//!   than the riser beneath it — the contrast that makes the step pop in truecolor.
//!
//! Each *ramp cell* IS one step (a 45° ramp rises exactly one cell per cell), so the
//! step rhythm comes from the per-cell world height: the integer part selects the
//! tread, the fraction places this row within the tread. We draw the edge where the
//! fraction crosses the tread lip.
//!
//! ## Direction correctness
//!
//! The step-edge glyph and the inner lean MUST point the correct way for the actual
//! [`RampDir`] AND the viewing angle, or an ascending stair reads as descending. The
//! mapping is pinned by [`step_edge_glyph`] + its exhaustive unit tests across all
//! four `RampDir`s and both screen-facing signs. The rule:
//!
//! * A ramp seen so its rise axis runs LEFT→RIGHT across the screen shows a DIAGONAL
//!   step nose: `/` when the slope ASCENDS to the right, `\` when it ascends to the
//!   left.
//! * A ramp seen edge-on / head-on (rise axis pointing toward/away from the eye)
//!   shows a HORIZONTAL tread lip: `‾` (overline) for the bright nose, `_` for the
//!   recessed back of the tread.
//!
//! Pure compute, allocation-free, no terminal — like the rest of `re_render`.

use re_core::RampDir;

// ───────────────────────────── tunables (named consts) ───────────────────────

/// Fraction of each tread (measured from its LIP downward in world height) that is
/// drawn as the bright STEP-EDGE band. The remainder of the tread is the dimmer
/// riser/body. `0.34` ≈ the top third of each step is the lit nose — wide enough to
/// survive the coarse cell quantisation (a step is only ~1 screen row tall when the
/// run recedes), narrow enough that the riser still reads as a separate band.
pub const STEP_EDGE_BAND: f32 = 0.34;

/// Brightness multiplier for the bright STEP-EDGE (tread lip) band. The nose of each
/// step is the meaningful structure, so it gets lifted well above the riser — this is
/// the fg contrast that makes the step pop. Hue-preserving (a scalar luma multiply).
pub const STEP_EDGE_LIFT: f32 = 1.6;

/// Brightness multiplier for the recessed RISER/body band — pushed DOWN so it reads
/// as the shadowed vertical face under the tread lip. The gap between this and
/// [`STEP_EDGE_LIFT`] is the step contrast.
pub const STEP_RISER_DIM: f32 = 0.62;

/// Background multiplier for the step-edge band: the lip sits over a slightly
/// DEEPENED groove so the bright nose has contrast against its own bg (never fg≈bg).
pub const STEP_EDGE_BG_SCALE: f32 = 0.5;

/// Background multiplier for the riser band — a touch deeper still, so the riser
/// reads as recessed shadow.
pub const STEP_RISER_BG_SCALE: f32 = 0.7;

// ── Brightness FLOORS (the "no black face" guarantee) ────────────────────────
//
// The tilted-slope Lambert term varies a lot with `RampDir`: a face whose normal
// rakes toward the KEY light (`PosX`/`PosY`) lands bright, while one that rakes
// AWAY (`NegX`/`NegY`) collapses to `ambient`-only (`N·L ≈ 0.26`). Multiplied by
// the wall's dark base and the depth falloff, those away-facing slopes sink to
// near-black on a near-black bg — a SPIRAL staircase (which uses all four
// directions) then reads as one black wall: the defect this fix targets.
//
// So after the multiplicative step bias we LIFT each band to a hue-preserving
// luminance floor: the lit slope keeps its colour identity and its lip-vs-riser
// contrast, but no RampDir face is ever darker than these floors — every step is
// legible from any angle. The floors are absolute Rec.601 luma in `[0,1]`, named
// here as the readability tunables `/iterate-render` drives.

/// Minimum fg luminance for the bright STEP-EDGE (tread lip). The step nose is the
/// load-bearing "these are stairs" cue, so it is held well clear of black on every
/// face. Hue-preserving (the colour is scaled UP toward this luma, not whitened).
pub const STEP_EDGE_FG_FLOOR: f32 = 0.46;

/// Minimum fg luminance for the recessed RISER/body band. Lower than
/// [`STEP_EDGE_FG_FLOOR`] so the lip↔riser contrast survives the floor, but still
/// clear of black so an away-facing slope's body is not an empty void.
pub const STEP_RISER_FG_FLOOR: f32 = 0.20;

/// Minimum bg luminance for the STEP-EDGE groove: the bright nose needs a backing
/// that is not pure black or the glyph vanishes into the void on a dark wall. A
/// small floor keeps an honest fg/bg contrast (nose brighter than its groove).
pub const STEP_EDGE_BG_FLOOR: f32 = 0.10;

/// Minimum bg luminance for the RISER band — a touch lower than the lip groove so
/// the riser still reads as the deeper, recessed face, but never collapses to a
/// black hole behind the body glyph.
pub const STEP_RISER_BG_FLOOR: f32 = 0.06;

// ── DEPTH brightness falloff (the head-on tread-separation cue) ──────────────
//
// A flight of stairs seen head-on recedes into the distance, but every tread is
// shaded by the SAME slope normal, so without a depth cue the near treads and the
// far treads land at identical brightness and the flight reads as one flat sheet.
// To make individual treads pop we modulate the per-band brightness by the cell's
// perpendicular depth: NEAR steps are lifted, FAR steps fade toward (but never
// below) a readable plateau. The result is a smooth front→back brightness gradient
// laid OVER the per-tread lip↔riser contrast — the two cues compound so each step
// nose reads as a distinct bright stroke against a progressively dimmer field.
//
// ## Why depth modulates the FLOOR, not just the multiplier
//
// The ramp's base wall fg arrives VERY dark (a tilted slope's Lambert term × the
// dark wall base). After the band multiply it sits at or below the per-band
// luminance floor, so `scale_then_floor` lifts EVERY band to its fixed floor — and
// a pure `fg_mul` depth scale below the floor is invisible (the floor erases it).
// Measured on `20-stairs-straight`: every lip clamped to luma 115 (floor 0.46) and
// every riser to 50 (floor 0.20), with NO depth gradient at all.
//
// So depth's primary action here is to modulate the per-band luma FLOOR itself: a
// NEAR step targets a floor lifted by [`STEP_DEPTH_NEAR_FLOOR_BONUS`] (a brighter
// readable target that stands clear above the base floor), a FAR step targets the
// plain base floor. That guarantees BOTH halves of the contract on a dark base:
// near is strictly brighter than far, and far never drops below the base floor (it
// IS the base floor). The `fg_mul` depth scale is kept as a secondary lever so the
// gradient also shows on the rare bright base where there is headroom above the
// floor. This mirrors the renderer's linear distance bands (`ascii_shader`'s
// `distance_fog` near/far edges) but stays LOCAL to the step relief. Continuous
// depth brightness only — there is deliberately NO LOD tiering here (deferred).

/// Perpendicular depth (world cells) at and below which a ramp step gets the FULL
/// near brightness boost. Closer than this, every tread is lit at the near plateau.
pub const STEP_DEPTH_NEAR: f32 = 1.5;

/// Perpendicular depth (world cells) at and beyond which the depth boost has fully
/// decayed to the far plateau. Past this, treads share the dim far shade (their
/// separation then comes from the lip↔riser contrast + the base floor, not depth).
/// Sized to a typical full flight so the gradient spans the whole staircase.
pub const STEP_DEPTH_FAR: f32 = 16.0;

/// Brightness multiplier applied to a NEAR step band (depth `<= STEP_DEPTH_NEAR`).
/// `> 1` so the closest treads are lifted ABOVE the mid-flight shade where the base
/// is bright enough to have headroom over the floor. Compounds with the band's own
/// `fg_mul` (lip lift / riser dim).
pub const STEP_DEPTH_NEAR_MUL: f32 = 1.35;

/// Brightness multiplier applied to a FAR step band (depth `>= STEP_DEPTH_FAR`).
/// `< 1` so distant treads dim into the background, opening a clear front→back
/// gradient. The per-band luminance FLOOR still clamps the final colour, so this
/// only DIMS toward the floor — it never produces a black/illegible far step.
pub const STEP_DEPTH_FAR_MUL: f32 = 0.55;

/// Luma bonus added to a NEAR step's per-band floor (depth `<= STEP_DEPTH_NEAR`),
/// decaying to 0 at [`STEP_DEPTH_FAR`]. This is the PRIMARY depth lever: because the
/// dark ramp base clamps every band to its floor, lifting the near floor is what
/// actually makes near treads brighter than far ones on screen. Sized so a near lip
/// targets ≈0.46+0.30 = 0.76 luma while a far lip stays at the 0.46 base floor — a
/// strong, legible front→back gradient. A far step's bonus is 0, so it sits exactly
/// on the base floor (never below it: the floor wins).
pub const STEP_DEPTH_NEAR_FLOOR_BONUS: f32 = 0.30;

/// Extra brightness multiplier for a FRONT-facing step (rise axis pointing
/// toward/away from the eye — the head-on tread lip). `> 1` so a front-facing run is
/// lifted slightly above an adjacent side-facing (perpendicular) run; this keeps the
/// two perpendicular flights of a SPIRAL staircase from merging into one tone where
/// they meet. Depth is the primary lever — this is a small secondary bias.
pub const STEP_FACE_FRONT_MUL: f32 = 1.08;

/// Brightness multiplier for a SIDE-facing step (rise axis running across the
/// screen). `< 1` (mildly) so it reads as the recessed perpendicular face next to a
/// front-facing run. The gap to [`STEP_FACE_FRONT_MUL`] is the spiral-merge guard.
pub const STEP_FACE_SIDE_MUL: f32 = 0.94;

/// Near-ness factor in `[0,1]` for a ramp step at perpendicular `depth` (world
/// cells): `1` at/inside [`STEP_DEPTH_NEAR`], falling linearly to `0` at
/// [`STEP_DEPTH_FAR`] and beyond. The shared shape both the depth multiplier and the
/// depth floor-bonus are derived from. Mirrors `ascii_shader`'s `distance_fog` band.
#[inline]
pub fn depth_nearness(depth: f32) -> f32 {
    if depth <= STEP_DEPTH_NEAR || STEP_DEPTH_FAR <= STEP_DEPTH_NEAR {
        1.0
    } else if depth >= STEP_DEPTH_FAR {
        0.0
    } else {
        1.0 - (depth - STEP_DEPTH_NEAR) / (STEP_DEPTH_FAR - STEP_DEPTH_NEAR)
    }
}

/// Depth-brightness MULTIPLIER for a ramp step at perpendicular `depth`: lerps from
/// [`STEP_DEPTH_NEAR_MUL`] (near) to [`STEP_DEPTH_FAR_MUL`] (far) over the band. The
/// secondary lever (matters only where the base has headroom above the floor); the
/// floor bonus is what carries the gradient on the dark ramp base. Always `> 0`.
#[inline]
pub fn depth_brightness(depth: f32) -> f32 {
    let n = depth_nearness(depth);
    STEP_DEPTH_FAR_MUL + n * (STEP_DEPTH_NEAR_MUL - STEP_DEPTH_FAR_MUL)
}

/// Per-band floor BONUS for a ramp step at perpendicular `depth`: [`STEP_DEPTH_NEAR_FLOOR_BONUS`]
/// near, decaying to `0` far. Added to the band's base floor so near treads target a
/// brighter readable luma than far treads — the primary on-screen depth cue. Far is
/// exactly `0`, so a far step rests on the base floor and never below it.
#[inline]
pub fn depth_floor_bonus(depth: f32) -> f32 {
    depth_nearness(depth) * STEP_DEPTH_NEAR_FLOOR_BONUS
}

/// Front-vs-side face brightness multiplier from the view↔rise-axis alignment.
/// `|view·rise|` near 1 ⇒ head-on (front-facing) ⇒ [`STEP_FACE_FRONT_MUL`]; near 0 ⇒
/// across-screen (side-facing) ⇒ [`STEP_FACE_SIDE_MUL`]. Uses the SAME dot the glyph
/// selection ([`step_edge_glyph`]) keys on, so the brightness bias and the chosen
/// glyph agree on which face this is. Continuous (a smooth lerp), not a hard step.
#[inline]
pub fn face_brightness(view_x: f32, view_y: f32, dir: RampDir) -> f32 {
    let (ax, ay) = rise_axis(dir);
    let vlen = (view_x * view_x + view_y * view_y).sqrt().max(1e-6);
    let dot = ((ax * view_x + ay * view_y) / vlen).abs().clamp(0.0, 1.0);
    STEP_FACE_SIDE_MUL + dot * (STEP_FACE_FRONT_MUL - STEP_FACE_SIDE_MUL)
}

/// How "head-on" (rise axis pointing toward/away from the eye) a ramp must look,
/// measured as `|view·rise_axis|`, before we draw the HORIZONTAL tread-lip glyph
/// instead of the DIAGONAL step-nose. Above this the lip is shown flat (`‾`/`_`);
/// below it the slope runs across the screen and the diagonal (`/`/`\`) reads better.
/// `0.5` ≈ 60° — past that the foreshortened slope is near edge-on.
pub const HEADON_DOT_THRESHOLD: f32 = 0.5;

/// Whether the ramp step-relief overlay is active. DEFAULT on (it is a pure
/// readability win — a smooth ramp face is the defect this module fixes). The
/// `/iterate-render` skill can disable it with `TS_RAMP_STEPS=0` to A/B the smooth
/// vs. stepped look. This is the only env read in the module and is consulted once
/// per ramp cell row — a single `var_os`/parse, no allocation, off the Full-wall path.
#[inline]
pub fn enabled() -> bool {
    match std::env::var("TS_RAMP_STEPS") {
        Ok(v) => {
            let t = v.trim();
            !(t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

// ───────────────────────────── glyph vocabulary ──────────────────────────────

/// The swappable step glyph set — Unicode (overline tread lip) and an ASCII degrade.
/// Mirrors [`crate::slope::SlopeTheme`]'s data-driven shape: detection logic is
/// fixed, the glyph vocabulary is DATA so `/iterate-render` / the present-layer
/// capability probe can swap it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepTheme {
    /// Theme name (debug / selection).
    pub name: &'static str,
    /// Bright tread-LIP glyph for a head-on view (the step's nose).
    pub lip: char,
    /// Recessed riser/back glyph for a head-on view.
    pub riser: char,
    /// Diagonal step-nose glyph when the slope ASCENDS to the RIGHT (a `/` stroke).
    pub diag_up_right: char,
    /// Diagonal step-nose glyph when the slope ASCENDS to the LEFT (a `\` stroke).
    pub diag_up_left: char,
    /// Faint inner lean glyph for a right-ascending slope body (a light `/`).
    pub inner_up_right: char,
    /// Faint inner lean glyph for a left-ascending slope body (a light `\`).
    pub inner_up_left: char,
}

impl StepTheme {
    /// The shipping Unicode theme: an OVERLINE `‾` (U+203E) for the bright tread lip
    /// (it sits high in the cell, reading as the step's top nose), a low underscore
    /// `_` for the recessed riser, true half-cell diagonals `╱`/`╲` for the slanted
    /// nose, and light box diagonals for the inner lean.
    pub const UNICODE: StepTheme = StepTheme {
        name: "unicode",
        lip: '‾',
        riser: '_',
        diag_up_right: '╱',
        diag_up_left: '╲',
        inner_up_right: '╱',
        inner_up_left: '╲',
    };

    /// ASCII degrade for glyph-poor terminals: `~`/`_` for the lip/riser and `/`/`\`
    /// for both the nose and the inner lean.
    pub const ASCII: StepTheme = StepTheme {
        name: "ascii",
        lip: '~',
        riser: '_',
        diag_up_right: '/',
        diag_up_left: '\\',
        inner_up_right: '/',
        inner_up_left: '\\',
    };
}

impl Default for StepTheme {
    fn default() -> Self {
        StepTheme::UNICODE
    }
}

// ───────────────────────────── result ────────────────────────────────────────

/// Which band of a tread a ramp row falls in — drives the brightness/bg bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepBand {
    /// The bright tread LIP (the step nose) — lit, high contrast.
    Edge,
    /// The recessed riser / tread body — dimmed so the edge reads above it.
    Riser,
}

/// The full step treatment for one ramp cell row: the glyph to stamp and the
/// brightness/bg multipliers for its band. `apply`-able by the caller over the
/// `shade_cell` result so the material's colour identity is preserved (we only
/// scale luma + swap the glyph), keeping the ramp face hue-true but stepped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepShade {
    /// The step glyph (lip / riser / diagonal nose / inner lean).
    pub glyph: char,
    /// fg luminance multiplier (hue-preserving).
    pub fg_mul: f32,
    /// bg luminance multiplier.
    pub bg_mul: f32,
    /// Minimum fg luminance (Rec.601, `[0,1]`) this band is lifted to AFTER the
    /// multiply — the "no black face" floor (see [`STEP_EDGE_FG_FLOOR`]).
    pub fg_floor: f32,
    /// Minimum bg luminance the groove behind the glyph is lifted to.
    pub bg_floor: f32,
    /// Which band this row is in.
    pub band: StepBand,
}

/// Rec.601 luminance of a packed `0x00RRGGBB` colour, in `[0,1]`. Local copy so
/// this module stays a leaf (no cross-module dep just for a dot product).
#[inline]
fn luma601(c: u32) -> f32 {
    let r = ((c >> 16) & 0xFF) as f32 / 255.0;
    let g = ((c >> 8) & 0xFF) as f32 / 255.0;
    let b = (c & 0xFF) as f32 / 255.0;
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Scale a packed colour by `mul`, then LIFT it (hue-preserving) so its luminance
/// is at least `floor`. The lift multiplies all channels by the same ratio, so the
/// hue is unchanged; channels saturate at `0xFF` (a very dark, near-monochrome
/// colour lifts toward grey, which is the desired "lit, readable" result). A pure
/// black input (luma 0) cannot be hue-lifted, so it is raised to a neutral grey at
/// the floor — a ramp face is never an inked-from-black void.
#[inline]
fn scale_then_floor(c: u32, mul: f32, floor: f32) -> u32 {
    let r0 = ((c >> 16) & 0xFF) as f32 / 255.0;
    let g0 = ((c >> 8) & 0xFF) as f32 / 255.0;
    let b0 = (c & 0xFF) as f32 / 255.0;
    let (mut r, mut g, mut b) = (r0 * mul, g0 * mul, b0 * mul);
    let l = 0.299 * r + 0.587 * g + 0.114 * b;
    if l < floor {
        if l > 1e-4 {
            // Hue-preserving lift: same ratio on every channel.
            let k = floor / l;
            r *= k;
            g *= k;
            b *= k;
        } else {
            // No hue to preserve (black) → neutral grey at the floor.
            r = floor;
            g = floor;
            b = floor;
        }
    }
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0) + 0.5) as u32 & 0xFF;
    (q(r) << 16) | (q(g) << 8) | q(b)
}

impl StepShade {
    /// Apply this step treatment over the already-shaded cell colours: bias the fg
    /// and bg by the band multipliers, then lift each to its "no black face" floor.
    /// Returns `(fg, bg)` packed `0x00RRGGBB`. This is the single place the relief
    /// brightness rule lives, so the floor guarantee holds for every `RampDir`.
    #[inline]
    pub fn apply(&self, fg: u32, bg: u32) -> (u32, u32) {
        (
            scale_then_floor(fg, self.fg_mul, self.fg_floor),
            scale_then_floor(bg, self.bg_mul, self.bg_floor),
        )
    }

    /// The post-apply fg luminance for a given base fg — used by tests to PIN that
    /// no `RampDir` face falls below the black threshold.
    #[inline]
    pub fn applied_fg_luma(&self, base_fg: u32) -> f32 {
        luma601(scale_then_floor(base_fg, self.fg_mul, self.fg_floor))
    }
}

// ───────────────────────────── core mapping ──────────────────────────────────

/// The rise axis (in world XY) a [`RampDir`] climbs along, and the sign of the climb.
/// `PosX` rises with +x, `NegX` with −x, etc. Returns `(axis_x, axis_y)` — a unit
/// vector pointing UPHILL in the ground plane.
#[inline]
pub fn rise_axis(dir: RampDir) -> (f32, f32) {
    match dir {
        RampDir::PosX => (1.0, 0.0),
        RampDir::NegX => (-1.0, 0.0),
        RampDir::PosY => (0.0, 1.0),
        RampDir::NegY => (0.0, -1.0),
    }
}

/// Pick the bright STEP-EDGE (nose) glyph for a ramp seen from a given view
/// direction. `dir` is the ramp's rise direction; `(view_x, view_y)` is the
/// horizontal direction from the SURFACE toward the CAMERA (un-normalised is fine —
/// only its components' relative sign/scale matter); `screen_right` is the world XY
/// direction that maps to +screen-x (camera right vector), so we can tell which way
/// the uphill axis leans ACROSS the screen.
///
/// The logic, pinned by the unit tests for all four `RampDir`s:
///
/// 1. If the uphill axis points mostly TOWARD/AWAY from the eye (`|view·rise|` large),
///    the slope is seen head-on → a HORIZONTAL tread lip (`‾`).
/// 2. Otherwise the uphill axis runs across the screen. Project it onto the camera's
///    RIGHT vector: a POSITIVE projection means the slope ascends toward screen-right
///    → `/`; NEGATIVE means it ascends toward screen-left → `\`.
#[inline]
pub fn step_edge_glyph(
    dir: RampDir,
    view_x: f32,
    view_y: f32,
    screen_right_x: f32,
    screen_right_y: f32,
    theme: &StepTheme,
) -> char {
    let (ax, ay) = rise_axis(dir);
    // Normalise the view for the head-on dot (rise axis is already unit).
    let vlen = (view_x * view_x + view_y * view_y).sqrt().max(1e-6);
    let dot = (ax * view_x + ay * view_y) / vlen;
    if dot.abs() >= HEADON_DOT_THRESHOLD {
        return theme.lip; // head-on: flat tread lip.
    }
    // Across-screen: which way does uphill lean? Project rise axis on camera-right.
    let lean = ax * screen_right_x + ay * screen_right_y;
    if lean >= 0.0 {
        theme.diag_up_right // ascends toward screen-right → `/`
    } else {
        theme.diag_up_left // ascends toward screen-left → `\`
    }
}

/// The faint INNER lean glyph for the tread BODY (riser). Always a diagonal leaning
/// the same way as the slope ascends across the screen, so the body is not blank and
/// the lean reinforces the climb direction. For a head-on view there is no across-
/// screen lean, so it degrades to the recessed `riser` glyph.
#[inline]
pub fn inner_lean_glyph(
    dir: RampDir,
    view_x: f32,
    view_y: f32,
    screen_right_x: f32,
    screen_right_y: f32,
    theme: &StepTheme,
) -> char {
    let (ax, ay) = rise_axis(dir);
    let vlen = (view_x * view_x + view_y * view_y).sqrt().max(1e-6);
    let dot = (ax * view_x + ay * view_y) / vlen;
    if dot.abs() >= HEADON_DOT_THRESHOLD {
        return theme.riser; // head-on body: plain recessed band.
    }
    let lean = ax * screen_right_x + ay * screen_right_y;
    if lean >= 0.0 {
        theme.inner_up_right
    } else {
        theme.inner_up_left
    }
}

/// Decide the step band + treatment for ONE ramp cell row.
///
/// * `dir` — the ramp's rise direction.
/// * `slope_height` — the WORLD height (z) of the slope surface the column sees at
///   this row. The FRACTIONAL part places the row within its tread; the lip is the
///   top [`STEP_EDGE_BAND`] of each unit.
/// * `depth` — the cell's PERPENDICULAR depth (world cells). NEAR steps are lifted,
///   FAR steps dim toward the far plateau (see [`depth_brightness`]) so a head-on
///   flight shows a front→back brightness gradient and each tread reads distinctly.
///   The per-band luminance FLOOR still clamps the result, so depth never blacks out
///   a far step — the floor wins.
/// * `view_*` / `screen_right_*` — view + camera-right vectors for glyph direction
///   (see [`step_edge_glyph`]) AND the front-vs-side face brightness bias.
/// * `theme` — the swappable glyph vocabulary.
///
/// Returns a [`StepShade`]: the glyph and the fg/bg luma multipliers for the band.
/// The caller multiplies the shaded cell's fg/bg by these (hue preserved) and stamps
/// the glyph — turning the smooth ramp face into visible, direction-correct steps.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_shade(
    dir: RampDir,
    slope_height: f32,
    depth: f32,
    view_x: f32,
    view_y: f32,
    screen_right_x: f32,
    screen_right_y: f32,
    theme: &StepTheme,
) -> StepShade {
    // Position within the current tread: 0 at the LIP (top of the step), rising
    // toward 1 at the back/bottom. We measure DOWN from the lip so the bright band is
    // a clean [0, STEP_EDGE_BAND).
    let frac = slope_height.rem_euclid(1.0);
    // The lip is the TOP of each unit-tall step → the band just BELOW an integer.
    // `frac` near 1.0 (just under the lip) and near 0.0 (just over the next lip) both
    // border a step edge; we treat the top STEP_EDGE_BAND of the tread as the nose.
    let dist_to_lip = 1.0 - frac; // 0 at the lip, →1 at the back of the tread.
    let on_edge = dist_to_lip <= STEP_EDGE_BAND;

    // Depth + face drive TWO levers (see the module-level note on why the floor is the
    // primary one on the dark ramp base):
    //   * `dim` scales `fg_mul` — the secondary lever, visible only where the base has
    //     headroom above the floor.
    //   * `floor_bonus` lifts the per-band fg FLOOR for near/front steps — the primary
    //     on-screen cue, because the dark base otherwise clamps every band to its
    //     fixed floor. A FAR step's bonus is 0, so it rests on the base floor and is
    //     never driven below it: the floor still wins, the far step stays legible.
    let depth_mul = depth_brightness(depth);
    let face_mul = face_brightness(view_x, view_y, dir);
    let dim = depth_mul * face_mul;
    // Near-ness floor lift, scaled by the front/side face factor so a front-facing run
    // reads slightly brighter than an adjacent perpendicular run (spiral-merge guard).
    let fg_floor_bonus = depth_floor_bonus(depth) * face_mul;

    if on_edge {
        StepShade {
            glyph: step_edge_glyph(dir, view_x, view_y, screen_right_x, screen_right_y, theme),
            fg_mul: STEP_EDGE_LIFT * dim,
            bg_mul: STEP_EDGE_BG_SCALE,
            fg_floor: STEP_EDGE_FG_FLOOR + fg_floor_bonus,
            bg_floor: STEP_EDGE_BG_FLOOR,
            band: StepBand::Edge,
        }
    } else {
        StepShade {
            glyph: inner_lean_glyph(dir, view_x, view_y, screen_right_x, screen_right_y, theme),
            fg_mul: STEP_RISER_DIM * dim,
            bg_mul: STEP_RISER_BG_SCALE,
            // The riser floor gets a SMALLER share of the bonus so the lip↔riser
            // contrast widens with proximity (near risers stay clearly under near lips).
            fg_floor: STEP_RISER_FG_FLOOR + fg_floor_bonus * 0.5,
            bg_floor: STEP_RISER_BG_FLOOR,
            band: StepBand::Riser,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A camera looking +y (north): view-from-surface-toward-camera is −y for a
    // surface the camera faces; camera-right is +x.
    const VIEW_NORTH_X: f32 = 0.0;
    const VIEW_NORTH_Y: f32 = -1.0;
    const RIGHT_X: f32 = 1.0;
    const RIGHT_Y: f32 = 0.0;

    /// A neutral mid-flight depth for band/direction tests that do not care about the
    /// depth gradient — inside the falloff band so the depth multiplier is benign.
    const MID_DEPTH: f32 = 6.0;

    #[test]
    fn rise_axis_points_uphill_per_dir() {
        assert_eq!(rise_axis(RampDir::PosX), (1.0, 0.0));
        assert_eq!(rise_axis(RampDir::NegX), (-1.0, 0.0));
        assert_eq!(rise_axis(RampDir::PosY), (0.0, 1.0));
        assert_eq!(rise_axis(RampDir::NegY), (0.0, -1.0));
    }

    #[test]
    fn headon_ramp_shows_flat_tread_lip() {
        // Camera looking +y at a PosY ramp (rise axis +y points toward/away from the
        // eye) → head-on → flat overline lip, not a diagonal.
        let t = StepTheme::UNICODE;
        let g = step_edge_glyph(
            RampDir::PosY,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(g, '‾', "a head-on ramp must show the flat tread lip");
        // NegY (rises away) is also head-on from this camera.
        let g2 = step_edge_glyph(
            RampDir::NegY,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(g2, '‾');
    }

    #[test]
    fn across_screen_ramp_picks_correct_diagonal_lean() {
        // Camera looking +y; camera-right is +x. A PosX ramp's uphill axis is +x →
        // ascends toward screen-RIGHT → `/`. A NegX ramp ascends toward screen-LEFT → `\`.
        let t = StepTheme::UNICODE;
        let pos_x = step_edge_glyph(
            RampDir::PosX,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(
            pos_x, '╱',
            "PosX rise (uphill toward screen-right) must be `╱`"
        );
        let neg_x = step_edge_glyph(
            RampDir::NegX,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(
            neg_x, '╲',
            "NegX rise (uphill toward screen-left) must be `╲`"
        );
    }

    #[test]
    fn camera_right_flip_mirrors_the_diagonal() {
        // If the camera faces the OTHER way (camera-right is −x), the SAME PosX ramp
        // now ascends toward screen-LEFT → `╲`. Pins the view-angle dependence: the
        // glyph follows the SCREEN lean, not the world axis — a mirror-inversion
        // guard.
        let t = StepTheme::UNICODE;
        let g = step_edge_glyph(RampDir::PosX, 0.0, 1.0, -1.0, 0.0, &t);
        assert_eq!(
            g, '╲',
            "PosX seen from the far side leans the other way on screen"
        );
    }

    #[test]
    fn all_four_dirs_classify_across_a_side_view() {
        // Camera looking +x (east): camera-right is −y (screen-right maps to world −y).
        // A ramp whose rise axis is along ±y now runs across the screen.
        let t = StepTheme::UNICODE;
        let (vx, vy) = (-1.0, 0.0); // surface→camera for a camera looking +x
        let (rx, ry) = (0.0, -1.0); // camera-right
                                    // PosY rise (+y) projected on camera-right (−y) is NEGATIVE → ascends left → `╲`.
        assert_eq!(step_edge_glyph(RampDir::PosY, vx, vy, rx, ry, &t), '╲');
        // NegY rise (−y) projects POSITIVE → ascends right → `╱`.
        assert_eq!(step_edge_glyph(RampDir::NegY, vx, vy, rx, ry, &t), '╱');
        // PosX / NegX rise axis is ±x = toward/away the eye → head-on lip.
        assert_eq!(step_edge_glyph(RampDir::PosX, vx, vy, rx, ry, &t), '‾');
        assert_eq!(step_edge_glyph(RampDir::NegX, vx, vy, rx, ry, &t), '‾');
    }

    #[test]
    fn step_shade_alternates_edge_and_riser_across_a_tread() {
        // March the slope height across one full tread: the top STEP_EDGE_BAND is the
        // bright Edge, the rest is the dim Riser.
        let t = StepTheme::UNICODE;
        // Just under an integer lip (dist_to_lip small) → Edge.
        let lip = step_shade(
            RampDir::PosY,
            4.95,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(lip.band, StepBand::Edge);
        assert!(
            lip.fg_mul > 1.0,
            "the lip must be BRIGHTER than the surface"
        );
        // Mid-tread (dist_to_lip ~0.5) → Riser.
        let mid = step_shade(
            RampDir::PosY,
            4.5,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(mid.band, StepBand::Riser);
        assert!(
            mid.fg_mul < 1.0,
            "the riser must be DIMMER than the surface"
        );
    }

    #[test]
    fn edge_is_strictly_brighter_than_riser() {
        // The contrast that makes a step pop: a lip row must end up brighter than a
        // riser row from the SAME ramp — asserted through the public `step_shade` output
        // so it exercises the real band selection, not a bare const comparison.
        let t = StepTheme::UNICODE;
        let lip = step_shade(
            RampDir::PosY,
            4.95,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        let riser = step_shade(
            RampDir::PosY,
            4.5,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert!(
            lip.fg_mul > riser.fg_mul,
            "the lit step lip ({}) must be brighter than the riser ({})",
            lip.fg_mul,
            riser.fg_mul
        );
        assert!(lip.fg_mul > 1.0, "the lip lifts above the surface");
        assert!(riser.fg_mul < 1.0, "the riser dims below the surface");
    }

    #[test]
    fn ascii_theme_degrades_glyphs() {
        let t = StepTheme::ASCII;
        assert_eq!(
            step_edge_glyph(
                RampDir::PosY,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t
            ),
            '~'
        );
        assert_eq!(
            step_edge_glyph(
                RampDir::PosX,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t
            ),
            '/'
        );
        assert_eq!(
            step_edge_glyph(
                RampDir::NegX,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t
            ),
            '\\'
        );
    }

    #[test]
    fn step_period_is_one_cell_per_tread() {
        // Two slope heights one unit apart must land in the SAME band (the step
        // rhythm has period 1.0 — each cell is one tread).
        let t = StepTheme::UNICODE;
        let a = step_shade(
            RampDir::PosY,
            2.2,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        let b = step_shade(
            RampDir::PosY,
            3.2,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(a.band, b.band, "the tread rhythm must repeat every cell");
    }

    // ── "no black face" floor — the spiral-staircase fix ─────────────────────

    /// Below this packed-fg luma a face reads as black on the dark wall bg. Every
    /// `RampDir` lip MUST clear it after `apply`, for any base shade — even a face
    /// the renderer shaded down to nearly nothing (the away-facing NegX/NegY case).
    const BLACK_THRESHOLD: f32 = 0.30;

    /// The wall material's lit fg can land very dark on an away-facing slope. Model
    /// the worst case as a near-black base and assert the floor still lifts it.
    const DARK_BASE_FG: u32 = 0x0A_0C_12; // ≈ the wall bg, luma ≈ 0.04

    #[test]
    fn every_rampdir_lip_clears_black_after_apply() {
        // For all four directions, at a lip height, the applied fg must clear the
        // black threshold even when the base shade is almost black — so a spiral
        // staircase (all four dirs) never renders a black step face.
        let t = StepTheme::UNICODE;
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            // 4.95 → just under a lip → Edge band. Use a FAR depth so the depth
            // multiplier is at its DIMMEST — the floor must still clear black there.
            let lip = step_shade(
                dir,
                4.95,
                STEP_DEPTH_FAR + 5.0,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t,
            );
            assert_eq!(lip.band, StepBand::Edge, "{dir:?}: 4.95 is a lip row");
            let fl = lip.applied_fg_luma(DARK_BASE_FG);
            assert!(
                fl >= BLACK_THRESHOLD,
                "{dir:?}: lip fg luma {fl:.3} fell below the black threshold \
                 {BLACK_THRESHOLD} — this RampDir face would render black"
            );
        }
    }

    #[test]
    fn every_rampdir_riser_clears_black_after_apply() {
        // The riser body is dimmer than the lip but must still not be a black void.
        let t = StepTheme::UNICODE;
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            // FAR depth → dimmest depth multiplier; the riser floor must still hold.
            let riser = step_shade(
                dir,
                4.5,
                STEP_DEPTH_FAR + 5.0,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t,
            );
            assert_eq!(riser.band, StepBand::Riser, "{dir:?}: 4.5 is a riser row");
            let fl = riser.applied_fg_luma(DARK_BASE_FG);
            assert!(
                fl >= STEP_RISER_FG_FLOOR - 1e-3,
                "{dir:?}: riser fg luma {fl:.3} sank below its floor — black body"
            );
        }
    }

    #[test]
    fn apply_preserves_lip_brighter_than_riser_contrast() {
        // The floor must not flatten the step: a lip is still strictly brighter than
        // a riser from the SAME ramp after both are floored — the contrast that makes
        // a step read. Use a mid-bright base so neither is clamped to its floor.
        let t = StepTheme::UNICODE;
        let base = 0x90_98_A8u32; // ≈ the wall fg #A0A8B8 after some shading
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            let lip = step_shade(
                dir,
                4.95,
                MID_DEPTH,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t,
            );
            let riser = step_shade(
                dir,
                4.5,
                MID_DEPTH,
                VIEW_NORTH_X,
                VIEW_NORTH_Y,
                RIGHT_X,
                RIGHT_Y,
                &t,
            );
            let (lfg, _) = lip.apply(base, base);
            let (rfg, _) = riser.apply(base, base);
            assert!(
                luma601(lfg) > luma601(rfg),
                "{dir:?}: lip ({:.3}) must stay brighter than riser ({:.3}) after the floor",
                luma601(lfg),
                luma601(rfg)
            );
        }
    }

    #[test]
    fn apply_lifts_pure_black_to_neutral_grey_floor() {
        // A pure-black base has no hue to preserve — it must lift to a neutral grey
        // AT the floor (r==g==b), never stay black.
        let t = StepTheme::UNICODE;
        let lip = step_shade(
            RampDir::NegY,
            4.95,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        let (fg, _) = lip.apply(0x000000, 0x000000);
        let r = (fg >> 16) & 0xFF;
        let g = (fg >> 8) & 0xFF;
        let b = fg & 0xFF;
        assert!(r > 0 && g > 0 && b > 0, "black must lift, not stay black");
        assert!(
            r.abs_diff(g) <= 2 && g.abs_diff(b) <= 2,
            "black lifts to NEUTRAL grey (got {r},{g},{b})"
        );
    }

    #[test]
    fn apply_is_hue_preserving_for_a_coloured_base() {
        // A coloured (non-black) base that needs lifting keeps its hue: the channel
        // RATIOS are unchanged by the floor lift.
        let t = StepTheme::UNICODE;
        let riser = step_shade(
            RampDir::PosX,
            4.5,
            MID_DEPTH,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        let base = 0x10_20_08u32; // a dark greenish tint, luma ≈ 0.10
        let (fg, _) = riser.apply(base, base);
        let r = ((fg >> 16) & 0xFF) as f32;
        let g = ((fg >> 8) & 0xFF) as f32;
        let b = (fg & 0xFF) as f32;
        // Original ratios: r:g:b = 16:32:8 = 2:4:1. Check g/r ≈ 2 and r/b ≈ 2.
        assert!((g / r - 2.0).abs() < 0.15, "hue (g/r) drifted: {}", g / r);
        assert!((r / b - 2.0).abs() < 0.25, "hue (r/b) drifted: {}", r / b);
    }

    // ── DEPTH brightness falloff — the head-on tread-separation cue ───────────

    #[test]
    fn depth_brightness_is_monotonic_near_brighter_than_far() {
        // NEAR steps must be brighter than FAR steps: the falloff is non-increasing
        // across the band and strictly higher near than far.
        assert!(depth_brightness(0.0) >= depth_brightness(STEP_DEPTH_FAR + 10.0));
        assert!(
            depth_brightness(STEP_DEPTH_NEAR) > depth_brightness(STEP_DEPTH_FAR),
            "near multiplier {} must exceed far multiplier {}",
            depth_brightness(STEP_DEPTH_NEAR),
            depth_brightness(STEP_DEPTH_FAR)
        );
        // Plateaus: clamped flat outside the band.
        assert_eq!(depth_brightness(0.0), STEP_DEPTH_NEAR_MUL);
        assert_eq!(depth_brightness(STEP_DEPTH_FAR + 99.0), STEP_DEPTH_FAR_MUL);
        // Smooth interior: a mid sample lies strictly between the plateaus.
        let mid = depth_brightness((STEP_DEPTH_NEAR + STEP_DEPTH_FAR) * 0.5);
        assert!(mid < STEP_DEPTH_NEAR_MUL && mid > STEP_DEPTH_FAR_MUL);
    }

    #[test]
    fn near_step_brighter_than_same_step_far_yet_far_clears_floor() {
        // THE acceptance test: a NEAR lip is brighter than the SAME lip kind far away
        // (so a head-on flight shows a front→back gradient), AND the far lip still
        // clears the "no black face" floor (depth dimming never wins over the floor).
        let t = StepTheme::UNICODE;
        let base = 0x90_98_A8u32; // a mid-bright wall fg (not floor-clamped near)
        let dir = RampDir::PosY; // head-on flight (straight stairs)

        let near = step_shade(
            dir,
            4.95,
            0.5,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        let far = step_shade(
            dir,
            4.95,
            STEP_DEPTH_FAR + 4.0,
            VIEW_NORTH_X,
            VIEW_NORTH_Y,
            RIGHT_X,
            RIGHT_Y,
            &t,
        );
        assert_eq!(near.band, StepBand::Edge);
        assert_eq!(far.band, StepBand::Edge);

        // 1. Multiplier-level: near is brighter than far.
        assert!(
            near.fg_mul > far.fg_mul,
            "near lip mul {} must exceed far lip mul {}",
            near.fg_mul,
            far.fg_mul
        );

        // 2. Applied luma on a bright base: near reads brighter than far.
        let near_l = near.applied_fg_luma(base);
        let far_l = far.applied_fg_luma(base);
        assert!(
            near_l > far_l,
            "near lip luma {near_l:.3} must read brighter than far {far_l:.3}"
        );

        // 3. Applied luma on the DARK ramp base (the real case): the floor bonus must
        //    still make near brighter than far. This is the cue that was previously
        //    invisible — a pure `fg_mul` scale below the floor clamps both to the same
        //    floor. The floor-bonus lever fixes it.
        let near_dark = near.applied_fg_luma(DARK_BASE_FG);
        let far_dark = far.applied_fg_luma(DARK_BASE_FG);
        assert!(
            near_dark > far_dark + 0.05,
            "on a dark base, near lip luma {near_dark:.3} must clearly exceed far \
             {far_dark:.3} — the depth gradient must survive the floor clamp"
        );

        // 4. The far lip is STILL legible — never below the BASE luminance floor, even
        //    from a near-black base where depth dimming is most aggressive.
        assert!(
            far_dark >= STEP_EDGE_FG_FLOOR - 1e-3,
            "a far step sank below the edge floor ({far_dark:.3} < {STEP_EDGE_FG_FLOOR}) \
             — depth dimming must not produce a black/illegible far step"
        );
    }

    #[test]
    fn front_face_brighter_than_side_face_spiral_guard() {
        // A front-facing run (rise axis toward/away the eye) reads slightly brighter
        // than an adjacent side-facing run, so the two perpendicular flights of a
        // SPIRAL staircase do not merge into one tone. Same depth on both.
        // Camera looking +y: PosY rise is head-on (front); PosX rise is across (side).
        let front = face_brightness(VIEW_NORTH_X, VIEW_NORTH_Y, RampDir::PosY);
        let side = face_brightness(VIEW_NORTH_X, VIEW_NORTH_Y, RampDir::PosX);
        assert!(
            front > side,
            "front-facing run ({front}) must be brighter than side-facing ({side})"
        );
        assert!((front - STEP_FACE_FRONT_MUL).abs() < 1e-4);
        assert!((side - STEP_FACE_SIDE_MUL).abs() < 1e-4);
    }
}
