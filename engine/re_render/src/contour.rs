//! `re_render::contour` — the STRUCTURE-as-LINES + COLOR-as-VOLUME post-pass.
//!
//! This is the project's defining look lever, applied as a single Layer-A→A pass
//! over the finalized [`Framebuffer`] (after the composite resolve / after the
//! single-shot cast). It does two independent things, each gated by a named
//! tunable so a manual tuning pass can drive them via the `--dump` path:
//!
//! 1. **Auto contour → box-drawing LINE glyphs, in an OUTER/INNER hierarchy.** For
//!    each cell we compare `depth` and `surface` against the four orthogonal
//!    neighbours and classify each boundary as OUTER or INNER:
//!
//!    - **OUTER (silhouette / depth edge)** — a solid surface meeting open space
//!      (Empty/sky), or a strong depth discontinuity (a corner / occlusion lip).
//!      Drawn as the PRIMARY line: a crisp, near-white, bold box-drawing stroke.
//!    - **INNER (material seam)** — a real material/surface-kind change between two
//!      SOLID cells. Drawn as a SECONDARY line: subtler — dimmer, tinted toward the
//!      surface colour (not pure white), a lighter glyph weight — so the eye reads
//!      the silhouette first and internal seams second.
//!
//!    The chosen box-drawing glyph (`│ ─ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼` …) matches the edge
//!    orientation. The outer contour of a voxel model thus auto-maps to clean
//!    lines — readable structure, not fill.
//!
//!    **Single-side (draw-once) convention.** In a character grid there is no gap
//!    between cells — a contour line *is* a cell. A boundary between block A and
//!    block B is therefore attributed to exactly ONE side so it is drawn once, not
//!    by both neighbours (which would stack two horizontal/vertical lines at a
//!    wall-base transition). We use the **near/top** rule: the boundary is owned by
//!    the nearer (smaller-depth) cell; on an equal-depth material seam it is owned
//!    by the upper/leading cell (smaller row for an N/S seam, smaller col for an
//!    E/W seam). See [`owns_boundary`].
//!
//! 2. **Slight COLOR VARIATION for volume.** Non-edge cells get a small, readable
//!    fg/bg brightness shift driven by face orientation (`side`) and depth, so a
//!    flat field of one element colour still reads as a 3D surface: faces at
//!    different angles differ slightly, nearer cells are a touch brighter. Three
//!    element colours become many subtle variations — volume via colour, not via
//!    a glyph in every cell.
//!
//! The pass is a **no-op pass-through** wherever no edge is found and when the
//! corresponding tunable is disabled, so the sparse colour field from the
//! minimalist retune (`ascii_shader.rs`) is preserved everywhere except on the
//! genuine structural seams this pass draws.
//!
//! Pure compute, no terminal, no allocation in the loop (two scratch row buffers
//! are reused across rows). Engine-pure like the rest of `re_render`.

use re_core::{Cell, Framebuffer, SurfaceKind, UnderlineStyle};

use crate::ascii_shader::{pack_rgb, unpack_rgb};
use crate::slope::{self, SlopeTheme};

// ───────────────────────────── tunables (named consts) ───────────────────────

/// Relative depth jump (fraction of the nearer cell's depth) above which two
/// orthogonal neighbours are considered separated by a STRUCTURAL edge. Relative
/// (not absolute) so a 1-cell step reads as an edge both up-close and far away —
/// an absolute threshold would light every far wall or no near wall.
///
/// 0.18 ≈ "the neighbour is ≥18 % nearer/farther than me": a wall corner, an
/// object silhouette against a wall behind it, a doorway lip. Tuned to catch real
/// geometry boundaries without firing on the gentle per-row depth gradient of a
/// flat floor receding into the distance.
pub const DEPTH_EDGE_REL_JUMP: f32 = 0.18;

/// Absolute depth floor (world cells) for the relative test, so two cells that are
/// both extremely near (tiny depth) don't trip the relative test on sub-cell
/// noise. Below this absolute jump, no edge regardless of the relative ratio.
pub const DEPTH_EDGE_MIN_ABS: f32 = 0.35;

/// fg colour for a drawn contour line — a crisp near-white ink. The line is the
/// meaningful structure, so it gets the highest-contrast fg in the frame; the
/// underlying colour field carries identity, the line carries shape.
pub const CONTOUR_FG: u32 = 0x00F2_F4F8;

/// How much of the contour fg is the pure ink vs. a tint of the cell's own fg.
/// 1.0 = pure `CONTOUR_FG`; lower values let MORE of the surface colour bleed into
/// the line so a red wall's outline reads red-ish, not white.
///
/// A higher value (e.g. `0.82`) forces the outline that much toward
/// near-white, washing every surface's hue toward white wherever an edge is
/// found. At `0.45`, COMBINED with the luma-LIFT below ([`CONTOUR_LUMA_LIFT`])
/// the line is a bright, crisp edge that still carries the surface hue (a red wall
/// keeps a red-ish outline), while retaining enough ink that the edge stays well
/// above the QA invisibility floor on dark surfaces. The MATERIAL is the primary
/// read; the line is a tasteful, mostly-hue-true accent. Tune via `TS_CONTOUR_INKMIX`.
pub const CONTOUR_INK_MIX: f32 = 0.45;

/// Brightness multiplier applied to the surface fg BEFORE the small ink mix, so the
/// outer silhouette reads as a bright, crisp stroke WITHOUT being de-hued toward
/// white. This is the key to keeping the edge high-contrast (the depth-legibility
/// win) while preserving the surface hue (the mush fix): we brighten the surface's
/// OWN colour rather than replacing it with grey-white ink. `2.2` ⇒ the edge is
/// ~2.2× brighter than its surface (clamped per-channel), hue intact, then leans a
/// touch toward the ink. Chosen via `re_imgqa`: at 2.2 the boundary colour contrast
/// (separation) recovers materially over the surface hue WITHOUT any rise in
/// near-field invisibility. Tune via `TS_CONTOUR_LIFT`.
pub const CONTOUR_LUMA_LIFT: f32 = 2.2;

/// Background behind a contour glyph: a fraction of the cell's own bg, so the line
/// sits in a deepened groove (reads as an edge, not a sticker) AND so the near-white
/// ink always has strong contrast against it — even where the surface bg was bright.
/// 0.35 darkens enough that a bright-walled cell's outline never collapses to
/// fg≈bg (the QA near-invisibility case the line pass would otherwise add).
pub const CONTOUR_BG_SCALE: f32 = 0.35;

// ── inner-line (material seam) hierarchy ──
//
// Inner lines are SECONDARY: the eye must read the bold silhouette first and the
// interior seams second. We keep them visibly subtler than the outer line in three
// independent ways — colour (tinted toward the surface, not white), weight (a lighter
// box-drawing stroke), and a DIM text attribute — and gate them behind a luma
// threshold so only REAL material boundaries draw, never adjacent-shade noise.

/// How much of an INNER line's fg is the contour ink vs. the cell's own surface fg.
/// Far lower than [`CONTOUR_INK_MIX`] (the outer line): an inner seam keeps most of
/// its surface colour and only leans a little toward the ink, so it reads as "a seam
/// in this material", not as a bright white rule competing with the silhouette.
///
/// Kept low (`0.16`, down from an earlier `0.34`). With the default
/// [`INNER_REPLACE_GLYPH`]` = false` the inner seam no longer OVERWRITES the
/// material's texture glyph; it only nudges fg/bg. A small ink lean is enough to
/// hint the seam — most of the surface hue is preserved.
pub const INNER_INK_MIX: f32 = 0.16;

/// Background scale behind an INNER line. Milder darkening than the outer groove
/// ([`CONTOUR_BG_SCALE`]) — an interior seam is a gentle crease, not a deep cut, so
/// it does not punch a dark hole in a flat surface. Nudged toward 1.0 (up from
/// an earlier `0.7`) so the seam crease is a whisper, not a groove.
pub const INNER_BG_SCALE: f32 = 0.85;

/// Whether an INNER (material-seam) line REPLACES the cell's body glyph with a
/// box-drawing stroke. Default **`false`**: replacing the material's own
/// texture/relief glyph on every interior seam was the dominant source of a
/// muddy look — the surface lost its glyph identity wherever two materials
/// met. With this off, an inner seam keeps the material glyph and is expressed only
/// as the subtle fg/bg tint above, so interior texture survives. Set `true` (env
/// `TS_INNER_GLYPH=1`) to restore the old box-stroke seams.
pub const INNER_REPLACE_GLYPH: bool = false;

/// Minimum fg-luma difference (0..1, Rec.601) between two solid neighbours for their
/// material/surface change to count as an INNER line. Below this the boundary is
/// trivial (two near-identical shades of the same material) and draws nothing — this
/// keeps the interior minimalist: only genuine material boundaries seam.
///
/// Raised from an earlier `0.16` to `0.22`. Now that the inner seam is a soft
/// tint (not a glyph replacement) it is cheap to draw, but we still only want it on
/// a CLEARLY different material — a higher bar keeps near-shades of one material
/// from accumulating tint noise across a flat surface.
pub const INNER_LUMA_THRESHOLD: f32 = 0.22;

// ── colour-volume model ──

/// Peak brightness swing (± fraction) the volume shift may apply. Small on
/// purpose: this is a *sense* of volume, not a relight — we must not reintroduce
/// the noise the minimalist retune removed.
///
/// Cut from an earlier `0.14` to `0.05`. The ±14 % per-face brightness shift
/// on every flat interior cell was muddying flat surfaces (a wall's face read as
/// two different colours for no structural reason). At ±5 % the side cue is a faint
/// whisper that still separates the two box faces without muddying the field. To
/// remove the cue entirely, set `color_volume = false` (env `TS_COLOR_VOLUME=0`).
pub const VOLUME_BRIGHT_SWING: f32 = 0.05;

/// Per-`side` brightness bias for the two wall faces. NS faces (`side == 0`) read
/// slightly brighter, EW faces (`side == 1`) slightly darker — the same cheap
/// "two faces of a box catch light differently" cue wolf3d used, but applied
/// gently in colour space so flat walls gain relief without a per-cell glyph.
/// Indexed by `side`.
pub const VOLUME_SIDE_BIAS: [f32; 2] = [1.0 + VOLUME_BRIGHT_SWING, 1.0 - VOLUME_BRIGHT_SWING];

/// Depth at/under which the volume "near boost" is at full strength, and the depth
/// over which it decays to neutral. Nearer cells get a small extra brightness so a
/// surface advancing toward the camera reads as advancing (volume via depth).
pub const VOLUME_NEAR_DEPTH: f32 = 2.0;
/// Depth beyond which the volume "near boost" has fully decayed to neutral.
pub const VOLUME_FAR_DEPTH: f32 = 14.0;
/// Extra brightness at `VOLUME_NEAR_DEPTH` (decays to 0 by `VOLUME_FAR_DEPTH`).
/// Kept small to match [`VOLUME_BRIGHT_SWING`]'s reduced swing — the near-boost
/// is a faint advance cue, not a relight, so it cannot muddy a flat surface.
pub const VOLUME_NEAR_BOOST: f32 = 0.03;

// ─────────────────────────────── config ──────────────────────────────────────

/// Which boundaries the contour pass draws as lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContourMode {
    /// No contour glyphs at all (the pass is a colour-volume-only / no-op for
    /// glyphs). Reproduces the pre-contour glyph field.
    Off,
    /// OUTER lines only: silhouettes (a finite surface against open sky) and strong
    /// DEPTH discontinuities (object/corner edges). Interior material changes draw
    /// NO line even when `inner_lines` is set — the strictest minimalist look.
    Silhouette,
    /// DEFAULT: OUTER silhouette + depth lines PLUS — when `inner_lines` is set —
    /// SECONDARY material/surface-kind seams (a node against a wall, a decal edge),
    /// rendered subtler than the outer silhouette so the hierarchy reads. With
    /// `inner_lines = false` this is identical to [`Silhouette`](Self::Silhouette).
    #[default]
    SilhouettePlusMaterial,
}

/// How a detected 45° silhouette SLOPE is drawn — the de-angularization vocabulary
/// (`slope.rs`). `Off` keeps the original stair-step box corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SlopeStyle {
    /// No slope handling: a diagonal run renders as the original staircase of box
    /// corners (`┐ └ …`). The pre-de-angularization look.
    Off,
    /// Replace the stair corner with the 45° diagonal OUTLINE stroke `╱`/`╲`. Best
    /// for a thin (one-cell-wide) silhouette where there is no solid mass to fill.
    Line,
    /// DEFAULT: fill the SOLID half of the corner cell with a chamfer TRIANGLE
    /// `◤ ◥ ◣ ◢` so a FILLED silhouette (a solid roof / pyramid flank) reads as a
    /// continuous slope. The bigger win for solid geometry.
    #[default]
    Triangle,
}

// ── active contour config (explicit per-cast, never env in the hot path) ────────
//
// `composite_render_sys` used to call `ContourConfig::from_env()` EVERY FRAME —
// a dozen `env::var` lookups plus String allocations on the hot path. Same
// pattern as `depth_contrast` / `lod`: the config is
// a Bevy resource seeded ONCE at plugin build, installed into a thread-local by the
// chained sync system before the cast, and read back here. Set ONLY explicitly; a
// thread that never had one applied (a `--dump`, a unit test) falls back to the env
// read, so every pre-existing path behaves exactly as before.

thread_local! {
    static ACTIVE_CONTOUR: std::cell::Cell<Option<ContourConfig>> = const { std::cell::Cell::new(None) };
}

/// The ACTIVE contour config for the current thread: the one last installed by
/// [`ContourConfig::apply`], else a fresh [`ContourConfig::from_env`].
#[inline]
pub fn active() -> ContourConfig {
    ACTIVE_CONTOUR
        .with(|c| c.get())
        .unwrap_or_else(ContourConfig::from_env)
}

impl ContourConfig {
    /// Install this config as the ACTIVE one the live render reads via [`active`].
    /// Called by the composite sync system each frame from the resource (a `Copy`),
    /// or directly by a test.
    #[inline]
    pub fn apply(&self) {
        ACTIVE_CONTOUR.with(|c| c.set(Some(*self)));
    }
}

/// The contour + colour-volume tunables, as a `Copy` config. Seeded from `TS_*`
/// env at plugin build (dev / manual tuning), constructed directly in tests.
/// A Bevy `Resource` so the live render reads it (via
/// [`active`]) instead of re-reading the environment every frame.
#[derive(Clone, Copy, Debug, PartialEq, bevy::ecs::resource::Resource)]
pub struct ContourConfig {
    /// Which boundaries become line glyphs (see [`ContourMode`]).
    pub mode: ContourMode,
    /// Relative depth jump for a structural edge (see [`DEPTH_EDGE_REL_JUMP`]).
    pub depth_rel_jump: f32,
    /// Contour line fg ink (see [`CONTOUR_FG`]).
    pub line_fg: u32,
    /// Fraction of the OUTER line fg that is the white ink vs. the (luma-lifted)
    /// surface fg (see [`CONTOUR_INK_MIX`]). Low ⇒ the edge keeps its surface hue.
    pub line_ink_mix: f32,
    /// Brightness multiplier on the surface fg before the ink mix, so a hue-true
    /// edge is still bright/crisp (see [`CONTOUR_LUMA_LIFT`]).
    pub line_luma_lift: f32,
    /// Draw INNER (material-seam) lines as a secondary hierarchy. On by default but
    /// clearly subtler than outer lines; gated by [`inner_luma_threshold`](Self::inner_luma_threshold).
    /// Independent of [`ContourMode`]: when `false`, only OUTER silhouette/depth
    /// lines draw regardless of mode. `SilhouettePlusMaterial` with `inner_lines =
    /// false` is just the silhouette look.
    pub inner_lines: bool,
    /// Minimum fg-luma difference for a material change to draw an INNER line (see
    /// [`INNER_LUMA_THRESHOLD`]). Higher ⇒ fewer, only-strong inner seams.
    pub inner_luma_threshold: f32,
    /// Whether an INNER seam REPLACES the cell's material glyph with a box stroke
    /// (see [`INNER_REPLACE_GLYPH`]). Default `false`: the seam keeps the
    /// material glyph and is drawn only as a subtle fg/bg tint, so interior texture
    /// is preserved. `true` restores the old glyph-replacing seam.
    pub inner_replace_glyph: bool,
    /// Enable the subtle colour-volume shift on non-edge cells.
    pub color_volume: bool,
    /// Peak brightness swing for the colour-volume shift (see [`VOLUME_BRIGHT_SWING`]).
    pub volume_swing: f32,
    /// How a 45° silhouette slope is drawn (de-angularization, see [`SlopeStyle`]).
    /// Only affects OUTER silhouette CORNER cells that sit on a diagonal run; an
    /// axis-aligned wall is untouched. `Off` reproduces the stair-step box corners.
    pub slope_style: SlopeStyle,
    /// The swappable diagonal/triangle glyph table for slopes (smart-materials §5).
    /// `SlopeTheme::UNICODE` ships; the present layer may pass `ASCII` to degrade.
    pub slope_theme: SlopeTheme,
    /// Whether a FLAT silhouette top is crowned by surface kind (see [`CrownStyle`]).
    /// `Off` by default — the crown is an un-reviewed LOOK proposal, so it changes
    /// nothing until an explicit tuning pass turns it on.
    pub crown_style: CrownStyle,
    /// The crown glyph table used when [`crown_style`](Self::crown_style) is
    /// [`CrownStyle::On`].
    pub crown_theme: CrownTheme,
}

impl Default for ContourConfig {
    /// The SHIPPING look: bold OUTER silhouette + depth-edge lines, subtle INNER
    /// material seams (secondary hierarchy), colour-volume on.
    fn default() -> Self {
        ContourConfig {
            mode: ContourMode::SilhouettePlusMaterial,
            depth_rel_jump: DEPTH_EDGE_REL_JUMP,
            line_fg: CONTOUR_FG,
            line_ink_mix: CONTOUR_INK_MIX,
            line_luma_lift: CONTOUR_LUMA_LIFT,
            inner_lines: true,
            inner_luma_threshold: INNER_LUMA_THRESHOLD,
            inner_replace_glyph: INNER_REPLACE_GLYPH,
            color_volume: true,
            volume_swing: VOLUME_BRIGHT_SWING,
            slope_style: SlopeStyle::Triangle,
            slope_theme: SlopeTheme::UNICODE,
            crown_style: CrownStyle::Off,
            crown_theme: CrownTheme::PROPOSAL,
        }
    }
}

impl ContourConfig {
    /// A fully-disabled config: no line glyphs, no colour-volume. The pass becomes
    /// a strict no-op (byte-identical input → output). Used by tests asserting the
    /// pre-pass framebuffer and as the `Off` rung.
    pub const OFF: ContourConfig = ContourConfig {
        mode: ContourMode::Off,
        depth_rel_jump: DEPTH_EDGE_REL_JUMP,
        line_fg: CONTOUR_FG,
        line_ink_mix: CONTOUR_INK_MIX,
        line_luma_lift: CONTOUR_LUMA_LIFT,
        inner_lines: false,
        inner_luma_threshold: INNER_LUMA_THRESHOLD,
        inner_replace_glyph: INNER_REPLACE_GLYPH,
        color_volume: false,
        volume_swing: VOLUME_BRIGHT_SWING,
        slope_style: SlopeStyle::Off,
        slope_theme: SlopeTheme::UNICODE,
        crown_style: CrownStyle::Off,
        crown_theme: CrownTheme::PROPOSAL,
    };

    /// True iff this config would change any cell (either pass enabled). When false
    /// the caller can skip the pass entirely.
    #[inline]
    pub fn is_active(self) -> bool {
        self.mode != ContourMode::Off || self.color_volume
    }

    /// Seed a config from the `TS_*` env vars for dev / manual tuning. Unset
    /// vars keep the field [`Default`]. The ONLY place this module reads the env.
    ///
    /// * `TS_CONTOUR` — `off` | `silhouette` | `material` (the mode).
    /// * `TS_CONTOUR_JUMP=<f32>` — the relative depth-jump threshold (`> 0`).
    /// * `TS_CONTOUR_INKMIX=<f32>` — outer-line white-ink mix fraction in `[0, 1]`.
    /// * `TS_CONTOUR_LIFT=<f32>` — outer-line surface-fg brightness lift in `[0.5, 4]`.
    /// * `TS_CONTOUR_CROWN` — `off` | `on`: crown a flat silhouette TOP by surface
    ///   kind ([`CrownStyle`]). `off` is the default and the shipped look.
    /// * `TS_INNER_LINES` — `0`/`false` disables the secondary inner material seams.
    /// * `TS_INNER_LUMA=<f32>` — inner-seam luma threshold in `[0, 1]`.
    /// * `TS_INNER_GLYPH` — `1`/`true` lets an inner seam REPLACE the material glyph
    ///   with a box stroke (default `0`: seam is a tint only, material glyph kept).
    /// * `TS_COLOR_VOLUME` — `0`/`false` disables the colour-volume shift.
    /// * `TS_VOLUME_SWING=<f32>` — the brightness swing in `[0, 0.5]`.
    /// * `TS_SLOPE` — `off` | `line` | `triangle` (the slope de-angularization style).
    /// * `TS_SLOPE_THEME` — `unicode` | `ascii` (the slope glyph vocabulary).
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(raw) = std::env::var("TS_CONTOUR") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "off" | "none" | "0" => cfg.mode = ContourMode::Off,
                "silhouette" | "sil" | "1" => cfg.mode = ContourMode::Silhouette,
                "material" | "mat" | "2" => cfg.mode = ContourMode::SilhouettePlusMaterial,
                _ => {}
            }
        }
        if let Ok(raw) = std::env::var("TS_CONTOUR_JUMP") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && v > 0.0 {
                    cfg.depth_rel_jump = v;
                }
            }
        }
        if let Ok(raw) = std::env::var("TS_CONTOUR_INKMIX") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && (0.0..=1.0).contains(&v) {
                    cfg.line_ink_mix = v;
                }
            }
        }
        if let Ok(raw) = std::env::var("TS_CONTOUR_LIFT") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && (0.5..=4.0).contains(&v) {
                    cfg.line_luma_lift = v;
                }
            }
        }
        if let Ok(raw) = std::env::var("TS_CONTOUR_CROWN") {
            let t = raw.trim();
            if t == "1" || t.eq_ignore_ascii_case("on") || t.eq_ignore_ascii_case("true") {
                cfg.crown_style = CrownStyle::On;
            } else if t == "0" || t.eq_ignore_ascii_case("off") || t.eq_ignore_ascii_case("false") {
                cfg.crown_style = CrownStyle::Off;
            }
        }
        if let Ok(raw) = std::env::var("TS_INNER_LINES") {
            let t = raw.trim();
            if t == "0" || t.eq_ignore_ascii_case("false") {
                cfg.inner_lines = false;
            } else if t == "1" || t.eq_ignore_ascii_case("true") {
                cfg.inner_lines = true;
            }
        }
        if let Ok(raw) = std::env::var("TS_INNER_LUMA") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && (0.0..=1.0).contains(&v) {
                    cfg.inner_luma_threshold = v;
                }
            }
        }
        if let Ok(raw) = std::env::var("TS_INNER_GLYPH") {
            let t = raw.trim();
            if t == "0" || t.eq_ignore_ascii_case("false") {
                cfg.inner_replace_glyph = false;
            } else if t == "1" || t.eq_ignore_ascii_case("true") {
                cfg.inner_replace_glyph = true;
            }
        }
        if let Ok(raw) = std::env::var("TS_COLOR_VOLUME") {
            let t = raw.trim();
            if t == "0" || t.eq_ignore_ascii_case("false") {
                cfg.color_volume = false;
            } else if t == "1" || t.eq_ignore_ascii_case("true") {
                cfg.color_volume = true;
            }
        }
        if let Ok(raw) = std::env::var("TS_VOLUME_SWING") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && (0.0..=0.5).contains(&v) {
                    cfg.volume_swing = v;
                }
            }
        }
        if let Ok(raw) = std::env::var("TS_SLOPE") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "off" | "none" | "0" => cfg.slope_style = SlopeStyle::Off,
                "line" | "1" => cfg.slope_style = SlopeStyle::Line,
                "triangle" | "tri" | "2" => cfg.slope_style = SlopeStyle::Triangle,
                _ => {}
            }
        }
        if let Ok(raw) = std::env::var("TS_SLOPE_THEME") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "ascii" => cfg.slope_theme = SlopeTheme::ASCII,
                "unicode" | "uni" => cfg.slope_theme = SlopeTheme::UNICODE,
                _ => {}
            }
        }
        cfg
    }
}

// ─────────────────────────── box-drawing line set ────────────────────────────

/// A 4-bit edge-occupancy mask → box-drawing glyph (marching-squares style). The
/// bit for each direction is set when there is a STRUCTURAL edge between this cell
/// and that orthogonal neighbour, i.e. the cell sits ON the contour and the line
/// runs ALONG the boundary. The glyph is chosen so the drawn strokes connect the
/// directions that have a neighbour edge.
///
/// Bit layout: `N = 1<<0`, `E = 1<<1`, `S = 1<<2`, `W = 1<<3`.
///
/// `pub(crate)` so the authored-line post-pass ([`crate::lines`]) builds the SAME
/// edge-occupancy masks and reuses [`contour_glyph_weighted`] — one box-drawing
/// vocabulary across the auto-contour and the authored strokes (no duplicate table).
pub(crate) const N: u8 = 1 << 0;
pub(crate) const E: u8 = 1 << 1;
pub(crate) const S: u8 = 1 << 2;
pub(crate) const W: u8 = 1 << 3;

/// A CROWN table: the glyph a silhouette's TOP edge gets, keyed by what kind of
/// surface is crowned.
///
/// ## What this is for
///
/// Today every outer silhouette is drawn by orientation alone — a flat top edge is
/// `━`, whatever it is the top of. A study of a reference ASCII-city renderer
/// found that it carries an `architecture` byte per hit purely so the ROOFLINE
/// glyph varies by structure type (`= ~ ^ *`), and that this is most of what
/// makes its skyline read as different buildings rather than one extruded mass.
///
/// The same lever exists here for free: `Cell::surface` is already the "what kind of
/// thing is this" byte, and the contour pass already reads it for INNER seams. A crown
/// table maps that byte to a top-edge glyph, so a circuit wall and a matrix wall no
/// longer terminate in the same stroke.
///
/// ## Why it is OFF by default
///
/// This is a LOOK decision, not a correctness one, and the right vocabulary can only
/// be judged from rendered frames, by eye. [`CrownStyle::Off`] therefore
/// reproduces the current silhouette byte-for-byte, and the table below is a
/// starting proposal to iterate on, not a
/// shipped verdict. Swappable exactly like [`crate::slope::SlopeTheme`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrownTheme {
    /// Theme name (for manual tuning selection / debug).
    pub name: &'static str,
    /// Crown for a plain opaque wall. `None` ⇒ keep the orientation glyph.
    pub wall_solid: Option<char>,
    /// Crown for a circuit-trace wall.
    pub wall_trace: Option<char>,
    /// Crown for a matrix-rain wall.
    pub wall_matrix: Option<char>,
    /// Crown for a non-wall solid (a puzzle node, the door anchor).
    pub wall_other: Option<char>,
}

impl CrownTheme {
    /// The proposal to iterate on. Deliberately conservative: a plain wall keeps its
    /// orientation stroke (`None`), so only the surfaces that genuinely *are* a
    /// different kind of thing pick up a different top.
    ///
    /// * `WallTrace` → `~` — a circuit run reads as something that continues past the
    ///   edge rather than stopping at it.
    /// * `WallMatrix` → `*` — the rain wall's top is where the fall begins.
    /// * `WallOther` → `^` — a node/door is a discrete OBJECT; a peaked top separates
    ///   it from the architecture behind it.
    pub const PROPOSAL: CrownTheme = CrownTheme {
        name: "proposal",
        wall_solid: None,
        wall_trace: Some('~'),
        wall_matrix: Some('*'),
        wall_other: Some('^'),
    };

    /// The crown for `surface`, or `None` to keep the orientation glyph.
    #[inline]
    pub fn crown_for(&self, surface: re_core::SurfaceKind) -> Option<char> {
        use re_core::SurfaceKind as S;
        match surface {
            S::WallSolid => self.wall_solid,
            S::WallTrace => self.wall_trace,
            S::WallMatrix => self.wall_matrix,
            S::WallOther => self.wall_other,
            // Floor/Ceiling/Empty are not silhouette tops, and a Portal rim is
            // already its own strongly-styled thing — leave every one of them alone.
            _ => None,
        }
    }
}

/// Whether the silhouette's TOP edge is crowned by surface kind (see [`CrownTheme`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrownStyle {
    /// DEFAULT: no crown. The top edge is chosen by orientation alone — byte-identical
    /// to the pre-crown silhouette.
    #[default]
    Off,
    /// Crown a FLAT top edge by surface kind, using the configured [`CrownTheme`].
    On,
}

/// How a boundary between two cells is classified, deciding which hierarchy level
/// (and styling) the drawn line gets — or that no line is drawn at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeClass {
    /// No structural or material boundary here.
    None,
    /// OUTER silhouette / depth edge — a solid surface against open space, or a
    /// strong depth discontinuity. The PRIMARY (bold, near-white) line.
    Outer,
    /// INNER material seam — a real material/surface-kind change between two solid
    /// cells. The SECONDARY (subtle, surface-tinted) line.
    Inner,
}

/// Pick the box-drawing glyph for an edge mask. The contour runs PERPENDICULAR to
/// the direction of the depth step: an edge to the cell's N/S (a horizontal seam
/// above/below) is drawn as a horizontal rule `─`; an edge to the E/W (a vertical
/// seam) is drawn as a vertical rule `│`; a corner where both a vertical and a
/// horizontal seam meet is drawn as the matching corner/junction glyph.
///
/// This is the key inversion that makes outlines READ: a silhouette edge to my
/// EAST means the surface ends on my right, so the line I draw is a VERTICAL stroke
/// down my right side → `│`. A run of such cells stacks into a clean vertical
/// outline. Likewise a depth step to my NORTH (surface ends above me) draws `─`.
#[inline]
pub fn contour_glyph(mask: u8) -> char {
    contour_glyph_weighted(mask, EdgeClass::Inner)
}

/// As [`contour_glyph`], but the glyph WEIGHT encodes the hierarchy: an
/// [`EdgeClass::Outer`] boundary draws the HEAVY box-drawing set (`┃ ━ ┏ ┓ ┗ ┛ …`)
/// so the silhouette reads as the bold primary line; an [`EdgeClass::Inner`]
/// material seam draws the LIGHT set (`│ ─ ┌ ┐ └ ┘ …`) so it reads as the subtler
/// secondary line. (`EdgeClass::None` is the empty-mask sentinel.)
#[inline]
pub fn contour_glyph_weighted(mask: u8, class: EdgeClass) -> char {
    let heavy = class == EdgeClass::Outer;
    // Reduce to the two axes: is there a horizontal seam (N or S edge) and/or a
    // vertical seam (E or W edge)?
    let h = mask & (N | S); // horizontal rule wanted
    let v = mask & (E | W); // vertical rule wanted
    match (h != 0, v != 0) {
        (false, false) => ' ', // no edge — caller should not have invoked us
        (true, false) => {
            if heavy {
                '━'
            } else {
                '─'
            }
        } // only horizontal seam(s) → horizontal rule
        (false, true) => {
            if heavy {
                '┃'
            } else {
                '│'
            }
        } // only vertical seam(s) → vertical rule
        (true, true) => {
            // A corner / junction: pick the connector by exactly which sides
            // carry an edge. The stroke connects the FREE directions (the ones
            // WITHOUT an edge), forming the inside of the corner.
            if heavy {
                // Same vertical fix as the light set below: connect toward the SOLID
                // (opposite of each edge), so a silhouette box is a proper rectangle,
                // not a vertically-inverted one. An `N` (open-above) edge sends its
                // heavy vertical stroke DOWN toward the solid.
                match mask {
                    m if m == (N | E) => '┓', // open up+right → solid down-left
                    m if m == (N | W) => '┏', // open up+left  → solid down-right
                    m if m == (S | E) => '┛', // open down+right → solid up-left
                    m if m == (S | W) => '┗', // open down+left  → solid up-right
                    m if m == (N | E | W) => '┻',
                    m if m == (S | E | W) => '┳',
                    m if m == (E | N | S) => '┣',
                    m if m == (W | N | S) => '┫',
                    _ => '╋',
                }
            } else {
                match mask {
                    // single corner (one H + one V edge). The stroke connects toward
                    // the SOLID mass (the opposite of each edge direction), so the
                    // outline hugs the solid the way a box outline should. Bit `N`
                    // means "open ABOVE" (smaller row); the solid is BELOW, so an N
                    // edge draws its vertical stroke DOWNWARD (toward S). The earlier
                    // table used the SAME vertical direction as the edge, which drew an
                    // INVERTED box — top corners read as bottom corners and vice-versa
                    // (the stairs "bottom line corners point down" bug). Screen-Y grows
                    // down, world-up is +row-toward-0; this is the y-axis fix.
                    m if m == (N | E) => '┐', // open up+right → solid down-left → ┐ (S+W strokes)
                    m if m == (N | W) => '┌', // open up+left  → solid down-right → ┌ (S+E strokes)
                    m if m == (S | E) => '┘', // open down+right → solid up-left → ┘ (N+W strokes)
                    m if m == (S | W) => '└', // open down+left  → solid up-right → └ (N+E strokes)
                    // T-junctions (two collinear + one perpendicular)
                    m if m == (N | E | W) => '┴',
                    m if m == (S | E | W) => '┬',
                    m if m == (E | N | S) => '├',
                    m if m == (W | N | S) => '┤',
                    // full cross
                    _ => '┼',
                }
            }
        }
    }
}

/// The OUTER glyph for `outer`, crowned by surface kind when the cell is a flat
/// silhouette TOP and [`CrownStyle::On`] is configured.
///
/// The crown replaces ONLY the plain horizontal top stroke:
///
/// * `outer` must carry `N` (open ABOVE — this really is a top edge, not a side or an
///   underside), and
/// * `outer` must carry NO vertical edge (`E`/`W`), so a corner keeps its connector
///   `┏ ┓` and a silhouette box does not lose its shape, and
/// * the theme must offer a crown for this surface.
///
/// Any of those failing falls through to [`contour_glyph_weighted`] unchanged — which
/// is also every case when the style is [`CrownStyle::Off`], the default. That is why
/// turning the feature off is byte-identical rather than merely similar.
#[inline]
fn crown_or_line(cfg: &ContourConfig, surface: re_core::SurfaceKind, outer: u8) -> char {
    if cfg.crown_style == CrownStyle::On && outer & N != 0 && outer & (E | W) == 0 {
        if let Some(g) = cfg.crown_theme.crown_for(surface) {
            return g;
        }
    }
    contour_glyph_weighted(outer, EdgeClass::Outer)
}

// ───────────────────────────── the pass ──────────────────────────────────────

/// The orthogonal direction from the centre cell `a` to its neighbour `b`, used by
/// the single-side ownership rule to break equal-depth ties consistently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    /// Neighbour is one row ABOVE (smaller row) — `a` is the lower cell.
    North,
    /// Neighbour is one row BELOW (larger row) — `a` is the upper cell.
    South,
    /// Neighbour is one column LEFT (smaller col) — `a` is the right cell.
    West,
    /// Neighbour is one column RIGHT (larger col) — `a` is the left cell.
    East,
}

/// **Single-side (draw-once) ownership.** A boundary between the centre cell `a` and
/// its neighbour `b` (which lies in direction `dir` from `a`) must be drawn by
/// exactly ONE of the two cells, or the same line is stacked twice (the wall-base
/// doubled-line artifact). This returns `true` iff `a` is the owner.
///
/// Rule — **near/top**:
/// 1. The NEARER cell (smaller `depth`) owns the boundary. A silhouette edge thus
///    belongs to the solid surface in front, not the sky behind it; an occlusion lip
///    belongs to the near object. (An off-grid / infinite-depth neighbour means `a`
///    — finite — is nearer, so `a` owns it; but the caller already suppresses the
///    frame border, and a sky neighbour never tries to draw, so the silhouette is
///    still drawn once, by the solid side.)
/// 2. On a near-equal depth seam (a flat material boundary), the tie breaks toward
///    the UPPER/LEADING cell: for an N/S seam the upper cell (smaller row) owns it
///    → `a` owns when `b` is to its South; for an E/W seam the left cell (smaller
///    col) owns it → `a` owns when `b` is to its East.
#[inline]
fn owns_boundary(a: &Cell, b: &Cell, dir: Dir) -> bool {
    // Equal-depth tolerance: within this fraction of the nearer depth the two cells
    // are "the same plane" and the positional tie-break decides ownership. This is
    // what makes a flat wall-vs-wall / material seam draw once.
    const DEPTH_TIE_REL: f32 = 0.02;

    let (da, db) = (a.depth, b.depth);
    let near = da.min(db);
    let tie = near.is_finite() && (da - db).abs() <= DEPTH_TIE_REL * near.max(1.0);

    if !tie {
        // Distinct depths (or one infinite): the nearer cell owns the boundary.
        // NaN-safe: if da is NaN the comparison is false, so b (the finite side) wins.
        return da < db || (da.is_finite() && !db.is_finite());
    }

    // Equal-depth seam: upper/leading owns. `a` owns the seam toward South / East.
    matches!(dir, Dir::South | Dir::East)
}

/// Classify the boundary between centre cell `a` and orthogonal neighbour `b` (which
/// lies in direction `dir`), applying BOTH the structural/material test AND the
/// single-side ownership rule. Returns [`EdgeClass::None`] when there is no boundary
/// OR when this boundary is owned by the OTHER cell (so it is drawn once).
///
/// `b` is `None` at the framebuffer border — never an edge: we do NOT want a box
/// drawn around the whole viewport when a wall fills the screen.
///
/// - **OUTER**: silhouette (one side solid, the other Empty/sky) or a strong depth
///   discontinuity (relative jump over threshold, a wall involved).
/// - **INNER**: both solid, near-equal depth, a real material/surface-kind change
///   whose fg-luma difference clears `inner_luma_threshold` (only when the mode and
///   `inner_lines` allow it).
#[inline]
fn classify_edge(a: &Cell, b: Option<&Cell>, dir: Dir, cfg: &ContourConfig) -> EdgeClass {
    let a_solid = is_solid_surface(a.surface);
    let Some(b) = b else {
        // Frame border: never an edge (no viewport-rectangle outline).
        return EdgeClass::None;
    };
    let (b_solid, b_depth, b_surface, b_material) =
        (is_solid_surface(b.surface), b.depth, b.surface, b.material);

    // Silhouette: solid against not-solid (sky / open) — an OUTER edge, owned by the
    // nearer (solid) side so it is drawn once.
    if a_solid != b_solid {
        return if owns_boundary(a, b, dir) {
            EdgeClass::Outer
        } else {
            EdgeClass::None
        };
    }

    // Both open (sky/floor-less): never an edge here.
    if !a_solid {
        return EdgeClass::None;
    }

    // Both solid: a WALL meeting a FLOOR is ALWAYS a genuine structural seam — the
    // wall's own base edge — drawn unconditionally, independent of the depth reading
    // at that row. `clamp_to_wall_depth` (`lib.rs`) intentionally bounds a floor row's
    // classic per-row depth to a nearer wall's own `perp` (the dark-strip fix), so a
    // wall-base row and the floor row right below it can now read as the SAME depth
    // (or arbitrarily close) — the depth JUMP this pass used to key off to draw the
    // wall-base contact line is exactly what that fix collapsed, so a jump test can
    // no longer be trusted to signal "a wall stands here"; the surface-kind change is
    // the real, always-present signal. (A wall meeting a CEILING doesn't need this
    // case: `Ceiling` is not [`is_solid_surface`], so that boundary already draws
    // unconditionally via the silhouette branch above.)
    if a.surface.is_wall() && b_surface == SurfaceKind::Floor
        || a.surface == SurfaceKind::Floor && b_surface.is_wall()
    {
        return if owns_boundary(a, b, dir) {
            EdgeClass::Outer
        } else {
            EdgeClass::None
        };
    }

    // Both solid, neither side a wall-vs-floor pair: a strong DEPTH discontinuity is
    // an OUTER corner / occlusion edge — BUT only when a WALL is involved (a
    // wall-vs-wall corner, or a wall-vs-portal seam). A floor/ceiling receding from
    // the camera has large per-row depth steps near the viewer that are NOT
    // structure; drawing them would stripe the floor. So floor↔floor / ceiling↔ceiling
    // depth jumps are ignored.
    let wall_involved = a.surface.is_wall() || b_surface.is_wall();
    if wall_involved {
        let near = a.depth.min(b_depth);
        let jump = (a.depth - b_depth).abs();
        if near.is_finite() && jump >= DEPTH_EDGE_MIN_ABS && jump >= cfg.depth_rel_jump * near {
            // Depth edge: the NEAR cell owns it (the near surface's occluding lip).
            return if owns_boundary(a, b, dir) {
                EdgeClass::Outer
            } else {
                EdgeClass::None
            };
        }
    }

    // INNER material seam: a real material/surface-kind change between two near-equal
    // depth solid cells. Gated by mode + the inner-lines toggle + a luma threshold so
    // only genuine boundaries seam, not adjacent shades of one material. Owned by the
    // upper/leading cell (the equal-depth tie-break) so it is drawn once.
    if cfg.mode == ContourMode::SilhouettePlusMaterial
        && cfg.inner_lines
        && (a.surface != b_surface || a.material != b_material)
        && luma_diff(a.fg, b.fg) >= cfg.inner_luma_threshold
    {
        return if owns_boundary(a, b, dir) {
            EdgeClass::Inner
        } else {
            EdgeClass::None
        };
    }

    EdgeClass::None
}

/// Rec.601 luma difference between two packed `0x00RRGGBB` colours, in `0..1`. Used
/// to gate inner-line seams so only a perceptible material boundary draws.
#[inline]
fn luma_diff(a: u32, b: u32) -> f32 {
    #[inline]
    fn luma(c: u32) -> f32 {
        let v = unpack_rgb(c); // components already in 0..1
        0.299 * v.x + 0.587 * v.y + 0.114 * v.z
    }
    (luma(a) - luma(b)).abs()
}

/// True for the surfaces that participate in a silhouette as "solid" — anything the
/// DDA actually hit (walls, portals, floor). The sky/ceiling Empty is the "open"
/// side. Floor counts as solid so the floor-vs-sky horizon and a wall-vs-floor lip
/// read as edges; ceiling/empty is open.
#[inline]
fn is_solid_surface(s: SurfaceKind) -> bool {
    matches!(
        s,
        SurfaceKind::WallSolid
            | SurfaceKind::WallTrace
            | SurfaceKind::WallMatrix
            | SurfaceKind::WallOther
            | SurfaceKind::Floor
            | SurfaceKind::Portal
    )
}

/// The colour-volume brightness multiplier for a non-edge cell: a gentle product of
/// a `side`-face bias and a depth near-boost, clamped to the configured swing so it
/// can never overpower the surface colour.
#[inline]
fn volume_factor(cell: &Cell, cfg: &ContourConfig) -> f32 {
    // Face bias scaled to the configured swing (the const table is at the default
    // swing; rescale so a custom swing still lands in range).
    let side = (cell.side as usize) & 1;
    let base_bias = VOLUME_SIDE_BIAS[side] - 1.0; // ±VOLUME_BRIGHT_SWING
    let face = 1.0 + base_bias * (cfg.volume_swing / VOLUME_BRIGHT_SWING).clamp(0.0, 4.0);

    // Depth near-boost: full at VOLUME_NEAR_DEPTH, 0 by VOLUME_FAR_DEPTH.
    let near_boost = if cell.depth.is_finite() {
        let t = ((VOLUME_FAR_DEPTH - cell.depth) / (VOLUME_FAR_DEPTH - VOLUME_NEAR_DEPTH))
            .clamp(0.0, 1.0);
        1.0 + VOLUME_NEAR_BOOST * t
    } else {
        1.0
    };

    (face * near_boost).clamp(1.0 - cfg.volume_swing - 0.05, 1.0 + cfg.volume_swing + 0.1)
}

/// The SUB-45° (2:1) half-step glyph for a TOP-edge silhouette cell at `(c, r)`, or
/// `None` if the cell is NOT on a genuine 2:1 shallow run (a flat top or a 45° run, both
/// of which must keep their existing glyph). `solid_at(col, row)` reports silhouette
/// solidity (off-grid → open), the same closure the 45° path uses.
///
/// Builds the `top_row(col)` sampler [`slope::classify_shallow_slope`] needs: the row of
/// the silhouette TOP boundary (topmost solid cell whose cell above is OPEN) near row `r`.
/// Scanning a small ±[`SHALLOW_SCAN`] window keeps it local — a far-away unrelated
/// silhouette of the same column never contaminates the run.
#[inline]
fn shallow_slope_glyph<F: Fn(isize, isize) -> bool>(
    c: isize,
    r: isize,
    solid_at: &F,
) -> Option<char> {
    /// Half-height of the vertical window scanned for the silhouette top boundary. A
    /// 2:1 run rises at most one row per two columns, so over the ±2-column detector
    /// span the boundary moves at most ±1 row — a ±3 window comfortably contains it
    /// while staying local (it will not lock onto a distant silhouette of the same col).
    const SHALLOW_SCAN: isize = 3;

    // top_row(col): the smallest row in [r-SCAN, r+SCAN] that is solid with OPEN above
    // (the silhouette's top edge for this column), or None if the column has no such
    // boundary in the window (fully open, or fully solid top-to-bottom in the window).
    // Returned as `i32` for the detector (grid coords always fit).
    let top_row = |col: i32| -> Option<i32> {
        let col = col as isize;
        for sr in (r - SHALLOW_SCAN)..=(r + SHALLOW_SCAN) {
            if solid_at(col, sr) && !solid_at(col, sr - 1) {
                return Some(sr as i32);
            }
        }
        None
    };

    let (dir, leading) = slope::classify_shallow_slope(c as i32, top_row)?;
    // Roofline (open ABOVE, solid BELOW) → the TOP-edge glyph family (filled lower band
    // on the trailing half, never an empty gap).
    Some(slope::subcell_half_step_edge(dir, leading, true))
}

/// Apply the contour + colour-volume pass over `fb` in place.
///
/// Two phases over the SAME read-only snapshot of the input (we read neighbours
/// from the original grid, never from already-mutated cells, so the result is
/// order-independent and deterministic):
///
/// 1. Detect + CLASSIFY edges per cell from the four neighbours into OUTER
///    (silhouette / depth) and INNER (material seam) masks, applying the single-side
///    ownership rule so each boundary is drawn exactly once. OUTER dominates a cell
///    that also has an inner seam — the silhouette is the primary line. The mask
///    decides the box-drawing glyph; the class decides the fg/bg/weight hierarchy.
/// 2. Non-edge solid cells get the colour-volume brightness shift.
///
/// To read neighbours from the unmutated grid without cloning the whole
/// framebuffer, we process row by row keeping the PREVIOUS row's original cells in
/// a small reused scratch buffer (the only state). Within a row we read left/right
/// from the not-yet-written cells and up/down from the prev/next original rows.
pub fn apply(fb: &mut Framebuffer, cfg: &ContourConfig) {
    if !cfg.is_active() {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }

    // We need each cell's ORIGINAL neighbours. Snapshot the whole grid once: it is a
    // cheap Vec<Cell> copy (one allocation for the whole pass, not per cell), and it
    // lets us read N/E/S/W originals trivially. The render entry points run this
    // once per frame; the snapshot is far cheaper than the cast that produced it.
    let snapshot: Vec<Cell> = fb.cells().to_vec();
    let at = |c: usize, r: usize| -> &Cell { &snapshot[r * cols + c] };

    for r in 0..rows {
        for c in 0..cols {
            let cell = at(c, r);

            // SKIP cells that already carry an AUTHORED structural overlay — the
            // hologram wireframe / scanline pass writes its own line glyphs onto the
            // scene cell and tags them with the CURLY scan-glow underline. Those are
            // already meaningful structure (the holo's own outline); the auto-contour
            // must not stomp them, and the colour-volume must not recolour them.
            if cell.style.effective_underline_style() == UnderlineStyle::Curly {
                continue;
            }

            // Build the edge mask from the four orthogonal neighbours (off-grid =
            // no edge). Only when contour glyphs are enabled.
            if cfg.mode != ContourMode::Off {
                let n = if r > 0 { Some(at(c, r - 1)) } else { None };
                let s = if r + 1 < rows {
                    Some(at(c, r + 1))
                } else {
                    None
                };
                let w = if c > 0 { Some(at(c - 1, r)) } else { None };
                let e = if c + 1 < cols {
                    Some(at(c + 1, r))
                } else {
                    None
                };

                // Only solid cells carry an outline (the sky side stays clean).
                if is_solid_surface(cell.surface) {
                    // Two masks: OUTER (silhouette/depth) and INNER (material seam).
                    // Each direction's class is decided by `classify_edge`, which has
                    // ALREADY applied the single-side ownership rule, so a boundary
                    // sets a bit in exactly ONE of the two adjacent cells.
                    let mut outer = 0u8;
                    let mut inner = 0u8;
                    let mut bit = |dir: Dir, nb: Option<&Cell>, b: u8| match classify_edge(
                        cell, nb, dir, cfg,
                    ) {
                        EdgeClass::Outer => outer |= b,
                        EdgeClass::Inner => inner |= b,
                        EdgeClass::None => {}
                    };
                    bit(Dir::North, n, N);
                    bit(Dir::East, e, E);
                    bit(Dir::South, s, S);
                    bit(Dir::West, w, W);

                    // OUTER dominates: the silhouette is the PRIMARY line. If this
                    // cell owns any outer boundary, draw the bold near-white heavy
                    // stroke for the outer mask (an inner seam coincident with a
                    // silhouette is subsumed by the silhouette).
                    if outer != 0 {
                        // DE-ANGULARIZATION: a corner cell on a 45° diagonal run is a
                        // SLOPE, not a stair step. Re-examine the 8 neighbours; if the
                        // silhouette marches diagonally, replace the box corner with a
                        // diagonal/triangle slope glyph from the swappable theme. Only
                        // the pure corner masks reach `classify_slope`; straight runs
                        // and junctions fall through to the box-drawing set unchanged.
                        let glyph = if cfg.slope_style != SlopeStyle::Off {
                            // A reference binding so the same closure feeds both the 45°
                            // neighbour gather and the 2:1 top-row sampler without a move
                            // (and without a needless per-call borrow).
                            let solid_at = &|sc: isize, sr: isize| -> bool {
                                // Off-grid → OPEN (not solid): a slope running off the
                                // viewport edge still reads as a slope, not a wall.
                                if sc < 0 || sr < 0 || sc as usize >= cols || sr as usize >= rows {
                                    false
                                } else {
                                    is_solid_surface(at(sc as usize, sr as usize).surface)
                                }
                            };
                            let nb8 = slope::gather_nb8(c, r, solid_at);
                            match slope::classify_slope(outer, nb8) {
                                Some(dir) => match cfg.slope_style {
                                    SlopeStyle::Triangle => cfg.slope_theme.triangle_for(dir),
                                    SlopeStyle::Line => cfg.slope_theme.line_for(dir),
                                    SlopeStyle::Off => {
                                        contour_glyph_weighted(outer, EdgeClass::Outer)
                                    }
                                },
                                // Not a 45° corner. If this is a TOP-edge cell (open
                                // ABOVE — the `N` outer bit) try the SUB-45° (2:1)
                                // half-step: a SHALLOW roofline the 45° path can't
                                // express. The run-detector (`classify_shallow_slope`)
                                // fires ONLY on a genuine 2:1 staircase — a flat top or
                                // a 45° run returns None and falls through to the box
                                // glyph unchanged, so flat wall tops are never shattered.
                                None if outer & N != 0 => {
                                    match shallow_slope_glyph(c as isize, r as isize, solid_at) {
                                        Some(g) => g,
                                        None => crown_or_line(cfg, cell.surface, outer),
                                    }
                                }
                                None => crown_or_line(cfg, cell.surface, outer),
                            }
                        } else {
                            crown_or_line(cfg, cell.surface, outer)
                        };
                        let mut out = *cell;
                        out.glyph = u32::from(glyph);
                        // Hue-true bright edge: LIFT the surface fg's luma (hue
                        // preserved — it is a scalar multiply), THEN lean a little
                        // toward the white ink. A red wall's outline stays red, just
                        // brighter and crisper — the mush fix's core move.
                        let lifted = scale_rgb(cell.fg, cfg.line_luma_lift);
                        out.fg = mix_rgb(lifted, cfg.line_fg, cfg.line_ink_mix);
                        out.bg = scale_rgb(cell.bg, CONTOUR_BG_SCALE);
                        // Outer line is bold structure; drop per-cell style, set BOLD
                        // so the silhouette reads as the heaviest ink in the frame.
                        // GLASS-FIX: but if the source cell carried a colored UNDERLINE
                        // (the pipe "glass tube" meniscus from `pipe_text::overlay`),
                        // CARRY it through — the glass edge must survive on the pipe's
                        // contour OUTLINE, not only on its text-body cells. Curly (holo)
                        // underlines never reach here (skipped above), so this preserves
                        // only the pipe-glass underline; cells WITHOUT underline are
                        // unchanged (plain BOLD), so the vast majority of cells, and
                        // every golden, are byte-identical.
                        out.style = contour_style_preserving_underline(
                            cell.style,
                            re_core::CellStyle::BOLD,
                        );
                        fb.set(c, r, out);
                        continue;
                    }

                    // INNER: a SECONDARY material seam. The seam is a SUBTLE
                    // TINT by default — it does NOT overwrite the
                    // material's body glyph, so the surface keeps its texture/relief
                    // identity at the seam. Only fg/bg are nudged (a faint crease).
                    // With `inner_replace_glyph` it reverts to the old behaviour: a
                    // light box stroke replacing the glyph, with the DIM attribute.
                    if inner != 0 {
                        let mut out = *cell;
                        out.fg = mix_rgb(cell.fg, cfg.line_fg, INNER_INK_MIX);
                        out.bg = scale_rgb(cell.bg, INNER_BG_SCALE);
                        if cfg.inner_replace_glyph {
                            out.glyph = u32::from(contour_glyph_weighted(inner, EdgeClass::Inner));
                            // GLASS-FIX: same underline carry-through as the outer line —
                            // a glass-pipe underline survives the box-stroke restyle.
                            out.style = contour_style_preserving_underline(
                                cell.style,
                                re_core::CellStyle::DIM,
                            );
                        }
                        // When NOT replacing the glyph we keep the cell's own glyph
                        // and style untouched — the material reads through the tint.
                        fb.set(c, r, out);
                        continue; // edge cell handled; no colour-volume on it.
                    }
                }
            }

            // Non-edge: subtle colour-volume on solid cells only (sky untouched).
            if cfg.color_volume && is_solid_surface(cell.surface) {
                let f = volume_factor(cell, cfg);
                if (f - 1.0).abs() > f32::EPSILON {
                    let mut out = *cell;
                    // Scale fg by the full volume factor (the brightness cue). The bg
                    // is anchored: a DARKENING face scales bg only HALF as far toward
                    // black, so the fg↔bg gap never shrinks (and slightly widens) —
                    // this keeps the volume shift from pushing a marginal cell under
                    // the QA invisibility threshold. A BRIGHTENING face scales bg in
                    // full so the whole surface lifts together.
                    out.fg = scale_rgb(cell.fg, f);
                    let bg_f = if f < 1.0 { 1.0 - (1.0 - f) * 0.5 } else { f };
                    out.bg = scale_rgb(cell.bg, bg_f);
                    fb.set(c, r, out);
                }
            }
        }
    }
}

/// Build the contour cell's replacement style: the contour `base_flags` (BOLD for the
/// outer silhouette, DIM for an inner box-stroke seam) PLUS — when the SOURCE cell
/// carried a colored underline — that underline carried through intact.
///
/// GLASS-FIX: the pipe "glass tube" effect (`pipe_text::overlay`) tags powered-pipe
/// text cells with a colored single underline (the lit meniscus). Before this fix the
/// contour pass replaced the whole style with just the contour flag, stripping the
/// underline + its colour, so the glass edge vanished on the pipe's OUTLINE cells (it
/// showed only on the interior text body). We now detect the incoming underline and
/// preserve its FLAG, its packed underline COLOUR, and its explicit SHAPE so the glass
/// rule reads continuously across the contour. The Curly (hologram scanline) underline
/// is handled earlier (those cells are skipped wholesale), so it never reaches here.
///
/// A source cell with NO underline (`is_underline() == false` and no underline colour)
/// gets exactly the old style — `base_flags` over `NONE`. That is the overwhelming
/// majority of cells and every golden in the corpus, so the output is byte-identical
/// for them.
#[inline]
fn contour_style_preserving_underline(
    src: re_core::CellStyle,
    base_flags: u8,
) -> re_core::CellStyle {
    let mut out = re_core::CellStyle {
        flags: base_flags,
        ..re_core::CellStyle::NONE
    };
    // Carry the underline only when the source actually had one (flag set OR a real
    // colour). Otherwise leave the clean base style untouched (no behaviour change).
    if src.is_underline() || src.has_underline_color() {
        out.flags |= re_core::CellStyle::UNDERLINE;
        out.underline = src.underline;
        out.underline_style = src.underline_style;
    }
    out
}

/// Multiply a packed `0x00RRGGBB` by `factor`, clamped per channel. Brightness-only
/// (hue stable): the volume cue is a luminance nudge, never a hue shift, so the
/// element colour still reads as itself.
#[inline]
fn scale_rgb(c: u32, factor: f32) -> u32 {
    pack_rgb(unpack_rgb(c) * factor)
}

/// Linear blend of two packed colours: `a*(1-t) + b*t`.
#[inline]
fn mix_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    pack_rgb(unpack_rgb(a) * (1.0 - t) + unpack_rgb(b) * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::Cell;

    fn solid(depth: f32, side: u8) -> Cell {
        Cell {
            glyph: u32::from('.'),
            fg: 0x0080_4020,
            bg: 0x0010_0804,
            depth,
            side,
            surface: SurfaceKind::WallSolid,
            ..Cell::default()
        }
    }
    fn sky() -> Cell {
        Cell {
            glyph: u32::from(' '),
            fg: 0x0000_0000,
            bg: 0x0008_0810,
            depth: f32::INFINITY,
            surface: SurfaceKind::Empty,
            ..Cell::default()
        }
    }

    #[test]
    fn off_config_is_a_strict_noop() {
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                fb.set(c, r, solid(4.0, 0));
            }
        }
        let before = fb.cells().to_vec();
        apply(&mut fb, &ContourConfig::OFF);
        assert_eq!(fb.cells(), &before[..], "OFF must not touch any cell");
    }

    #[test]
    fn silhouette_against_sky_draws_a_line_glyph() {
        // A single column of wall with sky to the WEST → that wall column's left
        // edge is a vertical OUTER silhouette → the HEAVY vertical rule '┃'. The
        // near (solid) side owns the boundary; the sky side never draws.
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            fb.set(0, r, sky());
            fb.set(1, r, solid(4.0, 0));
            fb.set(2, r, solid(4.0, 0));
        }
        apply(&mut fb, &ContourConfig::default());
        // Column 1 borders sky on the west → heavy vertical rule (OUTER hierarchy).
        let mid = fb.get(1, 1).unwrap();
        assert_eq!(
            char::from_u32(mid.glyph).unwrap(),
            '┃',
            "wall flush against sky on the west must draw a HEAVY vertical contour"
        );
        // The line fg is the crisp ink, not the original dim surface fg, and the
        // OUTER line carries the BOLD attribute (the primary hierarchy).
        assert_ne!(mid.fg, solid(4.0, 0).fg);
        assert!(mid.style.flags & re_core::CellStyle::BOLD != 0);
    }

    #[test]
    fn flat_interior_wall_keeps_its_glyph_no_line() {
        // A fully-surrounded equal-depth wall cell has NO edge → keeps its body.
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                fb.set(c, r, solid(4.0, 0));
            }
        }
        apply(&mut fb, &ContourConfig::default());
        let mid = fb.get(1, 1).unwrap();
        assert_eq!(
            char::from_u32(mid.glyph).unwrap(),
            '.',
            "an interior flat wall cell must NOT get a contour glyph"
        );
    }

    #[test]
    fn strong_depth_jump_draws_a_line() {
        // Center near (depth 2), east neighbour far (depth 8) → east edge. The NEAR
        // cell (center) owns the depth lip → it draws the HEAVY vertical OUTER rule
        // '┃'. The FAR cell does NOT draw the same boundary (single-side: no doubling).
        let mut fb = Framebuffer::new(3, 1);
        fb.set(0, 0, solid(2.0, 0));
        fb.set(1, 0, solid(2.0, 0));
        fb.set(2, 0, solid(8.0, 0));
        apply(&mut fb, &ContourConfig::default());
        let mid = fb.get(1, 0).unwrap();
        assert_eq!(char::from_u32(mid.glyph).unwrap(), '┃');
        // The far cell (col 2) must NOT also draw the boundary — draw-once convention.
        let far = fb.get(2, 0).unwrap();
        assert_ne!(
            char::from_u32(far.glyph).unwrap(),
            '┃',
            "the far side must not double-draw the depth boundary"
        );
    }

    #[test]
    fn wall_base_contact_edge_survives_the_dark_strip_depth_clamp() {
        // Regression for the dark-strip fix's contact sheet.
        // `clamp_to_wall_depth` (`lib.rs`) bounds a
        // floor row's classic-formula depth to a nearer wall's own `perp`, so a
        // wall-base row and the floor row right below it can now read as the EXACT
        // SAME depth — this test pins that a wall→floor boundary still draws its
        // contour line even at depth EQUALITY (not just a small gap), because the
        // signal is the SURFACE-KIND change, not a depth jump the clamp intentionally
        // removed.
        let mut fb = Framebuffer::new(1, 2);
        let wall = solid(13.5, 0); // WallSolid, matches the `06.lvl` col 53 example
        let mut floor = solid(13.5, 0); // SAME depth as the wall — the clamped case
        floor.surface = SurfaceKind::Floor;
        fb.set(0, 0, wall);
        fb.set(0, 1, floor);
        apply(&mut fb, &ContourConfig::default());

        // North/South boundary at equal depth: `owns_boundary` breaks the tie toward
        // the upper cell (row 0, the wall) for a South-ward seam, so row 0 must carry
        // the OUTER horizontal contact line.
        let top = fb.get(0, 0).unwrap();
        assert_eq!(
            char::from_u32(top.glyph).unwrap(),
            '━',
            "the wall's own base row must draw the OUTER wall-base contact line \
             even when the floor row directly below it has been clamped to the \
             SAME depth as the wall"
        );
        assert!(
            top.style.flags & re_core::CellStyle::BOLD != 0,
            "the wall-base contact line is a PRIMARY (outer) line, not a subtle inner seam"
        );

        // The floor row must NOT also draw the boundary — single-side convention.
        let bottom = fb.get(0, 1).unwrap();
        assert_ne!(
            char::from_u32(bottom.glyph).unwrap(),
            '━',
            "the floor side must not double-draw the wall-base boundary"
        );
    }

    #[test]
    fn gentle_depth_gradient_is_not_an_edge() {
        // A flat floor receding: each neighbour only ~5% deeper → below threshold.
        let mut fb = Framebuffer::new(4, 1);
        for (c, d) in [4.0, 4.2, 4.4, 4.6].into_iter().enumerate() {
            let mut cell = solid(d, 0);
            cell.surface = SurfaceKind::Floor;
            fb.set(c, 0, cell);
        }
        apply(&mut fb, &ContourConfig::default());
        for c in 1..3 {
            let cell = fb.get(c, 0).unwrap();
            assert_eq!(
                char::from_u32(cell.glyph).unwrap(),
                '.',
                "gentle gradient cell {c} must not draw a contour"
            );
        }
    }

    #[test]
    fn color_volume_brightens_ns_face_relative_to_ew() {
        // Two interior (no-edge) walls differing only in `side`: NS must end up
        // brighter than EW after the volume shift.
        let mut fb = Framebuffer::new(3, 3);
        // surround with equal-depth walls so the centre has no edge
        for r in 0..3 {
            for c in 0..3 {
                fb.set(c, r, solid(4.0, 0));
            }
        }
        fb.set(1, 1, solid(4.0, 0)); // NS
        apply(&mut fb, &ContourConfig::default());
        let ns_fg = unpack_rgb(fb.get(1, 1).unwrap().fg);

        let mut fb2 = Framebuffer::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                fb2.set(c, r, solid(4.0, 1));
            }
        }
        apply(&mut fb2, &ContourConfig::default());
        let ew_fg = unpack_rgb(fb2.get(1, 1).unwrap().fg);

        assert!(
            ns_fg.length() > ew_fg.length(),
            "NS face ({ns_fg:?}) must read brighter than EW ({ew_fg:?})"
        );
    }

    #[test]
    fn color_volume_keeps_hue_brightness_only() {
        // The volume shift must be a luminance scale, not a hue rotation: the
        // ratio between channels is preserved.
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                fb.set(c, r, solid(2.0, 0));
            }
        }
        let orig = unpack_rgb(solid(2.0, 0).fg);
        apply(&mut fb, &ContourConfig::default());
        let got = unpack_rgb(fb.get(1, 1).unwrap().fg);
        // r:g ratio preserved within rounding.
        let ro = orig.x / orig.y;
        let rg = got.x / got.y;
        assert!((ro - rg).abs() < 0.02, "hue drifted: {ro} vs {rg}");
    }

    /// OFF is the default, and OFF must be byte-identical to no crown at all — not
    /// "close". Every mask/surface combination falls through to the orientation glyph.
    #[test]
    fn crown_off_is_byte_identical_to_the_orientation_glyph() {
        use re_core::SurfaceKind as S;
        let cfg = ContourConfig::default();
        assert_eq!(
            cfg.crown_style,
            CrownStyle::Off,
            "OFF is the shipped default"
        );
        for surface in [
            S::WallSolid,
            S::WallTrace,
            S::WallMatrix,
            S::WallOther,
            S::Floor,
        ] {
            for mask in 1u8..16 {
                assert_eq!(
                    crown_or_line(&cfg, surface, mask),
                    contour_glyph_weighted(mask, EdgeClass::Outer),
                    "crown OFF must not alter mask {mask:#06b} on {surface:?}"
                );
            }
        }
    }

    /// ON crowns a FLAT TOP by surface kind, and nothing else. The three guards —
    /// open-above, no vertical edge, theme offers a glyph — are each checked, because
    /// each one failing is a different way to wreck a silhouette.
    #[test]
    fn crown_on_replaces_only_the_flat_top_stroke() {
        use re_core::SurfaceKind as S;
        let cfg = ContourConfig {
            crown_style: CrownStyle::On,
            ..ContourConfig::default()
        };

        // A flat top (open above, no vertical edge) IS crowned, per surface.
        assert_eq!(crown_or_line(&cfg, S::WallTrace, N), '~');
        assert_eq!(crown_or_line(&cfg, S::WallMatrix, N), '*');
        assert_eq!(crown_or_line(&cfg, S::WallOther, N), '^');

        // A plain wall declares no crown → keeps the heavy orientation stroke.
        assert_eq!(crown_or_line(&cfg, S::WallSolid, N), '━');

        // A CORNER keeps its connector: crowning it would break the box.
        assert_eq!(
            crown_or_line(&cfg, S::WallTrace, N | E),
            contour_glyph_weighted(N | E, EdgeClass::Outer),
            "a corner must keep its connector, not become a crown"
        );

        // An UNDERSIDE (open below) is not a top edge.
        assert_eq!(
            crown_or_line(&cfg, S::WallTrace, S),
            contour_glyph_weighted(S, EdgeClass::Outer),
            "an open-BELOW edge is an underside, never a crown"
        );

        // A pure side edge is not a top edge either.
        assert_eq!(
            crown_or_line(&cfg, S::WallTrace, W),
            contour_glyph_weighted(W, EdgeClass::Outer)
        );

        // Floor/Ceiling/Empty are not silhouette tops.
        assert_eq!(
            crown_or_line(&cfg, S::Floor, N),
            contour_glyph_weighted(N, EdgeClass::Outer)
        );
    }

    /// The env knob a manual tuning pass drives. Parsed here (not in the caller)
    /// so the spelling is testable without a process env round-trip in the hot path.
    #[test]
    fn crown_theme_maps_every_wall_surface_deliberately() {
        use re_core::SurfaceKind as S;
        let t = CrownTheme::PROPOSAL;
        // Declared, on purpose: a plain wall keeps its stroke.
        assert_eq!(t.crown_for(S::WallSolid), None);
        assert_eq!(t.crown_for(S::WallTrace), Some('~'));
        assert_eq!(t.crown_for(S::WallMatrix), Some('*'));
        assert_eq!(t.crown_for(S::WallOther), Some('^'));
        // Non-silhouette surfaces never crown, whatever the theme says.
        assert_eq!(t.crown_for(S::Floor), None);
        assert_eq!(t.crown_for(S::Ceiling), None);
        assert_eq!(t.crown_for(S::Empty), None);
    }

    #[test]
    fn contour_glyph_orientations() {
        assert_eq!(contour_glyph(N), '─');
        assert_eq!(contour_glyph(S), '─');
        assert_eq!(contour_glyph(E), '│');
        assert_eq!(contour_glyph(W), '│');
        // Corners connect toward the SOLID (opposite of each open edge): a cell open
        // up+right (N|E) is the TOP-RIGHT of a solid mass that lies down-left, so it
        // draws ┐; open down+left (S|W) is the BOTTOM-LEFT of a mass up-right → └.
        assert_eq!(contour_glyph(N | E), '┐');
        assert_eq!(contour_glyph(S | W), '└');
        assert_eq!(contour_glyph(N | E | W), '┴');
        assert_eq!(contour_glyph(N | E | S | W), '┼');
        assert_eq!(contour_glyph(0), ' ');
    }

    #[test]
    fn box_corners_form_an_upright_rectangle_not_an_inverted_one() {
        // REGRESSION: the auto-contour corner direction was inverted on the vertical
        // axis, so a solid box silhouette rendered as a vertically-FLIPPED rectangle:
        // the TOP corners pointed down and the BOTTOM corners pointed up (the
        // "stairs bottom line corners point the wrong way" bug). All four corners,
        // light and heavy, must hug the SOLID interior so a box reads upright.
        //
        // Light set (inner / generic) — a proper rectangle is ┌┐ over └┘:
        assert_eq!(
            contour_glyph_weighted(N | W, EdgeClass::Inner),
            '┌',
            "top-left"
        );
        assert_eq!(
            contour_glyph_weighted(N | E, EdgeClass::Inner),
            '┐',
            "top-right"
        );
        assert_eq!(
            contour_glyph_weighted(S | W, EdgeClass::Inner),
            '└',
            "BOTTOM-left must connect UPWARD into the shape (└, not ┌)"
        );
        assert_eq!(
            contour_glyph_weighted(S | E, EdgeClass::Inner),
            '┘',
            "BOTTOM-right must connect UPWARD into the shape (┘, not ┐)"
        );
        // Heavy set (outer silhouette — what stairs actually draw) — ┏┓ over ┗┛:
        assert_eq!(
            contour_glyph_weighted(N | W, EdgeClass::Outer),
            '┏',
            "top-left heavy"
        );
        assert_eq!(
            contour_glyph_weighted(N | E, EdgeClass::Outer),
            '┓',
            "top-right heavy"
        );
        assert_eq!(
            contour_glyph_weighted(S | W, EdgeClass::Outer),
            '┗',
            "BOTTOM-left heavy must connect UPWARD (┗, not ┏)"
        );
        assert_eq!(
            contour_glyph_weighted(S | E, EdgeClass::Outer),
            '┛',
            "BOTTOM-right heavy must connect UPWARD (┛, not ┓)"
        );
    }

    #[test]
    fn bottom_edge_corners_of_a_solid_box_point_up() {
        // End-to-end through `apply`: a 3×3 solid box on sky. With slopes OFF (so the
        // box corners reach the box-drawing table rather than being chamfered), the
        // BOTTOM-LEFT cell must draw ┗ (stroke UP+right, into the box) and the
        // BOTTOM-RIGHT cell ┛ (stroke UP+left). Pre-fix these were ┏ / ┓ — pointing
        // DOWN, away from the shape (the reported inversion).
        let mut fb = Framebuffer::new(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                let inside = (1..=3).contains(&c) && (1..=3).contains(&r);
                fb.set(c, r, if inside { solid(4.0, 0) } else { sky() });
            }
        }
        let cfg = ContourConfig {
            slope_style: SlopeStyle::Off,
            ..ContourConfig::default()
        };
        apply(&mut fb, &cfg);
        let bl = char::from_u32(fb.get(1, 3).unwrap().glyph).unwrap();
        let br = char::from_u32(fb.get(3, 3).unwrap().glyph).unwrap();
        let tl = char::from_u32(fb.get(1, 1).unwrap().glyph).unwrap();
        let tr = char::from_u32(fb.get(3, 1).unwrap().glyph).unwrap();
        assert_eq!(bl, '┗', "bottom-left corner must point UP into the box");
        assert_eq!(br, '┛', "bottom-right corner must point UP into the box");
        // And the top corners stay correct (must not regress the top line).
        assert_eq!(
            tl, '┏',
            "top-left corner stays ┏ (points down into the box)"
        );
        assert_eq!(
            tr, '┓',
            "top-right corner stays ┓ (points down into the box)"
        );
    }

    /// A solid wall cell with a contrasting fg, for inner-seam tests.
    fn solid_fg(depth: f32, fg: u32, surface: SurfaceKind) -> Cell {
        Cell {
            glyph: u32::from('.'),
            fg,
            bg: 0x0010_0804,
            depth,
            surface,
            ..Cell::default()
        }
    }

    #[test]
    fn inner_material_seam_is_secondary_and_drawn_once() {
        // Two equal-depth solid walls of DIFFERENT surface kind AND clearly different
        // fg luma, side by side. The E/W seam between them is an INNER material line.
        let bright = 0x00D0_D0D0; // luma ≈ 0.82
        let dark = 0x0020_2020; // luma ≈ 0.13  → diff ≈ 0.69 >> threshold
        let mut fb = Framebuffer::new(2, 1);
        fb.set(0, 0, solid_fg(4.0, bright, SurfaceKind::WallSolid));
        fb.set(1, 0, solid_fg(4.0, dark, SurfaceKind::WallTrace));

        // Silhouette-only mode: interior material change draws NOTHING.
        let mut sil = fb.clone();
        apply(
            &mut sil,
            &ContourConfig {
                mode: ContourMode::Silhouette,
                ..ContourConfig::default()
            },
        );
        assert_eq!(char::from_u32(sil.get(0, 0).unwrap().glyph).unwrap(), '.');
        assert_eq!(char::from_u32(sil.get(1, 0).unwrap().glyph).unwrap(), '.');

        // The shipping default no longer replaces the material glyph on an
        // inner seam — the LEFT/leading cell keeps its body glyph ('.') and
        // only gets a subtle fg/bg tint. The right cell is untouched
        // (single-side ownership still holds; the leading cell owns the seam).
        let mut def = fb.clone();
        apply(&mut def, &ContourConfig::default());
        let left_def = def.get(0, 0).unwrap();
        assert_eq!(
            char::from_u32(left_def.glyph).unwrap(),
            '.',
            "default inner seam keeps the material glyph (tint only, no box stroke)"
        );
        assert_ne!(
            left_def.fg, bright,
            "default inner seam still nudges the fg (a faint crease tint)"
        );
        // The trailing cell is untouched (no double-tint of the equal-depth seam).
        assert_eq!(char::from_u32(def.get(1, 0).unwrap().glyph).unwrap(), '.');

        // Opt-in (inner_replace_glyph): the LEFT/leading cell owns the equal-depth
        // E/W seam and draws a LIGHT vertical INNER rule '│' with the DIM attribute;
        // the right cell does NOT double-draw it.
        let mut mat = fb.clone();
        apply(
            &mut mat,
            &ContourConfig {
                inner_replace_glyph: true,
                ..ContourConfig::default()
            },
        );
        let left = mat.get(0, 0).unwrap();
        assert_eq!(
            char::from_u32(left.glyph).unwrap(),
            '│',
            "inner seam draws the LIGHT (secondary) vertical rule on the leading cell"
        );
        assert!(
            left.style.flags & re_core::CellStyle::DIM != 0,
            "inner line carries the DIM attribute (secondary hierarchy)"
        );
        assert_eq!(
            char::from_u32(mat.get(1, 0).unwrap().glyph).unwrap(),
            '.',
            "the trailing cell must NOT double-draw the equal-depth seam"
        );
    }

    #[test]
    fn inner_line_disabled_when_inner_lines_off() {
        let bright = 0x00D0_D0D0;
        let dark = 0x0020_2020;
        let mut fb = Framebuffer::new(2, 1);
        fb.set(0, 0, solid_fg(4.0, bright, SurfaceKind::WallSolid));
        fb.set(1, 0, solid_fg(4.0, dark, SurfaceKind::WallTrace));
        apply(
            &mut fb,
            &ContourConfig {
                inner_lines: false,
                ..ContourConfig::default()
            },
        );
        assert_eq!(char::from_u32(fb.get(0, 0).unwrap().glyph).unwrap(), '.');
    }

    #[test]
    fn trivial_shade_difference_does_not_draw_inner_line() {
        // Same depth, different surface KIND but near-identical fg luma → below the
        // inner luma threshold → NO inner line (interior stays minimalist).
        let a = 0x0080_8080;
        let b = 0x0084_8484; // luma diff ≈ 0.016 << 0.16 threshold
        let mut fb = Framebuffer::new(2, 1);
        fb.set(0, 0, solid_fg(4.0, a, SurfaceKind::WallSolid));
        fb.set(1, 0, solid_fg(4.0, b, SurfaceKind::WallTrace));
        apply(&mut fb, &ContourConfig::default());
        assert_eq!(
            char::from_u32(fb.get(0, 0).unwrap().glyph).unwrap(),
            '.',
            "a trivial shade difference must not spam an inner line"
        );
    }

    #[test]
    fn outer_dominates_coincident_inner_seam() {
        // A wall cell with sky to the west (OUTER silhouette) AND a material seam to
        // the east (INNER). The OUTER silhouette must win → BOLD heavy glyph, not the
        // dim light inner rule.
        let mut fb = Framebuffer::new(3, 1);
        fb.set(0, 0, sky());
        fb.set(1, 0, solid_fg(4.0, 0x00D0_D0D0, SurfaceKind::WallSolid));
        fb.set(2, 0, solid_fg(4.0, 0x0020_2020, SurfaceKind::WallTrace));
        apply(&mut fb, &ContourConfig::default());
        let mid = fb.get(1, 0).unwrap();
        // West silhouette + east material seam on the same cell → OUTER wins.
        assert!(
            mid.style.flags & re_core::CellStyle::BOLD != 0,
            "a cell on the silhouette must render the BOLD outer line even with a coincident inner seam"
        );
    }

    #[test]
    fn peaked_roof_renders_slope_glyphs_not_stairstep_corners() {
        // Build a FILLED triangular mass (a peaked roof) of equal-depth solid cells
        // against sky, and assert the diagonal flanks emit SLOPE glyphs (triangles)
        // instead of the stair-step box corners the 4-neighbour pass would place.
        //
        // Layout (7 cols × 4 rows), '#': solid, '.': sky. Peak at col 3.
        //   row0:    . . . # . . .
        //   row1:    . . # # # . .
        //   row2:    . # # # # # .
        //   row3:    # # # # # # #
        let cols = 7usize;
        let rows = 4usize;
        let peak = 3i32;
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                let half = r as i32; // solid span = peak ± r
                let solid_here = ((c as i32) - peak).abs() <= half;
                fb.set(c, r, if solid_here { solid(4.0, 0) } else { sky() });
            }
        }
        apply(&mut fb, &ContourConfig::default());

        // The LEFT flank descends down-and-LEFT: each leftmost solid cell is open to
        // the W/NW with solid mass to its SE (an UpLeft slope → ◢). The RIGHT flank
        // descends down-and-RIGHT: open E/NE, solid mass SW (an UpRight slope → ◣).
        // Collect every glyph that landed on the two flank cells.
        let tri_up_left = '◢'; // SlopeDir::UpLeft  (solid SE)
        let tri_up_right = '◣'; // SlopeDir::UpRight (solid SW)
        let stair = ['┐', '┌', '┘', '└', '╮', '╭'];

        let mut left_glyphs = Vec::new();
        let mut right_glyphs = Vec::new();
        for r in 1..rows {
            // leftmost / rightmost solid col in this row
            let lc = (peak - r as i32) as usize;
            let rc = (peak + r as i32) as usize;
            left_glyphs.push(char::from_u32(fb.get(lc, r).unwrap().glyph).unwrap());
            right_glyphs.push(char::from_u32(fb.get(rc, r).unwrap().glyph).unwrap());
        }

        // At least one flank cell on each side must be a slope triangle, and NO flank
        // cell may keep a stair-step box corner.
        assert!(
            left_glyphs.contains(&tri_up_left),
            "left flank must emit the ◢ slope triangle, got {left_glyphs:?}"
        );
        assert!(
            right_glyphs.contains(&tri_up_right),
            "right flank must emit the ◣ slope triangle, got {right_glyphs:?}"
        );
        for g in left_glyphs.iter().chain(right_glyphs.iter()) {
            assert!(
                !stair.contains(g),
                "no flank cell may keep a stair-step box corner, found {g:?}"
            );
        }
    }

    #[test]
    fn slope_off_keeps_stairstep_corners() {
        // The SAME roof with slope_style = Off must keep box corners (no regression
        // of the original behaviour for callers that disable de-angularization).
        let cols = 5usize;
        let rows = 3usize;
        let peak = 2i32;
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                let solid_here = ((c as i32) - peak).abs() <= r as i32;
                fb.set(c, r, if solid_here { solid(4.0, 0) } else { sky() });
            }
        }
        apply(
            &mut fb,
            &ContourConfig {
                slope_style: SlopeStyle::Off,
                ..ContourConfig::default()
            },
        );
        // The apex cell (peak, row 0) borders sky on N, E, W → a ┴ junction; the
        // left flank at row 1 col 1 is a box corner, NOT a triangle.
        let flank = char::from_u32(fb.get(1, 1).unwrap().glyph).unwrap();
        assert!(
            !matches!(flank, '◢' | '◣' | '◤' | '◥' | '╱' | '╲'),
            "slope OFF must keep box-drawing corners, got {flank:?}"
        );
    }

    #[test]
    fn axis_aligned_wall_unaffected_by_slope_pass() {
        // A plain vertical wall edge against sky (no diagonal run) must still draw the
        // straight heavy vertical rule — slopes only trigger on real diagonal runs.
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            fb.set(0, r, sky());
            fb.set(1, r, solid(4.0, 0));
            fb.set(2, r, solid(4.0, 0));
        }
        apply(&mut fb, &ContourConfig::default());
        let mid = fb.get(1, 1).unwrap();
        assert_eq!(
            char::from_u32(mid.glyph).unwrap(),
            '┃',
            "a straight vertical silhouette must NOT become a slope glyph"
        );
    }

    /// Set of every glyph the SUB-45° half-step pass can emit (TOP-edge variants).
    const HALF_STEP_GLYPHS: [char; 3] = ['▚', '▞', '▄'];

    #[test]
    fn shallow_2to1_stair_emits_half_step_glyphs() {
        // A FILLED mass whose TOP edge descends one row every TWO columns (a 2:1
        // roofline) against sky above. The shallow detector must fire and the contour
        // pass must place at least one half-step glyph on the run's top edge.
        //   col:   0 1 2 3 4 5 6 7
        //   row0:  . . . . . . . .
        //   row1:  # # . . . . . .
        //   row2:  # # # # . . . .
        //   row3:  # # # # # # . .
        //   row4:  # # # # # # # #
        //   row5:  # # # # # # # #
        let cols = 8usize;
        let rows = 6usize;
        let top_of = |c: usize| -> usize { 1 + c / 2 }; // 2:1 descending-right
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                let solid_here = r >= top_of(c);
                fb.set(c, r, if solid_here { solid(4.0, 0) } else { sky() });
            }
        }
        apply(&mut fb, &ContourConfig::default());

        let mut half_steps = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap();
                if HALF_STEP_GLYPHS.contains(&g) {
                    half_steps.push((c, r, g));
                }
            }
        }
        assert!(
            !half_steps.is_empty(),
            "a 2:1 shallow roofline must emit at least one sub-45 half-step glyph, found none"
        );
    }

    #[test]
    fn flat_wall_top_emits_no_half_step() {
        // The CRITICAL negative: a perfectly FLAT wall top against sky must NOT be
        // shattered into half-step quadrants. Every top-edge cell stays the straight
        // heavy rule '━' (or a corner at the ends), never a '▚'/'▞'/'▄'.
        let cols = 8usize;
        let rows = 4usize;
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                // Sky on the very top row; solid below → a flat top edge at row 1.
                fb.set(c, r, if r == 0 { sky() } else { solid(4.0, 0) });
            }
        }
        apply(&mut fb, &ContourConfig::default());
        for c in 0..cols {
            for r in 0..rows {
                let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap();
                assert!(
                    !HALF_STEP_GLYPHS.contains(&g),
                    "flat wall top must never emit a half-step glyph, found {g:?} at ({c},{r})"
                );
            }
        }
        // And the flat top edge itself is still the straight horizontal rule.
        let top_mid = char::from_u32(fb.get(4, 1).unwrap().glyph).unwrap();
        assert_eq!(
            top_mid, '━',
            "a flat wall top must stay the straight heavy rule"
        );
    }

    #[test]
    fn forty_five_degree_roof_still_triangles_not_half_steps() {
        // A 45° (1:1) peaked roof must keep its chamfer TRIANGLES — the 1:1 path owns
        // it, and the 2:1 half-step must not steal those cells.
        let cols = 7usize;
        let rows = 4usize;
        let peak = 3i32;
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                let solid_here = ((c as i32) - peak).abs() <= r as i32;
                fb.set(c, r, if solid_here { solid(4.0, 0) } else { sky() });
            }
        }
        apply(&mut fb, &ContourConfig::default());
        // The flanks carry the 45° triangles ◢/◣; no flank cell may be a half-step.
        let mut saw_triangle = false;
        for r in 1..rows {
            let lc = (peak - r as i32) as usize;
            let rc = (peak + r as i32) as usize;
            for c in [lc, rc] {
                let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap();
                if matches!(g, '◢' | '◣' | '◤' | '◥') {
                    saw_triangle = true;
                }
                assert!(
                    !HALF_STEP_GLYPHS.contains(&g),
                    "a 45° flank cell must stay a triangle, not a 2:1 half-step (got {g:?})"
                );
            }
        }
        assert!(
            saw_triangle,
            "the 45° roof must still emit chamfer triangles"
        );
    }

    #[test]
    fn ascending_vs_descending_stair_line_glyphs_are_distinct_and_correct() {
        // BUG #11 REGRESSION. A 1-up-1-over (45°) staircase silhouette against sky,
        // rendered with SlopeStyle::Line. A run RISING to the right (the top edge climbs
        // as the column increases) must lean `╱`; a run FALLING to the right must lean
        // `╲`. Pre-fix BOTH leaned the same wrong way (`╲`/`╱` swapped), so an ascending
        // stair read as descending — exactly the playtester's "up reads as down".
        //
        // We build the two silhouettes as thin one-cell outlines so the Line path runs.

        // top_of(c): the silhouette top-edge row for column c. RISING-RIGHT means the
        // top climbs (row decreases) as c grows.
        fn build(cols: usize, rows: usize, top_of: impl Fn(usize) -> usize) -> Framebuffer {
            let mut fb = Framebuffer::new(cols, rows);
            for r in 0..rows {
                for c in 0..cols {
                    let solid_here = r >= top_of(c);
                    fb.set(c, r, if solid_here { solid(4.0, 0) } else { sky() });
                }
            }
            fb
        }
        let line_cfg = ContourConfig {
            slope_style: SlopeStyle::Line,
            ..ContourConfig::default()
        };

        // Ascending to the right: top row DECREASES as col increases (1-up-1-over).
        //   col:  0 1 2 3 4 5
        //   top:  5 4 3 2 1 0
        let cols = 6usize;
        let rows = 7usize;
        let mut asc = build(cols, rows, |c| (cols - 1).saturating_sub(c));
        apply(&mut asc, &line_cfg);

        // Descending to the right: top row INCREASES as col increases.
        //   top:  0 1 2 3 4 5
        let mut desc = build(cols, rows, |c| c.min(rows - 1));
        apply(&mut desc, &line_cfg);

        // Collect the diagonal stroke glyphs that landed on each silhouette top edge.
        let collect = |fb: &Framebuffer| -> Vec<char> {
            let mut v = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap();
                    if matches!(g, '╱' | '╲') {
                        v.push(g);
                    }
                }
            }
            v
        };
        let asc_glyphs = collect(&asc);
        let desc_glyphs = collect(&desc);

        // ASCENDING (rising-right) must show `╱` strokes and NO `╲`.
        assert!(
            asc_glyphs.contains(&'╱'),
            "an ascending (rising-right) stair must lean ╱, got {asc_glyphs:?}"
        );
        assert!(
            !asc_glyphs.contains(&'╲'),
            "an ascending stair must NOT contain the descending ╲ stroke, got {asc_glyphs:?}"
        );
        // DESCENDING (falling-right) must show `╲` strokes and NO `╱`.
        assert!(
            desc_glyphs.contains(&'╲'),
            "a descending (falling-right) stair must lean ╲, got {desc_glyphs:?}"
        );
        assert!(
            !desc_glyphs.contains(&'╱'),
            "a descending stair must NOT contain the ascending ╱ stroke, got {desc_glyphs:?}"
        );
        // And the two directions must be DISTINGUISHABLE (not identical output).
        assert_ne!(
            asc_glyphs, desc_glyphs,
            "ascending and descending stairs must produce visibly different slope glyphs"
        );
    }

    #[test]
    fn contour_preserves_pipe_glass_underline_on_outer_line() {
        // GLASS-FIX: a powered-pipe text cell carries a colored single underline (the
        // "glass tube" meniscus). When that cell ALSO sits on a silhouette (sky to the
        // west), the OUTER contour line must keep BOTH the BOLD contour attribute AND
        // the cell's underline flag + underline colour — the glass edge survives onto
        // the pipe's outline, it is not stripped.
        let glass_ul = 0x0011_AAFF_u32;
        let mut wall = solid(4.0, 0);
        wall.style = re_core::CellStyle::NONE.with_underline_color(glass_ul);
        assert!(
            wall.style.is_underline(),
            "precondition: source has underline"
        );

        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            fb.set(0, r, sky());
            fb.set(1, r, wall);
            fb.set(2, r, solid(4.0, 0));
        }
        apply(&mut fb, &ContourConfig::default());

        let mid = fb.get(1, 1).unwrap();
        // OUTER silhouette still draws the bold heavy vertical rule.
        assert_eq!(char::from_u32(mid.glyph).unwrap(), '┃');
        assert!(
            mid.style.flags & re_core::CellStyle::BOLD != 0,
            "outer contour must still set BOLD"
        );
        // ...and the glass underline survived: flag + colour both carried through.
        assert!(
            mid.style.is_underline(),
            "the pipe-glass UNDERLINE flag must survive the contour restyle"
        );
        assert_eq!(
            mid.style.underline_color(),
            Some(glass_ul),
            "the glass underline COLOUR must survive the contour restyle"
        );
    }

    #[test]
    fn contour_outer_line_without_underline_is_plain_bold() {
        // The negative / no-regression case: a silhouette cell whose source had NO
        // underline gets exactly the old style — BOLD over NONE, no underline. This is
        // what guarantees every non-pipe cell (and every golden) is byte-identical.
        let mut fb = Framebuffer::new(3, 3);
        for r in 0..3 {
            fb.set(0, r, sky());
            fb.set(1, r, solid(4.0, 0));
            fb.set(2, r, solid(4.0, 0));
        }
        apply(&mut fb, &ContourConfig::default());

        let mid = fb.get(1, 1).unwrap();
        assert_eq!(
            mid.style,
            re_core::CellStyle {
                flags: re_core::CellStyle::BOLD,
                ..re_core::CellStyle::NONE
            },
            "an outer line on a non-underlined cell must be plain BOLD (no behaviour change)"
        );
        assert!(
            !mid.style.is_underline(),
            "a non-underlined source must NOT gain an underline"
        );
    }

    #[test]
    fn contour_inner_box_stroke_preserves_underline() {
        // The inner_replace_glyph branch also rebuilds the style (DIM box stroke); it
        // must carry the glass underline through too. Two equal-depth solid walls of
        // different surface kind + strong luma diff → the leading cell owns an inner
        // seam; give that leading cell a glass underline and assert it survives.
        let glass_ul = 0x0022_CCEE_u32;
        let bright = 0x00D0_D0D0;
        let dark = 0x0020_2020;
        let mut left = solid_fg(4.0, bright, SurfaceKind::WallSolid);
        left.style = re_core::CellStyle::NONE.with_underline_color(glass_ul);
        let mut fb = Framebuffer::new(2, 1);
        fb.set(0, 0, left);
        fb.set(1, 0, solid_fg(4.0, dark, SurfaceKind::WallTrace));
        apply(
            &mut fb,
            &ContourConfig {
                inner_replace_glyph: true,
                ..ContourConfig::default()
            },
        );
        let owner = fb.get(0, 0).unwrap();
        assert_eq!(
            char::from_u32(owner.glyph).unwrap(),
            '│',
            "the inner seam still draws the light vertical rule"
        );
        assert!(
            owner.style.flags & re_core::CellStyle::DIM != 0,
            "inner line stays DIM"
        );
        assert!(
            owner.style.is_underline(),
            "glass underline survives the inner box stroke"
        );
        assert_eq!(owner.style.underline_color(), Some(glass_ul));
    }

    #[test]
    fn no_doubled_line_at_wall_base_transition() {
        // The reported artifact: a near wall (small depth) sitting ABOVE a far floor
        // (larger depth) — the wall-base. The boundary must be drawn by EXACTLY ONE
        // of the two rows, never both, so no two stacked horizontal lines appear.
        let mut fb = Framebuffer::new(1, 2);
        let wall = solid(2.0, 0); // near, row 0
        let mut floor = solid(9.0, 0); // far, row 1
        floor.surface = SurfaceKind::Floor;
        fb.set(0, 0, wall);
        fb.set(0, 1, floor);
        apply(&mut fb, &ContourConfig::default());

        let top = char::from_u32(fb.get(0, 0).unwrap().glyph).unwrap();
        let bot = char::from_u32(fb.get(0, 1).unwrap().glyph).unwrap();
        let is_h = |g: char| matches!(g, '─' | '━');
        // The near (wall) row owns the lip; the far (floor) row must stay clean.
        assert!(
            is_h(top),
            "near wall row should draw the base line, got {top:?}"
        );
        assert!(
            !is_h(bot),
            "far floor row must NOT also draw the base line (no doubled line), got {bot:?}"
        );
    }
}
