//! [`PresentCache`] + [`PresentConfig`] — the present layer's **temporal memory**.
//!
//! Two jobs, both pure (no terminal, fully unit-tested):
//!
//! 1. **Idle-frame skip.** `present_sys` used to run every frame unconditionally:
//!    hide the cursor (twice), open a DEC 2026 synchronized-update window, re-encode
//!    every framebuffer cell into ratatui's buffer (two colour conversions per cell),
//!    let ratatui diff it, write four SGR resets, close the window — ≈55 bytes and
//!    ~6 `flush` syscalls even when *nothing on screen changed*. A puzzle player stands
//!    still for seconds at a time, so the common frame is the idle frame. The cache
//!    keeps the last **target** `{glyph, fg, bg, style}` grid it was handed and lets
//!    `present_sys` return before its first terminal write when the new framebuffer
//!    is identical and the terminal size is unchanged. Idle frame = 0 bytes.
//!
//! 2. **Colour deadband.** Cell colours in `re_render` read raw distance by contract
//!    (the shade must be continuous in depth), so a one-cell step or a one-bucket yaw
//!    change moves fg/bg by a few units on nearly every cell while the *glyphs* hold.
//!    Every such cell costs a `SetColors` + glyph re-emit — a near-full-screen write
//!    that no human can see. In `Ansi256` mode it is worse: the nearest-cube map has
//!    no memory, so a colour hovering on a cube midpoint flips its palette index frame
//!    to frame ("sparkle"). The cache remembers the colour it last **emitted** per
//!    cell and keeps emitting it while the new *target* stays within the deadband.
//!
//! ## The drift bound (why this cannot smear)
//!
//! The comparison is always **target vs. last emitted**, never target vs. the
//! previous target. A slow gradient therefore cannot creep away unnoticed: the moment
//! the true colour is more than `deadband` from what is on screen, the cell snaps to
//! the true colour. The on-screen error is bounded by `deadband` per channel at all
//! times, and a deadband of `0` disables the hold entirely (bit-exact legacy output).
//!
//! Glyph and style changes are **never** suppressed — only colour is held, and only
//! when the glyph would otherwise carry an imperceptible recolour.
//!
//! ## Ansi256 hold rule
//!
//! For the 256-colour path the "previous" colour is the palette entry last emitted.
//! We keep it when
//! `dist²(target, prev_palette_rgb) ≤ dist²(target, nearest_palette_rgb) + 3·deadband²`
//! — i.e. the previous entry is *almost as good* as the mathematically nearest one.
//! Because the cube's step is 40 units, a deadband of 6 only ever holds across a
//! genuine midpoint straddle; it can never keep a colour that is a whole cube step
//! away from the truth.
//!
//! ## What forces a present
//!
//! * first frame after startup,
//! * framebuffer `cols × rows` changed (a resize re-primes the cache),
//! * the terminal's reported size changed (ratatui's `autoresize` will clear its own
//!   buffers on the next `draw`, so we must draw or the screen stays stale),
//! * any cell whose `{glyph, fg, bg, style}` target differs from the last target.
//!
//! HUD / overlay painters write into the same `Framebuffer` before `PresentSet`, so
//! they are covered by the cell comparison — no extra bookkeeping.

use bevy::ecs::resource::Resource;
use re_core::{Cell, CellStyle, Framebuffer};

use crate::color::{ansi256_to_rgb, rgb_to_ansi256, ColorMode};

/// Environment variable that overrides the colour deadband (`0` disables it).
pub const DEADBAND_ENV: &str = "TS_PRESENT_DEADBAND";

/// Startup knobs for the present layer's temporal memory.
///
/// Seeded ONCE at plugin build from [`DEADBAND_ENV`]; never read from the
/// environment in the per-frame path.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentConfig {
    /// Per-channel tolerance (0..=255) under which a cell keeps the colour it last
    /// emitted. `0` = off (every colour delta is emitted, legacy behaviour).
    pub color_deadband: u8,
}

impl PresentConfig {
    /// Default per-channel deadband. Six units is below the visible threshold on
    /// every tested terminal and below the 40-unit step of the 6×6×6 ANSI cube.
    pub const DEFAULT_DEADBAND: u8 = 6;

    /// Build from the environment: `TS_PRESENT_DEADBAND=<0..=255>`; anything unset
    /// or unparsable keeps [`Self::DEFAULT_DEADBAND`].
    pub fn from_env() -> Self {
        let color_deadband = std::env::var(DEADBAND_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(Self::DEFAULT_DEADBAND);
        Self { color_deadband }
    }
}

impl Default for PresentConfig {
    fn default() -> Self {
        Self {
            color_deadband: Self::DEFAULT_DEADBAND,
        }
    }
}

/// The part of a framebuffer [`Cell`] the terminal can show. Depth, material and
/// geometry are render-side data and do not participate in the idle comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetCell {
    /// Packed glyph scalar (see [`crate::glyph_char`] for how it decodes).
    pub glyph: u32,
    /// Packed `0x00RRGGBB` foreground colour.
    pub fg: u32,
    /// Packed `0x00RRGGBB` background colour.
    pub bg: u32,
    /// Bold/italic/etc. style flags.
    pub style: CellStyle,
}

impl From<&Cell> for TargetCell {
    #[inline]
    fn from(c: &Cell) -> Self {
        Self {
            glyph: c.glyph,
            fg: c.fg,
            bg: c.bg,
            style: c.style,
        }
    }
}

/// Sentinel for "no colour has been emitted for this cell yet". Colours are
/// `0x00RRGGBB`, so a value with the top byte set can never be a real target.
const NO_COLOR: u32 = u32::MAX;

/// Per-cell memory of what was last handed to the present and what was last put
/// on the terminal. Lives as a Bevy resource beside the framebuffer.
#[derive(Resource, Debug, Default)]
pub struct PresentCache {
    cols: usize,
    rows: usize,
    /// Last *target* per cell (what `re_render` asked for) — the idle comparison.
    last_target: Vec<TargetCell>,
    /// Last *emitted* fg / bg per cell (what is on the terminal) — the deadband.
    last_fg: Vec<u32>,
    last_bg: Vec<u32>,
    /// Terminal `(cols, rows)` as reported at the last present.
    term_size: (u16, u16),
    /// `false` until the first present; a fresh cache always presents.
    primed: bool,
    /// Whether ratatui's own cursor-hidden flag has been asserted once.
    pub cursor_hidden: bool,
    /// Number of frames skipped as idle since the last present (diagnostics).
    pub idle_streak: u32,
}

impl PresentCache {
    /// Decide whether this frame must reach the terminal.
    ///
    /// Re-primes (and returns `true`) when the framebuffer or terminal size changed
    /// or on the very first call; otherwise `true` iff any cell's target differs from
    /// the previous target. Does NOT record the new targets — call
    /// [`Self::record_target`] from the blit loop so a present that was clamped by a
    /// smaller terminal area still remembers exactly what it drew.
    pub fn needs_present(&mut self, fb: &Framebuffer, term_size: (u16, u16)) -> bool {
        let (cols, rows) = (fb.cols(), fb.rows());
        if !self.primed || cols != self.cols || rows != self.rows || term_size != self.term_size {
            self.reset(cols, rows, term_size);
            return true;
        }
        let changed = fb
            .cells()
            .iter()
            .zip(self.last_target.iter())
            .any(|(c, t)| TargetCell::from(c) != *t);
        if changed {
            self.idle_streak = 0;
        } else {
            self.idle_streak = self.idle_streak.saturating_add(1);
        }
        changed
    }

    /// Forget everything: the next frame presents in full and every colour is
    /// emitted verbatim (no previous colour to hold).
    pub fn reset(&mut self, cols: usize, rows: usize, term_size: (u16, u16)) {
        self.cols = cols;
        self.rows = rows;
        self.term_size = term_size;
        let n = cols * rows;
        // A glyph that no framebuffer produces (control char 0 with a NO_COLOR fg)
        // so the first real frame compares as "changed" on every cell.
        let blank = TargetCell {
            glyph: 0,
            fg: NO_COLOR,
            bg: NO_COLOR,
            style: CellStyle::default(),
        };
        self.last_target.clear();
        self.last_target.resize(n, blank);
        self.last_fg.clear();
        self.last_fg.resize(n, NO_COLOR);
        self.last_bg.clear();
        self.last_bg.resize(n, NO_COLOR);
        self.primed = true;
        self.idle_streak = 0;
    }

    #[inline]
    fn index(&self, col: usize, row: usize) -> Option<usize> {
        (col < self.cols && row < self.rows).then(|| row * self.cols + col)
    }

    /// Remember the target the blit loop just consumed for `(col, row)`.
    #[inline]
    pub fn record_target(&mut self, col: usize, row: usize, cell: &Cell) {
        if let Some(i) = self.index(col, row) {
            self.last_target[i] = TargetCell::from(cell);
        }
    }

    /// The fg colour to put on the terminal for `(col, row)`, honouring the
    /// deadband against what this cell last showed. Records the emission.
    #[inline]
    pub fn resolve_fg(
        &mut self,
        col: usize,
        row: usize,
        target: u32,
        mode: ColorMode,
        deadband: u8,
    ) -> u32 {
        let Some(i) = self.index(col, row) else {
            return target;
        };
        let prev = (self.last_fg[i] != NO_COLOR).then_some(self.last_fg[i]);
        let out = deadband_color(target, prev, mode, deadband);
        self.last_fg[i] = out;
        out
    }

    /// The bg counterpart of [`Self::resolve_fg`].
    #[inline]
    pub fn resolve_bg(
        &mut self,
        col: usize,
        row: usize,
        target: u32,
        mode: ColorMode,
        deadband: u8,
    ) -> u32 {
        let Some(i) = self.index(col, row) else {
            return target;
        };
        let prev = (self.last_bg[i] != NO_COLOR).then_some(self.last_bg[i]);
        let out = deadband_color(target, prev, mode, deadband);
        self.last_bg[i] = out;
        out
    }

    /// Grid the cache is currently sized for.
    pub fn dims(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }
}

#[inline]
fn rgb8(c: u32) -> (u8, u8, u8) {
    (
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8) & 0xFF) as u8,
        (c & 0xFF) as u8,
    )
}

#[inline]
fn pack(r: u8, g: u8, b: u8) -> u32 {
    (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
}

#[inline]
fn dist2(a: (u8, u8, u8), b: (u8, u8, u8)) -> u32 {
    let dr = i32::from(a.0) - i32::from(b.0);
    let dg = i32::from(a.1) - i32::from(b.1);
    let db = i32::from(a.2) - i32::from(b.2);
    (dr * dr + dg * dg + db * db) as u32
}

/// The colour to EMIT for a cell whose true colour is `target`, given the colour
/// `prev` it last showed (`None` = never emitted) and the mode.
///
/// * `Truecolor`: `prev` is kept iff every channel is within `deadband` of `target`.
/// * `Ansi256`: the result is always an exact palette RGB (so the caller's
///   `rgb → index` map is the identity on it). `prev` is kept iff it is within
///   `3·deadband²` squared distance of being the nearest entry — see the module doc.
/// * `deadband == 0` or `prev == None` ⇒ the true nearest colour, no memory.
pub fn deadband_color(target: u32, prev: Option<u32>, mode: ColorMode, deadband: u8) -> u32 {
    let (tr, tg, tb) = rgb8(target);
    match mode {
        ColorMode::Truecolor => match prev {
            Some(p) if deadband > 0 => {
                let (pr, pg, pb) = rgb8(p);
                let d = i16::from(deadband);
                let within = |a: u8, b: u8| (i16::from(a) - i16::from(b)).abs() <= d;
                if within(tr, pr) && within(tg, pg) && within(tb, pb) {
                    p
                } else {
                    target
                }
            }
            _ => target,
        },
        ColorMode::Ansi256 => {
            let best_idx = rgb_to_ansi256(tr, tg, tb);
            let best_rgb = ansi256_to_rgb(best_idx);
            let best = pack(best_rgb.0, best_rgb.1, best_rgb.2);
            match prev {
                Some(p) if deadband > 0 && p != best => {
                    let pr = rgb8(p);
                    let margin = 3 * u32::from(deadband) * u32::from(deadband);
                    if dist2((tr, tg, tb), pr) <= dist2((tr, tg, tb), best_rgb) + margin {
                        p
                    } else {
                        best
                    }
                }
                _ => best,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fb(cols: usize, rows: usize, glyph: u32, fg: u32, bg: u32) -> Framebuffer {
        let mut f = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                f.set(
                    c,
                    r,
                    Cell {
                        glyph,
                        fg,
                        bg,
                        ..Cell::default()
                    },
                );
            }
        }
        f
    }

    fn record_all(cache: &mut PresentCache, f: &Framebuffer) {
        for r in 0..f.rows() {
            for c in 0..f.cols() {
                cache.record_target(c, r, f.get(c, r).unwrap());
            }
        }
    }

    #[test]
    fn first_frame_presents_identical_second_frame_is_skipped() {
        let mut cache = PresentCache::default();
        let f = fb(4, 2, '#' as u32, 0x80FF80, 0x000000);
        assert!(
            cache.needs_present(&f, (80, 24)),
            "first frame must present"
        );
        record_all(&mut cache, &f);
        assert!(
            !cache.needs_present(&f, (80, 24)),
            "identical frame is idle"
        );
        assert_eq!(cache.idle_streak, 1);
        assert!(!cache.needs_present(&f, (80, 24)));
        assert_eq!(cache.idle_streak, 2);
    }

    #[test]
    fn one_changed_cell_presents_and_resets_streak() {
        let mut cache = PresentCache::default();
        let f = fb(4, 2, '#' as u32, 0x80FF80, 0x000000);
        cache.needs_present(&f, (80, 24));
        record_all(&mut cache, &f);
        assert!(!cache.needs_present(&f, (80, 24)));
        let mut g = f.clone();
        let mut cell = *g.get(2, 1).unwrap();
        cell.glyph = '@' as u32;
        g.set(2, 1, cell);
        assert!(cache.needs_present(&g, (80, 24)));
        assert_eq!(cache.idle_streak, 0);
    }

    #[test]
    fn glyph_change_with_same_colour_is_a_present() {
        let mut cache = PresentCache::default();
        let f = fb(2, 1, 'a' as u32, 0xFFFFFF, 0);
        cache.needs_present(&f, (10, 10));
        record_all(&mut cache, &f);
        let g = fb(2, 1, 'b' as u32, 0xFFFFFF, 0);
        assert!(cache.needs_present(&g, (10, 10)));
    }

    #[test]
    fn style_change_alone_is_a_present() {
        let mut cache = PresentCache::default();
        let f = fb(2, 1, 'a' as u32, 0xFFFFFF, 0);
        cache.needs_present(&f, (10, 10));
        record_all(&mut cache, &f);
        let mut g = f.clone();
        let mut cell = *g.get(0, 0).unwrap();
        cell.style.flags = CellStyle::BOLD;
        g.set(0, 0, cell);
        assert!(cache.needs_present(&g, (10, 10)));
    }

    #[test]
    fn terminal_resize_forces_present_even_when_cells_match() {
        let mut cache = PresentCache::default();
        let f = fb(4, 2, '#' as u32, 0x80FF80, 0x000000);
        cache.needs_present(&f, (80, 24));
        record_all(&mut cache, &f);
        assert!(!cache.needs_present(&f, (80, 24)));
        assert!(cache.needs_present(&f, (100, 30)), "terminal grew → redraw");
        // After re-priming, the deadband memory is gone: colours emit verbatim.
        let out = cache.resolve_fg(0, 0, 0x80FF83, ColorMode::Truecolor, 6);
        assert_eq!(out, 0x80FF83);
    }

    #[test]
    fn framebuffer_resize_forces_present() {
        let mut cache = PresentCache::default();
        let f = fb(4, 2, '#' as u32, 0, 0);
        cache.needs_present(&f, (80, 24));
        record_all(&mut cache, &f);
        let g = fb(5, 2, '#' as u32, 0, 0);
        assert!(cache.needs_present(&g, (80, 24)));
        assert_eq!(cache.dims(), (5, 2));
    }

    #[test]
    fn truecolor_small_delta_holds_large_delta_emits() {
        let mut cache = PresentCache::default();
        cache.reset(1, 1, (1, 1));
        let m = ColorMode::Truecolor;
        assert_eq!(
            cache.resolve_fg(0, 0, 0x808080, m, 6),
            0x808080,
            "first emit verbatim"
        );
        // ±3 per channel: held.
        assert_eq!(cache.resolve_fg(0, 0, 0x838383, m, 6), 0x808080);
        assert_eq!(cache.resolve_fg(0, 0, 0x7D7D7D, m, 6), 0x808080);
        // exactly the deadband: still held.
        assert_eq!(cache.resolve_fg(0, 0, 0x868686, m, 6), 0x808080);
        // ±12: emitted, and becomes the new anchor.
        assert_eq!(cache.resolve_fg(0, 0, 0x8C8C8C, m, 6), 0x8C8C8C);
        assert_eq!(cache.resolve_fg(0, 0, 0x8E8E8E, m, 6), 0x8C8C8C);
    }

    #[test]
    fn truecolor_drift_is_bounded_by_the_deadband() {
        // A gradient that creeps 2 units per frame must snap once it is >6 from
        // what is on screen — it cannot ride the deadband indefinitely.
        let mut cache = PresentCache::default();
        cache.reset(1, 1, (1, 1));
        let m = ColorMode::Truecolor;
        let mut shown = cache.resolve_bg(0, 0, 0x101010, m, 6);
        for step in 1..=20u32 {
            let v = 0x10 + 2 * step;
            let target = (v << 16) | (v << 8) | v;
            shown = cache.resolve_bg(0, 0, target, m, 6);
            let err = i32::from(rgb8(shown).0) - i32::from(v as u8);
            assert!(
                err.abs() <= 6,
                "step {step}: shown {shown:06X} vs target {target:06X}"
            );
        }
        assert_ne!(shown, 0x101010, "the colour did follow the gradient");
    }

    #[test]
    fn deadband_zero_disables_the_hold_in_both_modes() {
        let mut cache = PresentCache::default();
        cache.reset(1, 1, (1, 1));
        assert_eq!(
            cache.resolve_fg(0, 0, 0x808080, ColorMode::Truecolor, 0),
            0x808080
        );
        assert_eq!(
            cache.resolve_fg(0, 0, 0x818181, ColorMode::Truecolor, 0),
            0x818181
        );

        // Ansi256: a midpoint straddle flips with deadband 0 …
        // Saturated red so the (denser) grayscale ramp cannot win the match; the
        // red channel sits exactly between two cube levels (95 | 135).
        let a = (95 + 135) / 2; // 115
        let lo = pack(a - 1, 0, 0);
        let hi = pack(a + 1, 0, 0);
        let first = cache.resolve_bg(0, 0, lo, ColorMode::Ansi256, 0);
        let second = cache.resolve_bg(0, 0, hi, ColorMode::Ansi256, 0);
        assert_ne!(first, second, "no memory ⇒ the straddle flips");
    }

    #[test]
    fn ansi256_holds_across_a_midpoint_straddle_but_not_a_whole_step() {
        let mut cache = PresentCache::default();
        cache.reset(1, 1, (1, 1));
        let m = ColorMode::Ansi256;
        let a = (95 + 135) / 2; // 115 — red straddles cube levels 95 | 135
        let lo = pack(a - 1, 0, 0);
        let hi = pack(a + 1, 0, 0);
        let first = cache.resolve_bg(0, 0, lo, m, 6);
        let second = cache.resolve_bg(0, 0, hi, m, 6);
        assert_eq!(first, second, "straddle held with deadband 6");
        // The emitted value is an exact palette entry (identity under the map).
        let (r, g, b) = rgb8(first);
        let idx = rgb_to_ansi256(r, g, b);
        assert_eq!(ansi256_to_rgb(idx), (r, g, b));
        // A whole cube step away (≥40 units) must NOT be held.
        let far = pack(175, 0, 0);
        let third = cache.resolve_bg(0, 0, far, m, 6);
        assert_ne!(third, first);
        assert_eq!(rgb8(third), (175, 0, 0));
    }

    #[test]
    fn deadband_never_returns_a_non_palette_colour_in_ansi256() {
        for t in [0x000000u32, 0x123456, 0x7F7F7F, 0xFFFFFF, 0xC0FFEE] {
            let out = deadband_color(t, Some(0x101010), ColorMode::Ansi256, 6);
            let (r, g, b) = rgb8(out);
            assert_eq!(ansi256_to_rgb(rgb_to_ansi256(r, g, b)), (r, g, b));
        }
    }

    #[test]
    fn out_of_range_cell_passes_through() {
        let mut cache = PresentCache::default();
        cache.reset(2, 2, (2, 2));
        assert_eq!(
            cache.resolve_fg(5, 5, 0xABCDEF, ColorMode::Truecolor, 6),
            0xABCDEF
        );
    }

    #[test]
    fn config_default_and_env_parse() {
        assert_eq!(PresentConfig::default().color_deadband, 6);
        // We do not mutate the process environment in tests (other tests run in
        // parallel); with the variable unset `from_env` must equal the default.
        if std::env::var(DEADBAND_ENV).is_err() {
            assert_eq!(PresentConfig::from_env(), PresentConfig::default());
        }
    }
}
