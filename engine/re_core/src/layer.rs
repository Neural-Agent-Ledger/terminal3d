//! `re_core::layer` — the layer-composite primitive.
//!
//! The frame is built from a fixed, ordered [`CompositeStack`] of four layers,
//! folded back-to-front into a [`Framebuffer`] by [`CompositeStack::resolve`].
//! This is the *how* axis of tiered compositing: independent of *how often*
//! each layer recomputes (the temporal-tier axis, owned by `re_render`). A
//! slow far layer and a fast emitter layer feed the SAME cheap fold; no
//! coupling.
//!
//! Engine-agnostic: pure data + arithmetic, no terminal, no `bevy_render`. Each
//! [`Layer`] carries a `present` coverage mask so the fold costs O(populated
//! cells), not O(cells × layers) — absent cells are skipped entirely.

use crate::{Cell, Framebuffer};

/// How a [`Layer`] composites onto the cell accumulated from the layers below it.
///
/// The four-layer stack uses one blend each (spec §2 table):
/// `Opaque` for the far backdrop and the world-opaque pass, `OverKeepBg` for the
/// transparent (hologram/glass) pass, and `GlyphOnly` for emitter splats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerBlend {
    /// Replace everything (glyph / fg / bg / depth / style) where present. The
    /// backdrop and solid-geometry passes use this.
    Opaque,
    /// Take glyph + fg + style and set `depth = min(below, this)`, but KEEP the
    /// `bg` from the layer below — a transparent surface tints the foreground but
    /// shows the colour behind it. The hologram/glass pass uses this.
    OverKeepBg,
    /// Take glyph + fg only (keep bg, keep style), depth-tested against the cell
    /// already in the accumulator. Particle emitters use this.
    GlyphOnly,
    /// Additive colour glow: ADD this layer's `fg` and `bg` into the accumulator
    /// (per channel, clamped to 255) and KEEP the accumulator's glyph, style and
    /// depth — no glyph is ever overwritten. An absent source cell leaves the
    /// accumulator untouched. Used by colour-only god-rays / light-shaft emitters
    /// (see `re_world::EmitterKind::GodRays`) to tint the scene without erasing
    /// detail. Not depth-tested: a glow is volumetric and brightens whatever it
    /// covers.
    ColorAdd,
    /// Intermediate-colour haze: alpha-weighted LERP of the accumulator's `fg`/`bg`
    /// TOWARD this layer's `fg`/`bg`, keeping the accumulator's glyph, style and
    /// depth — no glyph is ever overwritten. The per-cell alpha lives in the HIGH
    /// byte of the source cell's `fg`/`bg` (i.e. the source colours are read as
    /// `0xAARRGGBB`); `a = alpha / 255` is the lerp weight. `alpha = 0` (or an
    /// absent source cell) leaves the accumulator untouched; `alpha = 255` fully
    /// replaces the colour but still keeps the glyph. This is the recommended mode
    /// for atmospheric/haze emitters — it tints toward the emitter colour rather
    /// than brightening, so it reads as fog/mist rather than a light source.
    /// [`ColorAdd`](Self::ColorAdd) stays for bright additive light shafts. Not
    /// depth-tested: haze is volumetric and tints whatever it covers.
    ColorMix,
}

/// Add two packed `0x00RRGGBB` colours channel-by-channel, clamping each channel
/// to `0xFF`. The alpha/high byte of the result is always `0`. Used by the
/// [`LayerBlend::ColorAdd`] composite path.
#[inline]
fn add_rgb_clamped(a: u32, b: u32) -> u32 {
    let r = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF)).min(0xFF);
    let g = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF)).min(0xFF);
    let bl = ((a & 0xFF) + (b & 0xFF)).min(0xFF);
    (r << 16) | (g << 8) | bl
}

/// Per-channel linear blend of two packed `0x00RRGGBB` colours: returns
/// `a + (b - a) * t` per channel with `t` clamped to `0.0..=1.0`. The alpha/high
/// byte of the result is always `0`. Used by the [`LayerBlend::ColorMix`]
/// composite path (`t == 0` → `a`, `t == 1` → `b`, `t == 0.5` → the midpoint).
#[inline]
fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |ca: u32, cb: u32| -> u32 {
        let ca = ca as f32;
        let cb = cb as f32;
        (ca + (cb - ca) * t).round().clamp(0.0, 255.0) as u32
    };
    let r = lerp((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    let g = lerp((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    let bl = lerp(a & 0xFF, b & 0xFF);
    (r << 16) | (g << 8) | bl
}

/// A single dense composite layer: a `Cell` grid plus a parallel `present`
/// coverage mask. A cell only participates in the fold where `present[i]` is
/// `true`; everywhere else the layer is transparent and contributes nothing.
#[derive(Clone, Debug)]
pub struct Layer {
    cols: usize,
    rows: usize,
    cells: Vec<Cell>,
    present: Vec<bool>,
    /// How this layer composites onto the accumulator (see [`LayerBlend`]).
    pub blend: LayerBlend,
}

impl Layer {
    /// Allocate a `cols`×`rows` layer of default cells, all marked absent
    /// (`present = false`), with the given [`LayerBlend`].
    pub fn new(cols: usize, rows: usize, blend: LayerBlend) -> Self {
        Layer {
            cols,
            rows,
            cells: vec![Cell::default(); cols * rows],
            present: vec![false; cols * rows],
            blend,
        }
    }

    /// Grid width in cells.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Grid height in cells.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `(cols, rows)` of this layer.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    /// Resize (and clear) the layer if the dimensions changed. After a resize the
    /// whole layer is absent (`present = false`) — the producer must refill it.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        if cols != self.cols || rows != self.rows {
            self.cols = cols;
            self.rows = rows;
            self.cells.clear();
            self.cells.resize(cols * rows, Cell::default());
            self.present.clear();
            self.present.resize(cols * rows, false);
        } else {
            self.clear();
        }
    }

    /// Mark every cell absent (`present = false`). Cheap — leaves the cell data
    /// in place; the coverage mask alone decides participation in the fold.
    #[inline]
    pub fn clear(&mut self) {
        self.present.fill(false);
    }

    #[inline]
    fn index(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    /// Write `cell` at `(col, row)` and mark it present. Out-of-bounds writes are
    /// ignored. This is the producer's only entry — a written cell always
    /// participates in the fold.
    #[inline]
    pub fn set(&mut self, col: usize, row: usize, cell: Cell) {
        if col < self.cols && row < self.rows {
            let i = self.index(col, row);
            self.cells[i] = cell;
            self.present[i] = true;
        }
    }

    /// Read `(col, row)` as `Some(&cell)` only if it is present; `None` if out of
    /// bounds OR absent.
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> Option<&Cell> {
        if col < self.cols && row < self.rows {
            let i = self.index(col, row);
            if self.present[i] {
                return Some(&self.cells[i]);
            }
        }
        None
    }

    /// Whether `(col, row)` is present (covered). `false` if out of bounds.
    #[inline]
    pub fn is_present(&self, col: usize, row: usize) -> bool {
        col < self.cols && row < self.rows && self.present[self.index(col, row)]
    }

    /// The raw cell at `(col, row)` ignoring the present mask (for producers that
    /// radial-sample one layer into another). `None` if out of bounds.
    #[inline]
    pub fn cell_raw(&self, col: usize, row: usize) -> Option<&Cell> {
        if col < self.cols && row < self.rows {
            Some(&self.cells[self.index(col, row)])
        } else {
            None
        }
    }
}

/// The fixed, ordered five-layer stack folded into a [`Framebuffer`] each frame.
///
/// Back-to-front order: `far` → `opaque` → `transparent` → `emitters` → `godrays`.
/// The first four are the tiered-composite spec §2 layers; `godrays` is a
/// colour-only atmospheric pass ([`LayerBlend::ColorMix`]) that tints whatever the
/// fold has already produced WITHOUT touching any glyph (see
/// `re_world::EmitterKind::GodRays`). It sits ABOVE `emitters` so a glyph-splat
/// emitter and a god-rays shaft can coexist in one scene: the standard emitter
/// draws its glyph, then the god-rays haze tints it. The blends are wired once in
/// [`new`](Self::new) so a producer never has to remember which slot uses which.
#[derive(Clone, Debug)]
pub struct CompositeStack {
    /// Layer 0 — cached cylindrical panorama (radial sample). `Opaque`.
    pub far: Layer,
    /// Layer 1 — DDA raycast of solid voxels. `Opaque`.
    pub opaque: Layer,
    /// Layer 2 — hologram/glass materials (glyph+fg over bg behind). `OverKeepBg`.
    pub transparent: Layer,
    /// Layer 3 — STANDARD particle splats, depth-tested vs layer 1. `GlyphOnly`.
    pub emitters: Layer,
    /// Layer 4 — colour-only GOD-RAYS / light-shaft haze. `ColorMix`: lerps the
    /// accumulated colour toward the shaft colour by the alpha packed in the high
    /// byte of `fg`/`bg`, keeping the glyph/style/depth below. Not depth-tested (a
    /// volumetric glow tints whatever it covers). Populated by
    /// `re_world::EmitterKind::GodRays` emitters.
    pub godrays: Layer,
}

impl CompositeStack {
    /// Allocate a `cols`×`rows` stack with the five blends wired from the spec §2
    /// table (+ the god-rays `ColorMix` slot). All five layers start fully absent.
    pub fn new(cols: usize, rows: usize) -> Self {
        CompositeStack {
            far: Layer::new(cols, rows, LayerBlend::Opaque),
            opaque: Layer::new(cols, rows, LayerBlend::Opaque),
            transparent: Layer::new(cols, rows, LayerBlend::OverKeepBg),
            emitters: Layer::new(cols, rows, LayerBlend::GlyphOnly),
            godrays: Layer::new(cols, rows, LayerBlend::ColorMix),
        }
    }

    /// `(cols, rows)` — taken from the `far` layer (all four share dims).
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        self.far.dims()
    }

    /// Resize all five layers if the dimensions changed (clearing them).
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.far.resize(cols, rows);
        self.opaque.resize(cols, rows);
        self.transparent.resize(cols, rows);
        self.emitters.resize(cols, rows);
        self.godrays.resize(cols, rows);
    }

    /// Mark every layer absent. Cheap — the producers refill what they own.
    pub fn clear_all(&mut self) {
        self.far.clear();
        self.opaque.clear();
        self.transparent.clear();
        self.emitters.clear();
        self.godrays.clear();
    }

    /// Back-to-front fold into `out`. O(populated cells), not O(cells × layers):
    /// a cell is only written when at least one layer is present there.
    ///
    /// Per cell (spec §2):
    /// - start = the `far` cell (or the default empty cell if `far` is absent),
    /// - if `opaque` is present → replace all (glyph/fg/bg/depth/style),
    /// - if `transparent` is present → take glyph+fg+style, KEEP bg from below,
    ///   `depth = min`,
    /// - if `emitters` is present AND `emitter.depth ≤ current depth` → take
    ///   glyph+fg, keep bg (and keep style + depth — `GlyphOnly`).
    ///
    /// Cells where no layer is present are left untouched in `out` (the caller is
    /// expected to have cleared it, or to be reusing a dirty-rect bound).
    pub fn resolve(&self, out: &mut Framebuffer) {
        let (cols, rows) = self.dims();
        let cols = cols.min(out.cols());
        let rows = rows.min(out.rows());

        for row in 0..rows {
            for col in 0..cols {
                let far = self.far.get(col, row);
                let opaque = self.opaque.get(col, row);
                let transparent = self.transparent.get(col, row);
                let emitter = self.emitters.get(col, row);
                let godray = self.godrays.get(col, row);

                // Skip absent cells entirely — the economy claim.
                if far.is_none()
                    && opaque.is_none()
                    && transparent.is_none()
                    && emitter.is_none()
                    && godray.is_none()
                {
                    continue;
                }

                // start = far (or default empty).
                let mut acc = far.copied().unwrap_or_default();

                // opaque → replace all.
                if let Some(o) = opaque {
                    acc = *o;
                }

                // transparent → glyph+fg+style over kept bg, depth = min.
                if let Some(t) = transparent {
                    let bg = acc.bg;
                    let depth = acc.depth.min(t.depth);
                    acc.glyph = t.glyph;
                    acc.fg = t.fg;
                    acc.style = t.style;
                    acc.bg = bg;
                    acc.depth = depth;
                }

                // emitter (STANDARD glyph splats) → composited per the layer's blend.
                if let Some(e) = emitter {
                    apply_emitter_blend(&mut acc, e, self.emitters.blend);
                }

                // godrays (COLOUR-ONLY light shafts) → tint LAST, above the glyph the
                // standard emitter may have just drawn. Its `ColorMix`/`ColorAdd`
                // blend keeps the glyph/style/depth, so a shaft tints both walls and
                // any emitter glyph beneath it without erasing detail.
                if let Some(g) = godray {
                    apply_emitter_blend(&mut acc, g, self.godrays.blend);
                }

                out.set(col, row, acc);
            }
        }
    }
}

/// Composite one emitter-layer source cell `src` onto the accumulator `acc` using
/// `blend`. Shared by the `emitters` (glyph splats) and `godrays` (colour-only
/// haze) slots so the two passes agree on every blend's semantics.
#[inline]
fn apply_emitter_blend(acc: &mut Cell, src: &Cell, blend: LayerBlend) {
    match blend {
        LayerBlend::ColorAdd => {
            // Additive glow: add fg+bg into the accumulator, keep glyph/style/depth.
            // Not depth-tested — a shaft tints whatever it covers without erasing
            // detail.
            acc.fg = add_rgb_clamped(acc.fg, src.fg);
            acc.bg = add_rgb_clamped(acc.bg, src.bg);
        }
        LayerBlend::ColorMix => {
            // Intermediate-colour haze: lerp the accumulator's colours toward the
            // source's, weighted by the alpha packed in the high byte of the source
            // fg/bg. Keep glyph/depth — a haze tints, never erases. `style` is
            // carried through ONLY when the source authored a real one (Integration
            // 2026-09-23: a plain haze cell's `style` is `CellStyle::NONE`
            // (`is_empty()`), so this is a no-op for every existing caller —
            // see the `GlyphOnly` arm below for why this is gated the same way.
            let fa = (src.fg >> 24) as f32 / 255.0;
            let ba = (src.bg >> 24) as f32 / 255.0;
            acc.fg = lerp_rgb(acc.fg & 0x00FF_FFFF, src.fg & 0x00FF_FFFF, fa);
            acc.bg = lerp_rgb(acc.bg & 0x00FF_FFFF, src.bg & 0x00FF_FFFF, ba);
            if !src.style.is_empty() {
                acc.style = src.style;
            }
        }
        LayerBlend::Opaque => {
            if src.depth <= acc.depth {
                *acc = *src;
            }
        }
        LayerBlend::OverKeepBg => {
            if src.depth <= acc.depth {
                let bg = acc.bg;
                acc.glyph = src.glyph;
                acc.fg = src.fg;
                acc.style = src.style;
                acc.bg = bg;
                acc.depth = acc.depth.min(src.depth);
            }
        }
        LayerBlend::GlyphOnly => {
            if src.depth <= acc.depth {
                acc.glyph = src.glyph;
                acc.fg = src.fg;
                // bg and depth kept (GlyphOnly). `style` is carried through ONLY
                // when the source authored a real one (Integration 2026-09-23,
                // spec §4.2 W10 "ripple hem" / trail direction-glyph styling):
                // an ordinary particle splat's style is `CellStyle::NONE`
                // (`is_empty()` true), so this is a no-op for the common case and
                // the accumulator's existing style (e.g. a material's own
                // underline accent) survives exactly as before. A particle that
                // DOES author a style (a ripple's curly underline hem, a trail's
                // direction-glyph substyle) now reaches the composited frame,
                // closing the gap `docs/EMITTERS.md` "Ripples (W10)" documented.
                if !src.style.is_empty() {
                    acc.style = src.style;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(glyph: char, fg: u32, bg: u32, depth: f32) -> Cell {
        Cell {
            glyph: u32::from(glyph),
            fg,
            bg,
            depth,
            ..Cell::default()
        }
    }

    #[test]
    fn layer_present_mask_starts_absent() {
        let l = Layer::new(3, 2, LayerBlend::Opaque);
        assert_eq!(l.dims(), (3, 2));
        assert!(l.get(0, 0).is_none());
        assert!(!l.is_present(0, 0));
    }

    #[test]
    fn layer_set_marks_present_and_clear_resets() {
        let mut l = Layer::new(2, 2, LayerBlend::Opaque);
        l.set(1, 0, cell('X', 0xFF, 0, 1.0));
        assert!(l.is_present(1, 0));
        assert_eq!(l.get(1, 0).unwrap().glyph, u32::from('X'));
        l.clear();
        assert!(!l.is_present(1, 0));
        // cell data is still there via cell_raw, just not present.
        assert_eq!(l.cell_raw(1, 0).unwrap().glyph, u32::from('X'));
    }

    #[test]
    fn resolve_skips_absent_cells() {
        // out pre-seeded with a sentinel; an all-absent stack must leave it.
        let stack = CompositeStack::new(2, 1);
        let mut out = Framebuffer::new(2, 1);
        out.set(0, 0, cell('S', 0x111, 0x222, 5.0));
        stack.resolve(&mut out);
        assert_eq!(out.get(0, 0).unwrap().glyph, u32::from('S'));
        assert_eq!(out.get(0, 0).unwrap().bg, 0x222);
    }

    #[test]
    fn resolve_opaque_over_far() {
        let mut stack = CompositeStack::new(1, 1);
        stack.far.set(0, 0, cell('f', 0x0F, 0x0B, 100.0));
        stack.opaque.set(0, 0, cell('O', 0xAA, 0xBB, 3.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        // opaque replaces everything.
        assert_eq!(c.glyph, u32::from('O'));
        assert_eq!(c.fg, 0xAA);
        assert_eq!(c.bg, 0xBB);
        assert_eq!(c.depth, 3.0);
    }

    #[test]
    fn resolve_far_only_when_no_opaque() {
        let mut stack = CompositeStack::new(1, 1);
        stack.far.set(0, 0, cell('f', 0x0F, 0x0B, 100.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('f'));
        assert_eq!(c.bg, 0x0B);
        assert_eq!(c.depth, 100.0);
    }

    #[test]
    fn resolve_transparent_keeps_bg_below() {
        let mut stack = CompositeStack::new(1, 1);
        stack.opaque.set(0, 0, cell('O', 0xAA, 0xBB, 5.0));
        stack.transparent.set(0, 0, cell('T', 0xCC, 0x99, 2.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        // glyph + fg come from transparent, bg kept from opaque below.
        assert_eq!(c.glyph, u32::from('T'));
        assert_eq!(c.fg, 0xCC);
        assert_eq!(c.bg, 0xBB, "transparent must keep the bg from below");
        // depth = min(5, 2) = 2.
        assert_eq!(c.depth, 2.0);
    }

    #[test]
    fn resolve_emitter_depth_test_in_front() {
        let mut stack = CompositeStack::new(1, 1);
        stack.opaque.set(0, 0, cell('O', 0xAA, 0xBB, 5.0));
        // emitter in front (depth 2 <= 5) → its glyph+fg win, bg kept.
        stack.emitters.set(0, 0, cell('*', 0xEE, 0x00, 2.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('*'));
        assert_eq!(c.fg, 0xEE);
        assert_eq!(c.bg, 0xBB, "emitter is GlyphOnly — keeps bg");
        assert_eq!(c.depth, 5.0, "GlyphOnly keeps the accumulator depth");
    }

    /// Integration 2026-09-23: a `GlyphOnly` emitter cell with a real (non-empty)
    /// `style` — e.g. a ripple's curly underline hem — must now reach the resolved
    /// frame, closing the composite-level gap `docs/EMITTERS.md` "Ripples (W10)"
    /// documented.
    #[test]
    fn resolve_emitter_carries_a_non_empty_style_through() {
        let mut stack = CompositeStack::new(1, 1);
        stack.opaque.set(0, 0, cell('O', 0xAA, 0xBB, 5.0));
        let mut hem = cell('*', 0xEE, 0x00, 2.0);
        hem.style.underline = 0x00_40_A0_FF;
        hem.style.underline_style = crate::UnderlineStyle::Curly;
        stack.emitters.set(0, 0, hem);
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.style.underline_color(), Some(0x40_A0_FF));
        assert_eq!(c.style.underline_style(), crate::UnderlineStyle::Curly);
    }

    /// The ORDINARY case — an emitter cell with the default (empty) style — must
    /// NOT disturb whatever style the accumulator already carries from below (a
    /// material's own underline accent, say). This is the back-compat half of the
    /// same fix: only a NON-empty source style overwrites.
    #[test]
    fn resolve_emitter_with_empty_style_keeps_the_accumulators_style() {
        let mut stack = CompositeStack::new(1, 1);
        let mut wall = cell('O', 0xAA, 0xBB, 5.0);
        wall.style.underline = 0x00_11_22_33;
        wall.style.underline_style = crate::UnderlineStyle::Dotted;
        stack.opaque.set(0, 0, wall);
        // A plain particle splat — CellStyle::default() (empty) — must be a no-op
        // on style.
        stack.emitters.set(0, 0, cell('*', 0xEE, 0x00, 2.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(
            c.style.underline_color(),
            Some(0x11_22_33),
            "the wall's underline must survive a plain splat"
        );
        assert_eq!(c.style.underline_style(), crate::UnderlineStyle::Dotted);
    }

    #[test]
    fn resolve_emitter_occluded_behind_wall() {
        let mut stack = CompositeStack::new(1, 1);
        stack.opaque.set(0, 0, cell('O', 0xAA, 0xBB, 2.0));
        // emitter behind the wall (depth 9 > 2) → skipped.
        stack.emitters.set(0, 0, cell('*', 0xEE, 0x00, 9.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('O'), "occluded emitter must not draw");
        assert_eq!(c.fg, 0xAA);
    }

    #[test]
    fn resolve_full_stack_order() {
        // far < opaque < transparent < emitter — verify the whole fold.
        let mut stack = CompositeStack::new(1, 1);
        stack.far.set(0, 0, cell('f', 0x01, 0x10, 50.0));
        stack.opaque.set(0, 0, cell('O', 0x02, 0x20, 6.0));
        stack.transparent.set(0, 0, cell('T', 0x03, 0x30, 4.0));
        stack.emitters.set(0, 0, cell('*', 0x04, 0x40, 1.0));
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        // emitter glyph+fg on top.
        assert_eq!(c.glyph, u32::from('*'));
        assert_eq!(c.fg, 0x04);
        // bg traces back: opaque bg kept through transparent + emitter.
        assert_eq!(c.bg, 0x20);
        // depth = min(6,4) from transparent; emitter is GlyphOnly so it keeps it.
        assert_eq!(c.depth, 4.0);
    }

    #[test]
    fn add_rgb_clamped_saturates_per_channel() {
        // No overflow case.
        assert_eq!(add_rgb_clamped(0x10_20_30, 0x01_02_03), 0x11_22_33);
        // Each channel clamps independently at 0xFF.
        assert_eq!(add_rgb_clamped(0xF0_80_01, 0x20_90_02), 0xFF_FF_03);
        // High/alpha byte is dropped.
        assert_eq!(add_rgb_clamped(0xFF_00_00_00, 0x00_00_00), 0);
    }

    #[test]
    fn resolve_color_add_adds_color_keeps_glyph() {
        // Build a stack whose emitter layer is ColorAdd (god-rays style).
        let mut stack = CompositeStack {
            far: Layer::new(1, 1, LayerBlend::Opaque),
            opaque: Layer::new(1, 1, LayerBlend::Opaque),
            transparent: Layer::new(1, 1, LayerBlend::OverKeepBg),
            emitters: Layer::new(1, 1, LayerBlend::ColorAdd),
            godrays: Layer::new(1, 1, LayerBlend::ColorMix),
        };
        // Base scene: a wall glyph with distinctive fg/bg/depth/style.
        let mut base = cell('W', 0x10_20_30, 0x05_05_05, 4.0);
        base.style = base.style.with_underline_color(0x123456);
        stack.opaque.set(0, 0, base);
        // The glow: pure colour, no meaningful glyph (depth far behind the wall to
        // prove ColorAdd is NOT depth-tested).
        stack
            .emitters
            .set(0, 0, cell(' ', 0x40_10_05, 0x10_10_10, 99.0));

        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();

        // Glyph, style and depth come from the base — the glow does not overwrite them.
        assert_eq!(c.glyph, u32::from('W'), "ColorAdd must keep the base glyph");
        assert_eq!(c.depth, 4.0, "ColorAdd must keep the base depth");
        assert_eq!(c.style, base.style, "ColorAdd must keep the base style");
        // Colours are added (clamped) into the base.
        assert_eq!(c.fg, 0x50_30_35, "fg must be base+glow per channel");
        assert_eq!(c.bg, 0x15_15_15, "bg must be base+glow per channel");
    }

    #[test]
    fn resolve_color_add_empty_cell_leaves_base_untouched() {
        let mut stack = CompositeStack {
            far: Layer::new(1, 1, LayerBlend::Opaque),
            opaque: Layer::new(1, 1, LayerBlend::Opaque),
            transparent: Layer::new(1, 1, LayerBlend::OverKeepBg),
            emitters: Layer::new(1, 1, LayerBlend::ColorAdd),
            godrays: Layer::new(1, 1, LayerBlend::ColorMix),
        };
        let base = cell('W', 0x10_20_30, 0x05_05_05, 4.0);
        stack.opaque.set(0, 0, base);
        // Emitter layer left ABSENT at (0,0): nothing to add.
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('W'));
        assert_eq!(c.fg, 0x10_20_30, "absent ColorAdd cell must not change fg");
        assert_eq!(c.bg, 0x05_05_05, "absent ColorAdd cell must not change bg");
        assert_eq!(c.depth, 4.0);
    }

    #[test]
    fn lerp_rgb_endpoints_and_midpoint() {
        // t = 0 returns a; t = 1 returns b.
        assert_eq!(lerp_rgb(0x10_20_30, 0xA0_B0_C0, 0.0), 0x10_20_30);
        assert_eq!(lerp_rgb(0x10_20_30, 0xA0_B0_C0, 1.0), 0xA0_B0_C0);
        // t = 0.5 is the per-channel midpoint (0x10+0xA0)/2 = 0x58, etc.
        assert_eq!(lerp_rgb(0x10_20_30, 0xA0_B0_C0, 0.5), 0x58_68_78);
        // t is clamped outside [0,1].
        assert_eq!(lerp_rgb(0x10_20_30, 0xA0_B0_C0, -1.0), 0x10_20_30);
        assert_eq!(lerp_rgb(0x10_20_30, 0xA0_B0_C0, 2.0), 0xA0_B0_C0);
    }

    fn color_mix_stack() -> CompositeStack {
        CompositeStack {
            far: Layer::new(1, 1, LayerBlend::Opaque),
            opaque: Layer::new(1, 1, LayerBlend::Opaque),
            transparent: Layer::new(1, 1, LayerBlend::OverKeepBg),
            emitters: Layer::new(1, 1, LayerBlend::ColorMix),
            godrays: Layer::new(1, 1, LayerBlend::ColorMix),
        }
    }

    #[test]
    fn resolve_color_mix_half_alpha_is_midpoint_keeps_glyph() {
        let mut stack = color_mix_stack();
        let mut base = cell('W', 0x10_20_30, 0x40_40_40, 4.0);
        base.style = base.style.with_underline_color(0x123456);
        stack.opaque.set(0, 0, base);
        // alpha = 128 in the high byte of fg/bg.
        stack
            .emitters
            .set(0, 0, cell(' ', 0x80_A0_B0_C0, 0x80_00_00_00, 99.0));

        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();

        // Glyph, style and depth untouched.
        assert_eq!(c.glyph, u32::from('W'), "ColorMix must keep the base glyph");
        assert_eq!(c.depth, 4.0, "ColorMix must keep the base depth");
        assert_eq!(c.style, base.style, "ColorMix must keep the base style");
        // a = 128/255 ≈ 0.50196. fg: lerp(0x10,0xA0)→0x58, (0x20,0xB0)→0x68,
        // (0x30,0xC0)→0x78. bg: lerp(0x40,0x00) toward black → 0x20.
        assert_eq!(c.fg, 0x58_68_78, "fg must be the alpha-weighted blend");
        assert_eq!(c.bg, 0x20_20_20, "bg must blend toward 0x000000");
    }

    #[test]
    fn resolve_color_mix_zero_alpha_leaves_base_untouched() {
        let mut stack = color_mix_stack();
        let base = cell('W', 0x10_20_30, 0x05_05_05, 4.0);
        stack.opaque.set(0, 0, base);
        // Present source, but alpha = 0 in the high byte → no change.
        stack
            .emitters
            .set(0, 0, cell(' ', 0x00_FF_FF_FF, 0x00_FF_FF_FF, 99.0));

        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('W'));
        assert_eq!(c.fg, 0x10_20_30, "alpha=0 ColorMix must not change fg");
        assert_eq!(c.bg, 0x05_05_05, "alpha=0 ColorMix must not change bg");
        assert_eq!(c.depth, 4.0);
    }

    #[test]
    fn resolve_color_mix_absent_cell_leaves_base_untouched() {
        let mut stack = color_mix_stack();
        let base = cell('W', 0x10_20_30, 0x05_05_05, 4.0);
        stack.opaque.set(0, 0, base);
        // Emitter layer left ABSENT at (0,0).
        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('W'));
        assert_eq!(c.fg, 0x10_20_30, "absent ColorMix cell must not change fg");
        assert_eq!(c.bg, 0x05_05_05, "absent ColorMix cell must not change bg");
    }

    #[test]
    fn resolve_color_mix_full_alpha_replaces_color_keeps_glyph() {
        let mut stack = color_mix_stack();
        let base = cell('W', 0x10_20_30, 0x05_05_05, 4.0);
        stack.opaque.set(0, 0, base);
        // alpha = 255 → colour fully replaced, glyph kept.
        stack
            .emitters
            .set(0, 0, cell(' ', 0xFF_AA_BB_CC, 0xFF_11_22_33, 99.0));

        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();
        assert_eq!(c.glyph, u32::from('W'), "ColorMix must keep the base glyph");
        assert_eq!(c.fg, 0x00_AA_BB_CC, "alpha=255 fully replaces fg colour");
        assert_eq!(c.bg, 0x00_11_22_33, "alpha=255 fully replaces bg colour");
        assert_eq!(c.depth, 4.0);
    }

    #[test]
    fn godrays_slot_tints_over_a_standard_emitter_glyph() {
        // A wall, a STANDARD emitter glyph in front of it, and a GOD-RAYS shaft over
        // both. The god-rays slot (`ColorMix`, default wiring) must tint the colour
        // while KEEPING the emitter glyph below it — proving standard + god-rays
        // coexist in one fold.
        let mut stack = CompositeStack::new(1, 1);
        // Wall behind.
        stack
            .opaque
            .set(0, 0, cell('W', 0x10_20_30, 0x05_05_05, 5.0));
        // A standard glyph emitter in front (GlyphOnly slot keeps wall bg).
        stack.emitters.set(0, 0, cell('*', 0x40_40_40, 0x00, 2.0));
        // A warm god-rays shaft at half alpha (128) — colour only, glyph is space.
        stack
            .godrays
            .set(0, 0, cell(' ', 0x80_FF_E0_A0, 0x80_FF_E0_A0, 99.0));

        let mut out = Framebuffer::new(1, 1);
        stack.resolve(&mut out);
        let c = out.get(0, 0).unwrap();

        // The emitter glyph survives the god-rays tint (the whole point).
        assert_eq!(
            c.glyph,
            u32::from('*'),
            "god-rays must keep the emitter glyph"
        );
        // fg tinted toward warm: lerp(0x40, 0xFF, 128/255) ≈ 0x9F per top channel.
        // Just assert it moved toward the warm colour, not exact rounding.
        assert!(
            (c.fg >> 16) & 0xFF > 0x40,
            "red channel must rise toward warm"
        );
        assert_ne!(c.fg, 0x40_40_40, "fg must be tinted by the shaft");
        // GlyphOnly kept the wall bg (0x05_05_05); god-rays then tints THAT toward warm.
        assert_ne!(c.bg, 0x05_05_05, "bg must be tinted by the shaft");
        // Depth is the wall's (GlyphOnly + ColorMix both keep depth).
        assert_eq!(c.depth, 5.0, "god-rays keeps the accumulated depth");
    }

    #[test]
    fn resize_clears_presence() {
        let mut stack = CompositeStack::new(2, 2);
        stack.opaque.set(0, 0, cell('A', 1, 1, 1.0));
        stack.resize(4, 4);
        assert_eq!(stack.dims(), (4, 4));
        assert!(!stack.opaque.is_present(0, 0));
    }
}
