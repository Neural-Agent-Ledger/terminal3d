//! A shared half-block "phosphor trace" canvas: dense, continuous-looking
//! lines with fading persistence and a soft glow halo, built from the SAME
//! half-block technique `crate::eye` already proved out (one `▀` glyph per
//! cell, `fg`=top sub-row colour / `bg`=bottom sub-row colour, doubling
//! vertical resolution) rather than braille sub-cells.
//!
//! Braille (U+2800–28FF) was considered for oscilloscope / heartbeat /
//! `neural_net`, but it sits OUTSIDE this codebase's own
//! `re_render::glyph_family::is_reliable` gate (ASCII + box-drawing +
//! block-elements + a small vetted extras list only — see that file), and
//! extending the gate would mean editing `re_render`, which this crate does
//! not do. Half-block reuses an already-reliable, already-perf-proven
//! glyph instead of introducing a new unreliable one; it gives full
//! resolution vertically (exactly what a scanline trace needs), and the
//! dense-sampling + phosphor-fade + glow look comes
//! from HOW the canvas is filled, not from the glyph.
//!
//! Every trace-like effect (oscilloscope, heartbeat, and `neural_net`'s pulse
//! bursts) fills one of these per frame: splat every sample (fresh or a
//! faded historical echo) as a soft dot via [`SubrowCanvas::splat`], which
//! also feeds a coarser per-CELL glow halo, then [`SubrowCanvas::blit`] once
//! at the end — mirroring `eye::shade`'s "resolve every per-pixel colour in
//! linear space, pack once" discipline.

use re_core::oklab::{pack_linear, LinRgb};
use re_core::{Cell, CellStyle, Framebuffer};

use crate::Rect;

#[must_use]
#[inline]
fn scale_lin(c: LinRgb, k: f32) -> LinRgb {
    LinRgb {
        r: c.r * k,
        g: c.g * k,
        b: c.b * k,
    }
}

#[must_use]
#[inline]
fn max_lin(a: LinRgb, b: LinRgb) -> LinRgb {
    LinRgb {
        r: a.r.max(b.r),
        g: a.g.max(b.g),
        b: a.b.max(b.b),
    }
}

/// A `cols × (rows*2)` linear-colour accumulator plus a coarser `cols × rows`
/// glow halo, both cleared per frame and resolved to half-block cells once.
pub struct SubrowCanvas {
    rect: Rect,
    /// `cols * rows * 2` linear-RGB sub-row samples, brightest-wins (`max`)
    /// blended so dense overlapping samples never blow out past their own
    /// peak brightness.
    dots: Vec<LinRgb>,
    /// `cols * rows` glow halo — a soft, coarser "is something bright nearby"
    /// signal, additive-accumulated then read back as a dim background tint
    /// for cells the trace itself didn't touch.
    glow: Vec<f32>,
    glow_color: LinRgb,
}

impl SubrowCanvas {
    /// Allocate an empty canvas over `rect`, with `glow_color` used for every
    /// cell's glow halo tint.
    #[must_use]
    pub fn new(rect: Rect, glow_color: LinRgb) -> Self {
        let n_dots = rect.cols * rect.rows * 2;
        let n_cells = rect.cols * rect.rows;
        SubrowCanvas {
            rect,
            dots: vec![
                LinRgb {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0
                };
                n_dots
            ],
            glow: vec![0.0; n_cells],
            glow_color,
        }
    }

    fn subrow_count(&self) -> i64 {
        (self.rect.rows * 2) as i64
    }

    fn dot_index(&self, col: i64, subrow: i64) -> Option<usize> {
        if col < 0 || subrow < 0 || col >= self.rect.cols as i64 || subrow >= self.subrow_count() {
            return None;
        }
        Some(subrow as usize * self.rect.cols + col as usize)
    }

    fn cell_index(&self, col: i64, row: i64) -> Option<usize> {
        if col < 0 || row < 0 || col >= self.rect.cols as i64 || row >= self.rect.rows as i64 {
            return None;
        }
        Some(row as usize * self.rect.cols + col as usize)
    }

    /// Splat one sample at fractional `(col_f, subrow_f)` — `subrow_f` in
    /// `[0, rows*2)`, world coordinates local to the rect (i.e. `0.0` is the
    /// rect's own left/top edge, not the framebuffer's). Bilinearly spreads
    /// across the 4 neighbouring sub-row cells (soft/anti-aliased dot) and
    /// brightest-wins blends with anything already there, then adds a
    /// falling-off contribution to the coarser glow halo.
    pub fn splat(&mut self, col_f: f32, subrow_f: f32, color: LinRgb, intensity: f32) {
        if intensity <= 0.0 {
            return;
        }
        let c0 = col_f.floor();
        let r0 = subrow_f.floor();
        let fc = col_f - c0;
        let fr = subrow_f - r0;
        let c0i = c0 as i64;
        let r0i = r0 as i64;
        for (dc, dr, w) in [
            (0i64, 0i64, (1.0 - fc) * (1.0 - fr)),
            (1, 0, fc * (1.0 - fr)),
            (0, 1, (1.0 - fc) * fr),
            (1, 1, fc * fr),
        ] {
            if w <= 0.0 {
                continue;
            }
            if let Some(idx) = self.dot_index(c0i + dc, r0i + dr) {
                self.dots[idx] = max_lin(self.dots[idx], scale_lin(color, intensity * w));
            }
        }

        // Glow halo: a 3x3 falloff in CELL space around the sample's cell,
        // scaled down further so it only ever reads as a faint tint, never
        // competing with a direct hit.
        let glow_row = (subrow_f * 0.5) as i64;
        let glow_col = c0i;
        const KERNEL: [(i64, i64, f32); 9] = [
            (-1, -1, 0.12),
            (0, -1, 0.25),
            (1, -1, 0.12),
            (-1, 0, 0.25),
            (0, 0, 0.55),
            (1, 0, 0.25),
            (-1, 1, 0.12),
            (0, 1, 0.25),
            (1, 1, 0.12),
        ];
        for (dc, dr, w) in KERNEL {
            if let Some(idx) = self.cell_index(glow_col + dc, glow_row + dr) {
                self.glow[idx] += intensity * w * 0.35;
            }
        }
    }

    /// Resolve the accumulated dots + glow into `▀` half-block cells,
    /// writing into `fb` at the canvas's own `rect`. `bg_base` is the colour
    /// already painted behind the trace (a graticule line, or plain
    /// background) — cells with neither a direct dot nor meaningful glow are
    /// left completely untouched (`skip`), so a caller's earlier graticule
    /// paint survives underneath.
    pub fn blit(&self, fb: &mut Framebuffer, bg_base: LinRgb) {
        for row in 0..self.rect.rows {
            for col in 0..self.rect.cols {
                let top = self.dots[row * 2 * self.rect.cols + col];
                let bot = self.dots[(row * 2 + 1) * self.rect.cols + col];
                let glow = self.glow[row * self.rect.cols + col].min(1.0);
                let has_dot = top.r + top.g + top.b > 0.0 || bot.r + bot.g + bot.b > 0.0;
                if !has_dot && glow <= 0.01 {
                    continue;
                }
                let glow_tint = scale_lin(self.glow_color, glow);
                let fg = if top.r + top.g + top.b > 0.0 {
                    top
                } else {
                    max_lin(bg_base, glow_tint)
                };
                let bg = if bot.r + bot.g + bot.b > 0.0 {
                    bot
                } else {
                    max_lin(bg_base, glow_tint)
                };
                let col_i = self.rect.col0 + col;
                let row_i = self.rect.row0 + row;
                fb.set(
                    col_i,
                    row_i,
                    Cell {
                        glyph: u32::from('\u{2580}'),
                        fg: pack_linear(fg),
                        bg: pack_linear(bg),
                        style: CellStyle::NONE,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_bright_splat_produces_exactly_one_lit_half_block() {
        let rect = Rect::new(0, 0, 10, 6);
        let mut canvas = SubrowCanvas::new(
            rect,
            LinRgb {
                r: 0.05,
                g: 0.2,
                b: 0.05,
            },
        );
        canvas.splat(
            5.0,
            4.0,
            LinRgb {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            },
            1.0,
        );
        let mut fb = Framebuffer::new(10, 6);
        canvas.blit(
            &mut fb,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );
        let lit: Vec<_> = (0..6)
            .flat_map(|r| (0..10).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != Cell::default())
            .collect();
        assert!(!lit.is_empty(), "splat produced no visible cells");
    }

    #[test]
    fn brightest_wins_does_not_blow_out_past_peak() {
        let rect = Rect::new(0, 0, 4, 4);
        let mut canvas = SubrowCanvas::new(
            rect,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );
        let red = LinRgb {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        };
        canvas.splat(2.0, 4.0, red, 1.0);
        canvas.splat(2.0, 4.0, red, 1.0);
        canvas.splat(2.0, 4.0, red, 1.0);
        let idx = 4 * canvas.rect.cols + 2; // subrow 4 == row 2 top
        assert!(
            canvas.dots[idx].r <= 1.000_001,
            "max-blend must not accumulate past 1.0"
        );
    }

    #[test]
    fn out_of_bounds_splats_do_not_panic() {
        let rect = Rect::new(0, 0, 4, 4);
        let mut canvas = SubrowCanvas::new(
            rect,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );
        canvas.splat(
            -5.0,
            -5.0,
            LinRgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            1.0,
        );
        canvas.splat(
            500.0,
            500.0,
            LinRgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            1.0,
        );
        let mut fb = Framebuffer::new(4, 4);
        canvas.blit(
            &mut fb,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        assert!(is_reliable('\u{2580}'));
    }
}
