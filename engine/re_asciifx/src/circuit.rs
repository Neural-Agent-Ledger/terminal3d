//! CIRCUIT BOARD / CHIPS — IC packages with pins and fake part numbers,
//! orthogonally-routed traces (box-drawing bends), data pulses travelling
//! along those traces at independent clock rates, blinking LEDs, one CPU die
//! with a heat shimmer, and a periodic coordinated "bus burst" across every
//! trace at once.
//!
//! ## What is baked at construction vs. animated at paint time
//!
//! Every chip's rect/label/pins and every trace's PATH (the sequence of
//! `(col, row, glyph)` box-drawing cells it occupies) are pure geometry,
//! resolved once by [`Circuit::random`] — none of that moves. What animates
//! is strictly per-trace: a single bright pulse travels the path length at
//! that trace's own `clock_hz` (so a fast trace's pulse laps a slow trace's
//! several times over), each chip's LED blinks at its own duty/frequency, and
//! the CPU die's fill brightness breathes at yet another independent rate —
//! "above all the frequencies", per the brief, spread across every moving
//! part rather than one shared clock.

use crate::hash::{hash_unit, range};
use crate::{put_str_clipped, secs, set_clipped, AsciiEffect, Rect};
use re_core::{Cell, Framebuffer};
use std::time::Duration;

/// Default chip count for [`Circuit::random`] when a caller does not size it
/// off the rect itself — kept for `random_with_count` callers, but
/// [`Circuit::random`] now uses `density_chip_count` so a wide/tall rect
/// gets more chips instead of 4 lonely ones in a sea of empty background
/// (denser routing reads better at large sizes).
pub const DEFAULT_CHIP_COUNT: usize = 4;
/// One chip roughly every this many cells of rect area — the density target
/// `density_chip_count` sizes toward, clamped to a sane range.
const CELLS_PER_CHIP: usize = 260;
const MAX_DENSITY_CHIPS: usize = 14;

const CHIP_W_RANGE: (usize, usize) = (9, 15);
const CHIP_H_RANGE: (usize, usize) = (4, 7);
const PIN_STRIDE_RANGE: (i64, i64) = (2, 4);
const LED_FREQ_RANGE: (f32, f32) = (0.4, 3.0);
const LED_DUTY_RANGE: (f32, f32) = (0.2, 0.6);
const TRACE_SPEED_RANGE: (f32, f32) = (6.0, 24.0); // cells / second
/// The travelling data pulse fades from full brightness at its head over
/// this many trailing cells — a comet, not a flat lit block.
const TRACE_PULSE_LEN: i64 = 7;
/// Probability a chip ALSO gets a cross-link trace to a chip other than its
/// immediate successor — extra routing on top of the main chain, purely to
/// fill the board densely the way real multi-chip PCBs are wired.
const P_CROSS_LINK: f32 = 0.55;
const CPU_BREATHE_FREQ_RANGE: (f32, f32) = (0.15, 0.6);
const BUS_PERIOD_RANGE: (f32, f32) = (3.0, 8.0);
const BUS_DURATION_RANGE: (f32, f32) = (0.15, 0.35);

const TRACE_COLOR: u32 = 0x0030_7048;
const TRACE_GLOW_BG: u32 = 0x0006_140A;
const TRACE_PULSE_COLOR: u32 = 0x0080_FFC0;
const BUS_FLASH_COLOR: u32 = 0x00C0_FFE0;
const CHIP_BORDER: u32 = 0x0070_8890;
const CHIP_FILL: u32 = 0x0008_1012;
const CHIP_LABEL: u32 = 0x0090_E0D0;
const LED_ON: u32 = 0x00FF_5030;
const LED_OFF: u32 = 0x0030_1008;
const CPU_BASE: u32 = 0x0018_2830;

const S_CHIP_W: u64 = 1;
const S_CHIP_H: u64 = 2;
const S_JITTER_X: u64 = 3;
const S_JITTER_Y: u64 = 4;
const S_LED_FREQ: u64 = 5;
const S_LED_DUTY: u64 = 6;
const S_LED_PHASE: u64 = 7;
const S_LABEL: u64 = 8;
const S_TRACE_SPEED: u64 = 9;
const S_TRACE_PHASE: u64 = 10;
const S_CPU_PICK: u64 = 11;
const S_CPU_FREQ: u64 = 12;
const S_CPU_PHASE: u64 = 13;
const S_BUS_PERIOD: u64 = 14;
const S_BUS_DURATION: u64 = 15;
const S_BUS_PHASE: u64 = 16;
const S_CROSS_LINK: u64 = 17;
const S_CROSS_TARGET: u64 = 18;

struct Chip {
    col0: i64,
    row0: i64,
    w: i64,
    h: i64,
    label: String,
    led_freq: f32,
    led_duty: f32,
    led_phase: f32,
}

impl Chip {
    fn center(&self) -> (i64, i64) {
        (self.col0 + self.w / 2, self.row0 + self.h / 2)
    }
}

struct Trace {
    path: Vec<(i64, i64, char)>,
    speed: f32,
    phase: f32,
}

/// A small procedural circuit board.
pub struct Circuit {
    chips: Vec<Chip>,
    traces: Vec<Trace>,
    cpu_index: usize,
    cpu_freq: f32,
    cpu_phase: f32,
    bus_period_s: f32,
    bus_duration_s: f32,
    bus_phase: f32,
}

/// A fake, deterministic part number: two letters, a hyphen, four hex-ish
/// characters. Never a real manufacturer scheme — just plausible silkscreen
/// filler, per the brief's "fake part numbers".
fn fake_part_number(seed: u64, idx: u64) -> String {
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ";
    const HEX: &[u8] = b"0123456789ABCDEF";
    let mut s = String::with_capacity(8);
    for slot in 0..2u64 {
        let i = (hash_unit(seed, idx, S_LABEL + slot) * LETTERS.len() as f64) as usize;
        s.push(LETTERS[i.min(LETTERS.len() - 1)] as char);
    }
    s.push('-');
    for slot in 2..6u64 {
        let i = (hash_unit(seed, idx, S_LABEL + slot) * HEX.len() as f64) as usize;
        s.push(HEX[i.min(HEX.len() - 1)] as char);
    }
    s
}

/// Extend `path` from its current last cell to `(to_c, to_r)` — a straight
/// axis-aligned run (caller guarantees exactly one of `to_c`/`to_r` differs
/// from the current last cell) — stepping one cell at a time so every
/// consecutive pair stays a unit step REGARDLESS of which direction the run
/// travels. Does not duplicate the current last cell.
fn extend_line(path: &mut Vec<(i64, i64, char)>, to_c: i64, to_r: i64, glyph: char) {
    let (mut c, mut r, _) = *path
        .last()
        .expect("extend_line needs a starting cell already in path");
    let sc = (to_c - c).signum();
    let sr = (to_r - r).signum();
    while c != to_c || r != to_r {
        c += sc;
        r += sr;
        path.push((c, r, glyph));
    }
}

/// The box-drawing glyph for a bend that connects a vertical run on its
/// `down` side (below the corner if `true`, above if `false`) to a
/// horizontal run on its `right` side (`true`) or left side (`false`).
fn corner_glyph(down: bool, right: bool) -> char {
    match (down, right) {
        (true, true) => '\u{250C}',   // ┌
        (true, false) => '\u{2510}',  // ┐
        (false, true) => '\u{2514}',  // └
        (false, false) => '\u{2518}', // ┘
    }
}

/// Build the orthogonal, single-bend path from `(c0, r0)` to `(c1, r1)`:
/// straight down (or up) from the source, across at the bend row, then
/// straight to the target, using box-drawing corners at the bends — correct
/// for EVERY sign of `c1 - c0` / `r1 - r0` (a chip can sit above, below,
/// left of, or right of the one before it once per-cell jitter is folded
/// in), not just "always downward, always rightward". Pure geometry.
fn route(c0: i64, r0: i64, c1: i64, r1: i64) -> Vec<(i64, i64, char)> {
    let mut path = vec![(c0, r0, if r0 == r1 { '\u{2500}' } else { '\u{2502}' })];
    if r0 == r1 {
        extend_line(&mut path, c1, r0, '\u{2500}');
        return path;
    }
    if c0 == c1 {
        extend_line(&mut path, c0, r1, '\u{2502}');
        return path;
    }
    let moving_down = r1 > r0;
    let moving_right = c1 > c0;
    let mid_row = r0 + (r1 - r0) / 2;

    extend_line(&mut path, c0, mid_row, '\u{2502}');
    if let Some(last) = path.last_mut() {
        last.2 = corner_glyph(!moving_down, moving_right);
    }
    extend_line(&mut path, c1, mid_row, '\u{2500}');
    if let Some(last) = path.last_mut() {
        last.2 = corner_glyph(moving_down, !moving_right);
    }
    extend_line(&mut path, c1, r1, '\u{2502}');
    path
}

/// How many chips a rect of this size should get, so a wide/tall backdrop
/// rect reads as a densely populated board rather than a handful of chips
/// adrift in empty space — one chip per `CELLS_PER_CHIP` cells of area,
/// clamped to `[DEFAULT_CHIP_COUNT, MAX_DENSITY_CHIPS]`.
#[must_use]
fn density_chip_count(rect: Rect) -> usize {
    let area = rect.cols * rect.rows;
    (area / CELLS_PER_CHIP).clamp(DEFAULT_CHIP_COUNT, MAX_DENSITY_CHIPS)
}

impl Circuit {
    /// Build a deterministic circuit-board layout for `rect` from `seed`,
    /// with a chip count scaled to the rect's area (see
    /// `density_chip_count`).
    #[must_use]
    pub fn random(seed: u64, rect: Rect) -> Self {
        Self::random_with_count(seed, rect, density_chip_count(rect))
    }

    /// As [`Self::random`], but with an explicit chip `count` instead of the
    /// area-scaled default.
    #[must_use]
    pub fn random_with_count(seed: u64, rect: Rect, count: usize) -> Self {
        let count = count.max(1);
        let cols_n = (count as f32).sqrt().ceil().max(1.0) as usize;
        let rows_n = count.div_ceil(cols_n).max(1);
        let cell_w = (rect.cols as f32 / cols_n as f32).max(4.0);
        let cell_h = (rect.rows as f32 / rows_n as f32).max(4.0);

        let mut chips = Vec::with_capacity(count);
        for i in 0..count {
            let gx = (i % cols_n) as f32;
            let gy = (i / cols_n) as f32;
            let w = (CHIP_W_RANGE.0 as f32
                + hash_unit(seed, i as u64, S_CHIP_W) as f32
                    * (CHIP_W_RANGE.1 - CHIP_W_RANGE.0) as f32)
                .min(cell_w - 1.0)
                .max(5.0) as i64;
            let h = (CHIP_H_RANGE.0 as f32
                + hash_unit(seed, i as u64, S_CHIP_H) as f32
                    * (CHIP_H_RANGE.1 - CHIP_H_RANGE.0) as f32)
                .min(cell_h - 1.0)
                .max(3.0) as i64;
            let jitter_room_x = (cell_w - w as f32).max(0.0);
            let jitter_room_y = (cell_h - h as f32).max(0.0);
            let jx = hash_unit(seed, i as u64, S_JITTER_X) as f32 * jitter_room_x;
            let jy = hash_unit(seed, i as u64, S_JITTER_Y) as f32 * jitter_room_y;
            let col0 = rect.col0 as i64 + (gx * cell_w + jx) as i64;
            let row0 = rect.row0 as i64 + (gy * cell_h + jy) as i64;
            chips.push(Chip {
                col0,
                row0,
                w,
                h,
                label: fake_part_number(seed, i as u64),
                led_freq: range(
                    seed,
                    i as u64,
                    S_LED_FREQ,
                    LED_FREQ_RANGE.0,
                    LED_FREQ_RANGE.1,
                ),
                led_duty: range(
                    seed,
                    i as u64,
                    S_LED_DUTY,
                    LED_DUTY_RANGE.0,
                    LED_DUTY_RANGE.1,
                ),
                led_phase: range(seed, i as u64, S_LED_PHASE, 0.0, 1.0),
            });
        }

        let mut traces = Vec::new();
        for i in 0..chips.len().saturating_sub(1) {
            let (c0, _) = chips[i].center();
            let (c1, _) = chips[i + 1].center();
            let path = route(c0, chips[i].row0 + chips[i].h, c1, chips[i + 1].row0);
            if path.is_empty() {
                continue;
            }
            traces.push(Trace {
                path,
                speed: range(
                    seed,
                    i as u64,
                    S_TRACE_SPEED,
                    TRACE_SPEED_RANGE.0,
                    TRACE_SPEED_RANGE.1,
                ),
                phase: range(seed, i as u64, S_TRACE_PHASE, 0.0, 1.0),
            });
        }

        // Cross-link traces: extra routing between NON-consecutive chips, on
        // top of the main daisy-chain, so a bigger board reads as densely
        // wired rather than one thin chain snaking through empty space.
        if chips.len() > 2 {
            for i in 0..chips.len() {
                if !crate::hash::chance(seed, i as u64, S_CROSS_LINK, P_CROSS_LINK) {
                    continue;
                }
                let mut j = (hash_unit(seed, i as u64, S_CROSS_TARGET) * chips.len() as f64)
                    as usize
                    % chips.len();
                if j == i || j == (i + 1) % chips.len() {
                    j = (j + 2) % chips.len();
                }
                if j == i {
                    continue;
                }
                let (c0, _) = chips[i].center();
                let (c1, _) = chips[j].center();
                // Same edge convention as the main chain (exit the source's
                // bottom, enter the target's top) so a cross-link's path
                // starts/ends at the chip's BORDER, not buried mid-box — the
                // chip is drawn over the trace afterward either way, but this
                // keeps the visible stub outside the box a sensible short run
                // rather than an arbitrary center-to-center line.
                let path = route(c0, chips[i].row0 + chips[i].h, c1, chips[j].row0);
                if path.len() < 2 {
                    continue;
                }
                traces.push(Trace {
                    path,
                    speed: range(
                        seed,
                        i as u64,
                        S_TRACE_SPEED + 100,
                        TRACE_SPEED_RANGE.0,
                        TRACE_SPEED_RANGE.1,
                    ),
                    phase: range(seed, i as u64, S_TRACE_PHASE + 100, 0.0, 1.0),
                });
            }
        }

        let cpu_index = if chips.is_empty() {
            0
        } else {
            (hash_unit(seed, 0, S_CPU_PICK) * chips.len() as f64) as usize % chips.len()
        };

        Circuit {
            chips,
            traces,
            cpu_index,
            cpu_freq: range(
                seed,
                0,
                S_CPU_FREQ,
                CPU_BREATHE_FREQ_RANGE.0,
                CPU_BREATHE_FREQ_RANGE.1,
            ),
            cpu_phase: range(seed, 0, S_CPU_PHASE, 0.0, std::f32::consts::TAU),
            bus_period_s: range(
                seed,
                0,
                S_BUS_PERIOD,
                BUS_PERIOD_RANGE.0,
                BUS_PERIOD_RANGE.1,
            ),
            bus_duration_s: range(
                seed,
                0,
                S_BUS_DURATION,
                BUS_DURATION_RANGE.0,
                BUS_DURATION_RANGE.1,
            ),
            bus_phase: range(seed, 0, S_BUS_PHASE, 0.0, 1.0),
        }
    }

    /// Number of chips this layout placed.
    #[must_use]
    pub fn chip_count(&self) -> usize {
        self.chips.len()
    }

    /// Number of traces routed between chips.
    #[must_use]
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// `true` iff `t` falls inside a coordinated "bus burst" — every trace
    /// flashes together, not just its own travelling pulse.
    #[must_use]
    pub fn in_bus_burst(&self, t: Duration) -> bool {
        let period = self.bus_period_s.max(0.1);
        let phase = (secs(t) / period + self.bus_phase).rem_euclid(1.0);
        phase < (self.bus_duration_s / period).clamp(0.0, 1.0)
    }

    fn led_on(chip: &Chip, t: f32) -> bool {
        let phase = (t * chip.led_freq + chip.led_phase).rem_euclid(1.0);
        phase < chip.led_duty
    }
}

impl AsciiEffect for Circuit {
    fn name(&self) -> &'static str {
        "circuit"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let bus_burst = self.in_bus_burst(t);

        for trace in &self.traces {
            let len = trace.path.len().max(1) as f32;
            let pulse_pos = ((secs_t * trace.speed + trace.phase * len).rem_euclid(len)) as i64;
            for (idx, &(col, row, glyph)) in trace.path.iter().enumerate() {
                // Distance BEHIND the pulse head (0 = the head itself), so the
                // comet fades smoothly over its trailing `TRACE_PULSE_LEN`
                // cells instead of a flat block of uniform brightness.
                let d = (pulse_pos - idx as i64).rem_euclid(trace.path.len() as i64);
                let comet_k = if d < TRACE_PULSE_LEN {
                    1.0 - d as f32 / TRACE_PULSE_LEN as f32
                } else {
                    0.0
                };
                let color = if bus_burst {
                    BUS_FLASH_COLOR
                } else if comet_k > 0.0 {
                    lerp_color(TRACE_COLOR, TRACE_PULSE_COLOR, comet_k)
                } else {
                    TRACE_COLOR
                };
                let bg = if bus_burst || comet_k > 0.3 {
                    scale_color(TRACE_GLOW_BG, 1.0 + comet_k * 3.0)
                } else {
                    TRACE_GLOW_BG
                };
                set_clipped(
                    fb,
                    rect,
                    col,
                    row,
                    Cell {
                        glyph: u32::from(glyph),
                        fg: color,
                        bg,
                        ..Default::default()
                    },
                );
            }
        }

        for (i, chip) in self.chips.iter().enumerate() {
            let is_cpu = i == self.cpu_index;
            let fill = if is_cpu {
                let k = 1.0
                    + 0.35
                        * (std::f32::consts::TAU * self.cpu_freq * secs_t + self.cpu_phase).sin();
                scale_color(CPU_BASE, k.max(0.3))
            } else {
                CHIP_FILL
            };
            draw_chip_box(fb, rect, chip, fill);

            let label_row = chip.row0 + chip.h / 2;
            let label_col = chip.col0 + (chip.w - chip.label.chars().count() as i64) / 2;
            put_str_clipped(
                fb,
                rect,
                label_col,
                label_row,
                &chip.label,
                CHIP_LABEL,
                fill,
            );

            let led_col = chip.col0 + 1;
            let led_row = chip.row0 + 1;
            let on = Self::led_on(chip, secs_t);
            set_clipped(
                fb,
                rect,
                led_col,
                led_row,
                Cell {
                    glyph: u32::from(if on { '\u{25CF}' } else { '\u{25CB}' }),
                    fg: if on { LED_ON } else { LED_OFF },
                    bg: fill,
                    ..Default::default()
                },
            );
        }
    }
}

fn scale_color(c: u32, k: f32) -> u32 {
    let l = re_core::oklab::unpack_linear(c);
    re_core::oklab::pack_linear(re_core::oklab::LinRgb {
        r: (l.r * k).clamp(0.0, 4.0),
        g: (l.g * k).clamp(0.0, 4.0),
        b: (l.b * k).clamp(0.0, 4.0),
    })
}

/// Linear-space blend from `a` to `b` by `k` in `[0,1]` — the comet pulse's
/// fade from the dim base trace colour up to the bright pulse colour.
fn lerp_color(a: u32, b: u32, k: f32) -> u32 {
    let la = re_core::oklab::unpack_linear(a);
    let lb = re_core::oklab::unpack_linear(b);
    re_core::oklab::pack_linear(re_core::oklab::lerp_linear(la, lb, k.clamp(0.0, 1.0)))
}

fn draw_chip_box(fb: &mut Framebuffer, rect: Rect, chip: &Chip, fill: u32) {
    for row in chip.row0..(chip.row0 + chip.h) {
        for col in chip.col0..(chip.col0 + chip.w) {
            let top = row == chip.row0;
            let bottom = row == chip.row0 + chip.h - 1;
            let left = col == chip.col0;
            let right = col == chip.col0 + chip.w - 1;
            let glyph = match (top, bottom, left, right) {
                (true, _, true, _) => '\u{250C}',
                (true, _, _, true) => '\u{2510}',
                (_, true, true, _) => '\u{2514}',
                (_, true, _, true) => '\u{2518}',
                (true, false, false, false) => pin_or('\u{2500}', '\u{252C}', chip, col),
                (false, true, false, false) => pin_or('\u{2500}', '\u{2534}', chip, col),
                (false, false, true, false) | (false, false, false, true) => '\u{2502}',
                _ => ' ',
            };
            set_clipped(
                fb,
                rect,
                col,
                row,
                Cell {
                    glyph: u32::from(glyph),
                    fg: CHIP_BORDER,
                    bg: fill,
                    ..Default::default()
                },
            );
        }
    }
}

/// Every `PIN_STRIDE_RANGE`-th column along the top/bottom edge gets a pin
/// tick (`┬`/`┴`) instead of a plain border rule — deterministic in the
/// chip's own geometry (not randomized further; the chip's position/size
/// already vary per seed, which is enough texture for a silkscreen pin row).
fn pin_or(plain: char, pin: char, chip: &Chip, col: i64) -> char {
    let stride = PIN_STRIDE_RANGE.0 + (chip.w % (PIN_STRIDE_RANGE.1 - PIN_STRIDE_RANGE.0 + 1));
    let offset = col - chip.col0;
    if stride > 0 && offset % stride == 1 {
        pin
    } else {
        plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 100,
        rows: 40,
    };

    #[test]
    fn determinism_same_seed_same_board() {
        let a = Circuit::random(11, RECT);
        let b = Circuit::random(11, RECT);
        assert_eq!(a.chip_count(), b.chip_count());
        for (ca, cb) in a.chips.iter().zip(b.chips.iter()) {
            assert_eq!(ca.label, cb.label);
            assert_eq!(
                (ca.col0, ca.row0, ca.w, ca.h),
                (cb.col0, cb.row0, cb.w, cb.h)
            );
        }
    }

    #[test]
    fn different_seeds_diverge_in_frequencies() {
        let a = Circuit::random(1, RECT);
        let b = Circuit::random(2, RECT);
        assert_ne!(a.cpu_freq, b.cpu_freq);
        assert_ne!(a.bus_period_s, b.bus_period_s);
        assert!(a
            .chips
            .iter()
            .zip(b.chips.iter())
            .any(|(x, y)| x.led_freq != y.led_freq));
        assert!(a
            .traces
            .iter()
            .zip(b.traces.iter())
            .any(|(x, y)| x.speed != y.speed));
    }

    #[test]
    fn part_numbers_look_plausible_and_are_reliable_glyphs() {
        use re_render::glyph_family::is_reliable;
        let c = Circuit::random(3, RECT);
        for chip in &c.chips {
            assert_eq!(chip.label.chars().count(), 7, "AA-HHHH shape");
            assert!(chip.label.chars().all(is_reliable));
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in [
            '\u{2500}', '\u{2502}', '\u{250C}', '\u{2510}', '\u{2514}', '\u{2518}', '\u{252C}',
            '\u{2534}', '\u{25CF}', '\u{25CB}', ' ',
        ] {
            assert!(
                is_reliable(g),
                "circuit glyph {g:?} is not in the reliable set"
            );
        }
    }

    #[test]
    fn traces_route_orthogonally_between_consecutive_chips() {
        let c = Circuit::random(5, RECT);
        // At least the main daisy-chain, possibly more from cross-links —
        // never fewer.
        assert!(c.trace_count() >= c.chip_count().saturating_sub(1));
        for trace in &c.traces {
            for w in trace.path.windows(2) {
                let (c0, r0, _) = w[0];
                let (c1, r1, _) = w[1];
                let dc = (c1 - c0).abs();
                let dr = (r1 - r0).abs();
                assert!(dc + dr == 1, "trace step not axis-aligned/unit: {w:?}");
            }
        }
    }

    #[test]
    fn bus_burst_is_periodic_and_bounded() {
        let c = Circuit::random(4, RECT);
        let mut any_in = false;
        let mut any_out = false;
        for ms in (0..15000u64).step_by(97) {
            if c.in_bus_burst(Duration::from_millis(ms)) {
                any_in = true;
            } else {
                any_out = true;
            }
        }
        assert!(any_in && any_out);
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(110, 44);
        let rect = Rect::new(3, 2, 100, 40);
        let circuit = Circuit::random(7, rect);
        for ms in [0u64, 500, 3000, 9000] {
            circuit.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..44 {
            for col in 0..110 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }
}
