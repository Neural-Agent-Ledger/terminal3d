//! `intro_tear` — Phase 1 of the OPENING two-phase boot sequence
//! (2026-09-24): "tear" the player's REAL, pre-existing terminal text on the
//! MAIN screen, using only blind VT editing sequences, before the alternate-
//! screen TUI (`bevy_ratatui`'s `RatatuiPlugins`) ever takes over.
//!
//! We cannot READ the screen (there is no portable "what is on screen" query),
//! but we CAN deform it blindly: delete/insert-char shears a row sideways,
//! insert/delete-line drops or lifts rows, a scroll region slides a whole
//! band, and erase-char opens holes we sprinkle with shade glyphs and a few
//! coloured "bit-rot" characters. The glyph vocabulary (`░▒▓·:`) matches the
//! menu FX in `re_asciifx`, so the tear reads as the same visual language.
//!
//! ## Two layers, on purpose
//!
//! * [`generate_tear_ops`] + [`render_ops`] are PURE — a seed + a step index
//!   in, a `Vec<u8>` of VT bytes out. No stdout, no clock, no terminal. This is
//!   what the unit tests exercise: determinism, bounds, and "only the allowed
//!   sequences ever appear".
//! * [`run_screen_tear`] is the one side-effecting driver: it enables raw mode
//!   (needed to poll individual keypresses without waiting for Enter), loops
//!   the pure generator at `cfg.fps`, writes the bytes straight to stdout, and
//!   disables raw mode again before returning — handing off to
//!   `bevy_ratatui`'s own `RatatuiPlugins` Startup, which enables raw mode +
//!   the alternate screen itself. The enable/disable pair is symmetric and
//!   touches no alternate-screen state, so it cannot break that later init.
//!
//! ## Gating
//!
//! [`should_run_tear`] is the pure decision function `main.rs` calls with the
//! `Settings::boot_intro` flag, the `TS_NO_INTRO` env override, and a live
//! TTY + VT-capability probe ([`tty_and_vt_capable`]). Any `false` and
//! `run_screen_tear` is never called at all — no bytes are ever written to a
//! piped/redirected stdout, a `--dump` run, or a `cargo test` process.
//!
//! ## Exit: clearing the main screen
//!
//! [`finalize_terminal_teardown`] forces the terminal back to the main
//! screen, cooked mode, and CLEARS it — called both after a normal
//! `app.run()` return (belt-and-suspenders: `bevy_ratatui`'s own
//! `RatatuiContext::Drop` has usually already left the alt screen by then)
//! and from the crate's panic hook (`terminal.rs`), where it runs FIRST,
//! before any `World`/resource `Drop` — see that call site's comment for why
//! resource-drop ORDER cannot be relied on here. Every operation it performs
//! is idempotent, so calling it twice (normal path) or from a state that
//! never entered the alt screen at all (tear disabled) is always safe.

use std::io::Write as _;
use std::time::Duration;

// ─────────────────────────── deterministic hashing ───────────────────────────
//
// Same splitmix64-style mix `re_game::menu_fx::overgrowth` uses: a pure
// function of its integer inputs, never a wall clock or thread-local RNG.

fn hash64(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

// ────────────────────────────────── config ───────────────────────────────────

/// Tuning for one tear run. `Default` sits inside the approved "~1.5-2.5 s,
/// escalating, ~30 fps" window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TearConfig {
    /// Deterministic seed. `main.rs` derives a fresh one per real launch
    /// (wall-clock entropy is fine THERE — every function below is still a
    /// pure, testable function of whatever seed it is given).
    pub seed: u64,
    /// Total wall-clock duration of the tear.
    pub duration: Duration,
    /// Steps per second.
    pub fps: u32,
}

impl Default for TearConfig {
    fn default() -> Self {
        TearConfig {
            seed: 0,
            duration: Duration::from_millis(2000),
            fps: 30,
        }
    }
}

/// A handful of named 16-colour SGR codes for the "bit-rot" sprinkle —
/// portable on every VT terminal with no truecolor/256 probe needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TearColor {
    /// No SGR at all — whatever the terminal's current default is.
    None,
    /// `CSI 31 m`.
    Red,
    /// `CSI 32 m`.
    Green,
    /// `CSI 33 m`.
    Yellow,
    /// `CSI 35 m`.
    Magenta,
    /// `CSI 36 m`.
    Cyan,
}

impl TearColor {
    /// The `CSI 3<n> m` sequence for this colour, or `""` for `None`.
    fn sgr(self) -> &'static str {
        match self {
            TearColor::None => "",
            TearColor::Red => "\x1b[31m",
            TearColor::Green => "\x1b[32m",
            TearColor::Yellow => "\x1b[33m",
            TearColor::Magenta => "\x1b[35m",
            TearColor::Cyan => "\x1b[36m",
        }
    }

    fn from_index(i: u64) -> TearColor {
        match i % 5 {
            0 => TearColor::Red,
            1 => TearColor::Green,
            2 => TearColor::Yellow,
            3 => TearColor::Magenta,
            _ => TearColor::Cyan,
        }
    }
}

/// The shade-glyph vocabulary sprinkled into erased holes — the SAME reliable
/// "bit-rot" glyph set used elsewhere (Block Elements + the vetted `·`/`:`),
/// so this stays consistent with the rest of the game's corruption vocabulary
/// even though it never
/// reaches `re_render`'s glyph-reliability gate (that gate is for the LIVE
/// framebuffer; this writes the user's real terminal, before the game's own
/// renderer exists).
pub const SHADE_GLYPHS: [char; 5] = ['░', '▒', '▓', '·', ':'];

/// One deformation step: a single VT editing action, already bounded to a
/// real `cols × rows` grid by [`generate_tear_ops`]. `row`/`col` are 1-based
/// (VT convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TearOp {
    /// `CSI row;col H` then `CSI n P` — delete `n` chars at `(row, col)`,
    /// shearing the rest of the row left.
    DeleteChars {
        /// 1-based row.
        row: u16,
        /// 1-based column.
        col: u16,
        /// Number of characters to delete.
        n: u16,
    },
    /// `CSI row;col H` then `CSI n @` — insert `n` blanks at `(row, col)`,
    /// pushing the rest of the row right.
    InsertChars {
        /// 1-based row.
        row: u16,
        /// 1-based column.
        col: u16,
        /// Number of blanks to insert.
        n: u16,
    },
    /// `CSI row;1 H` then `CSI n L` — insert `n` blank lines at `row` (the
    /// rows below drop down).
    InsertLines {
        /// 1-based row.
        row: u16,
        /// Number of lines to insert.
        n: u16,
    },
    /// `CSI row;1 H` then `CSI n M` — delete `n` lines at `row` (the rows
    /// below rise up, collapsing into the gap).
    DeleteLines {
        /// 1-based row.
        row: u16,
        /// Number of lines to delete.
        n: u16,
    },
    /// `CSI row;col H` then `CSI n X` — erase `n` chars at `(row, col)` to
    /// blank, opening a hole (no shift). The one ERASURE op — deliberately
    /// the rarest kind and gated to the tail of the run (see
    /// [`generate_tear_ops`]'s phase weighting): the brief must read as
    /// TEARING (things move, shear, slide) with erasure only as the very
    /// last thing that happens to a cell, not the dominant look throughout.
    EchHole {
        /// 1-based row.
        row: u16,
        /// 1-based column.
        col: u16,
        /// Number of characters to erase.
        n: u16,
    },
    /// `CSI row;col H`, an optional colour, then ONE literal glyph byte — a
    /// single-cell "bit-rot" sprinkle. Never more than one cell so it can
    /// never line-wrap into the next row. [`generate_tear_ops`] emits these
    /// in adjacent CLUSTERS (several consecutive `Sprinkle`s sharing a row,
    /// neighbouring columns) rather than one at a time, so they read as a
    /// visible patch of corruption instead of a single stray pixel.
    Sprinkle {
        /// 1-based row.
        row: u16,
        /// 1-based column.
        col: u16,
        /// The single glyph written into this cell.
        glyph: char,
        /// SGR colour applied before writing the glyph.
        color: TearColor,
    },
    /// `CSI top;bot r`, then `CSI n S` (up) or `CSI n T` (down), then `CSI r`
    /// to restore the full-screen scroll region — slides the `top..=bot` band.
    /// The "block visibly slides" op: content moves, it does not vanish.
    ScrollRegion {
        /// 1-based first row of the scroll region.
        top: u16,
        /// 1-based last row of the scroll region.
        bot: u16,
        /// `true` scrolls up (`CSI n S`), `false` scrolls down (`CSI n T`).
        up: bool,
        /// Number of lines to scroll.
        n: u16,
    },
    /// A STAIRCASE shear: `rows` consecutive lines starting at `top`, each
    /// sheared by `CSI row;col H` + `CSI n P` (delete, drags left) or
    /// `CSI n @` (insert, drags right) — the SAME vocabulary as
    /// [`TearOp::DeleteChars`]/[`TearOp::InsertChars`], just applied down a
    /// whole band with `n` walking by `step_n` per row so consecutive rows
    /// visibly offset against each other (line halves sliding against each
    /// other, a staircase silhouette down the band) instead of one lone row
    /// twitching. `insert == true` selects the `@` (push-right) direction.
    ShearBand {
        /// 1-based first row of the band.
        top: u16,
        /// Number of consecutive rows the shear spans.
        band_rows: u16,
        /// 1-based starting column for the first row's shear.
        col: u16,
        /// Shear amount (chars) for the first row.
        n: u16,
        /// Additional shear amount added per subsequent row.
        step_n: u16,
        /// `true` inserts (push-right, `@`); `false` deletes (drag-left, `P`).
        insert: bool,
    },
    /// A one-frame "hiccup": `CSI row;1 H` + `CSI n L` (insert — the band
    /// drops) immediately followed by `CSI row;1 H` + `CSI n M` (delete — it
    /// rises back), same vocabulary as [`TearOp::InsertLines`]/
    /// [`TearOp::DeleteLines`] fired back to back at the same row. Reads as
    /// the row band visibly stuttering/jumping in place rather than a clean
    /// single shift.
    Stutter {
        /// 1-based row.
        row: u16,
        /// Number of lines inserted then immediately deleted.
        n: u16,
    },
}

/// How many discrete BURST events happen across the whole tear. Sparse and
/// RHYTHMIC, deliberately: every step that is not one of these `BURST_COUNT`
/// indices emits NOTHING at all, so most of the run is a calm hold at
/// whatever the last burst left on screen (a VT edit is a lasting change, not
/// an animation, so "nothing new this step" still reads as "the tear
/// happened, then paused" rather than "the tear stopped"). This replaces an
/// earlier design that drew on every single frame and escalated
/// continuously — read in practice as constant noise rather than "a few
/// clear events that build up, calm pauses between bursts".
const BURST_COUNT: u32 = 6;

/// Ops per burst at its sparsest (the very first burst) and densest (the
/// final burst — which is also the ONLY burst where [`KIND_ECH_HOLE`], the
/// one ERASURE op, becomes reachable at all: "erasure is the minority op,
/// reserved for the end"). Small numbers on purpose — "a few clear events",
/// not a glitch storm.
const BURST_MIN_OPS: u32 = 1;
const BURST_MAX_OPS: u32 = 3;

const SALT_ROW: u64 = 1;
const SALT_COL: u64 = 2;
const SALT_N: u64 = 3;
const SALT_KIND: u64 = 4;
const SALT_GLYPH: u64 = 5;
const SALT_COLOR: u64 = 6;
const SALT_TOP: u64 = 7;
const SALT_BOT: u64 = 8;
const SALT_STEP_N: u64 = 10;
const SALT_DIR: u64 = 11;

/// Which burst (if any) owns `step`: the `BURST_COUNT` steps are spread
/// evenly across the MIDDLE of the run (never the very first or very last
/// step, so the tear neither opens nor closes mid-burst), and every other
/// step belongs to no burst at all. Pure function of `(step, total_steps)` —
/// same inputs, same answer, forever.
fn burst_index_for_step(step: u32, total_steps: u32) -> Option<u32> {
    let total_steps = u64::from(total_steps.max(1));
    for i in 0..BURST_COUNT {
        let target = (u64::from(i) + 1) * total_steps / (u64::from(BURST_COUNT) + 1);
        if u64::from(step) == target {
            return Some(i);
        }
    }
    None
}

/// Generate the deterministic ops for one step. Pure: the same
/// `(seed, cols, rows, step, total_steps)` always produces the same
/// `Vec<TearOp>`, on any machine, forever. Every op is bounded to
/// `1..=cols` / `1..=rows` — never off-grid, never zero-length, never able to
/// wrap a [`TearOp::Sprinkle`] onto the next row.
///
/// EMPTY on every step except the `BURST_COUNT` burst steps
/// `burst_index_for_step` picks out — the tear is a handful of sparse,
/// rhythmic events with calm holds between them, not a continuous escalating
/// glitch storm. A later burst draws mildly more ops than an earlier one
/// (`BURST_MIN_OPS..=BURST_MAX_OPS`), so the run still reads as "building up"
/// without ever being dense.
///
/// Also empty when the grid itself is degenerate (`cols == 0 || rows == 0`) —
/// the caller (`run_screen_tear`) never calls this at all when the tear is
/// disabled, so "disabled" and "degenerate grid" are the two OTHER ways this
/// function can be observed to emit nothing.
pub fn generate_tear_ops(
    seed: u64,
    cols: u16,
    rows: u16,
    step: u32,
    total_steps: u32,
) -> Vec<TearOp> {
    if cols == 0 || rows == 0 {
        return Vec::new();
    }
    let total_steps = total_steps.max(1);
    let Some(burst_idx) = burst_index_for_step(step, total_steps) else {
        return Vec::new();
    };
    let burst_frac = if BURST_COUNT <= 1 {
        1.0
    } else {
        f64::from(burst_idx) / f64::from(BURST_COUNT - 1)
    };
    let op_count =
        BURST_MIN_OPS + (burst_frac * f64::from(BURST_MAX_OPS - BURST_MIN_OPS)).round() as u32;
    let final_burst = burst_idx + 1 == BURST_COUNT;

    // A too-small grid makes ScrollRegion/ShearBand unsafe to place at all
    // (there is no band to slide/shear) — their weight drops to 0 below and
    // DeleteChars/InsertChars/Stutter (always safe for any cols>=1, rows>=1)
    // absorb the share, the same fallback discipline the original uniform
    // pick used for ScrollRegion alone.
    let scroll_region_ok = rows >= 4;
    let shear_band_ok = rows >= 3 && cols >= 2;

    // Kind-selection weights favour things that visibly MOVE (shear, stutter,
    // sliding blocks) over erasure or noise. `Sprinkle` ("glyph flips") is
    // deliberately a MINORITY weight in every burst — "sprinkle glyphs only
    // sparingly" — and `EchHole` (the one ERASURE op) is gated to ZERO until
    // the FINAL burst, where it becomes reachable but still stays a minority
    // — "erasure is the minority op, reserved for the end". Index == the kind
    // tag matched below.
    const KIND_DELETE_CHARS: u8 = 0;
    const KIND_INSERT_CHARS: u8 = 1;
    const KIND_INSERT_LINES: u8 = 2;
    const KIND_DELETE_LINES: u8 = 3;
    const KIND_ECH_HOLE: u8 = 4;
    const KIND_SPRINKLE: u8 = 5;
    const KIND_SCROLL_REGION: u8 = 6;
    const KIND_SHEAR_BAND: u8 = 7;
    const KIND_STUTTER: u8 = 8;
    let weights: [(u8, u32); 9] = if final_burst {
        [
            (KIND_DELETE_CHARS, 1),
            (KIND_INSERT_CHARS, 1),
            (KIND_INSERT_LINES, 1),
            (KIND_DELETE_LINES, 1),
            (KIND_ECH_HOLE, 2),
            (KIND_SPRINKLE, 2),
            (KIND_SCROLL_REGION, if scroll_region_ok { 2 } else { 0 }),
            (KIND_SHEAR_BAND, if shear_band_ok { 3 } else { 0 }),
            (KIND_STUTTER, 2),
        ]
    } else {
        [
            (KIND_DELETE_CHARS, 2),
            (KIND_INSERT_CHARS, 2),
            (KIND_INSERT_LINES, 1),
            (KIND_DELETE_LINES, 1),
            (KIND_ECH_HOLE, 0),
            (KIND_SPRINKLE, 1),
            (KIND_SCROLL_REGION, if scroll_region_ok { 3 } else { 0 }),
            (KIND_SHEAR_BAND, if shear_band_ok { 4 } else { 0 }),
            (KIND_STUTTER, 3),
        ]
    };
    let total_weight: u32 = weights.iter().map(|(_, w)| *w).sum::<u32>().max(1);

    let mut ops = Vec::with_capacity(op_count as usize);
    for i in 0..op_count {
        let base = u64::from(step) * 1_000 + u64::from(i);
        let h = |salt: u64| hash64(seed, base, salt);

        let row = 1 + (h(SALT_ROW) % u64::from(rows)) as u16;
        let pick = (h(SALT_KIND) % u64::from(total_weight)) as u32;
        let mut acc = 0u32;
        let mut kind = KIND_DELETE_CHARS;
        for &(k, w) in &weights {
            acc += w;
            if pick < acc {
                kind = k;
                break;
            }
        }

        match kind {
            KIND_INSERT_CHARS => {
                let col = 1 + (h(SALT_COL) % u64::from(cols)) as u16;
                let max_n = cols.saturating_sub(col - 1).max(1);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::InsertChars { row, col, n });
            }
            KIND_INSERT_LINES => {
                let max_n = rows.saturating_sub(row - 1).clamp(1, 3);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::InsertLines { row, n });
            }
            KIND_DELETE_LINES => {
                let max_n = rows.saturating_sub(row - 1).clamp(1, 3);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::DeleteLines { row, n });
            }
            KIND_ECH_HOLE => {
                let col = 1 + (h(SALT_COL) % u64::from(cols)) as u16;
                let max_n = cols.saturating_sub(col - 1).clamp(1, 8);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::EchHole { row, col, n });
            }
            KIND_SPRINKLE => {
                // ALWAYS a single glyph — "sprinkle glyphs only sparingly",
                // never a cluster: one stray bit-rot character, not a patch.
                let col = 1 + (h(SALT_COL) % u64::from(cols)) as u16;
                let gi = (h(SALT_GLYPH) % SHADE_GLYPHS.len() as u64) as usize;
                let color = TearColor::from_index(h(SALT_COLOR));
                ops.push(TearOp::Sprinkle {
                    row,
                    col,
                    glyph: SHADE_GLYPHS[gi],
                    color,
                });
            }
            KIND_SCROLL_REGION => {
                // only reachable when scroll_region_ok (rows >= 4).
                let top = 1 + (h(SALT_TOP) % u64::from(rows.saturating_sub(3))) as u16;
                let span = rows.saturating_sub(top).max(2).min(rows - top + 1);
                let bot_step = (h(SALT_BOT) % u64::from(span.saturating_sub(1).max(1))) as u16;
                let bot = (top + 1 + bot_step).min(rows);
                // `n` is a SLIDE, not a wipe: capped to a small constant
                // regardless of how tall the region is (the same discipline
                // InsertLines/DeleteLines/Stutter already use for their own
                // `n`). An `n` anywhere near the region's full span would
                // scroll the ENTIRE band's content out of the terminal's
                // buffer in one call — the region reads as erased, not
                // slid, which is exactly the "text mostly just disappears"
                // failure this op exists to avoid.
                let max_n = bot.saturating_sub(top).clamp(1, 3);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                let up = h(SALT_COLOR) % 2 == 0;
                ops.push(TearOp::ScrollRegion { top, bot, up, n });
            }
            KIND_SHEAR_BAND => {
                // only reachable when shear_band_ok (rows >= 3, cols >= 2).
                let band_span_cap = rows.clamp(2, 5);
                let top_max = rows.saturating_sub(band_span_cap - 1).max(1);
                let top = 1 + (h(SALT_TOP) % u64::from(top_max)) as u16;
                let band_rows = band_span_cap.min(rows - top + 1);
                let col = 1 + (h(SALT_COL) % u64::from(cols.saturating_sub(1).max(1))) as u16;
                let remaining = cols.saturating_sub(col - 1).max(1);
                let base_n = 1 + (h(SALT_N) % u64::from(remaining)) as u16;
                let max_growth_total = remaining.saturating_sub(base_n);
                let step_n = if band_rows > 1 && max_growth_total > 0 {
                    let cap = max_growth_total / (band_rows - 1);
                    if cap == 0 {
                        0
                    } else {
                        1 + (h(SALT_STEP_N) % u64::from(cap)) as u16
                    }
                } else {
                    0
                };
                let insert = h(SALT_DIR) % 2 == 0;
                ops.push(TearOp::ShearBand {
                    top,
                    band_rows,
                    col,
                    n: base_n,
                    step_n,
                    insert,
                });
            }
            KIND_STUTTER => {
                let max_n = rows.saturating_sub(row - 1).clamp(1, 3);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::Stutter { row, n });
            }
            _ => {
                // KIND_DELETE_CHARS, and the fallback for anything unmatched
                // above — always safe for any cols >= 1.
                let col = 1 + (h(SALT_COL) % u64::from(cols)) as u16;
                let max_n = cols.saturating_sub(col - 1).max(1);
                let n = 1 + (h(SALT_N) % u64::from(max_n)) as u16;
                ops.push(TearOp::DeleteChars { row, col, n });
            }
        }
    }
    ops
}

/// Render ops to their raw VT byte sequence. Pure (no I/O) so it is directly
/// unit-testable — including the "only the allowed sequences ever appear"
/// invariant, which parses this exact output.
pub fn render_ops(ops: &[TearOp]) -> Vec<u8> {
    let mut out = Vec::new();
    for op in ops {
        match *op {
            TearOp::DeleteChars { row, col, n } => {
                let _ = write!(out, "\x1b[{row};{col}H\x1b[{n}P");
            }
            TearOp::InsertChars { row, col, n } => {
                let _ = write!(out, "\x1b[{row};{col}H\x1b[{n}@");
            }
            TearOp::InsertLines { row, n } => {
                let _ = write!(out, "\x1b[{row};1H\x1b[{n}L");
            }
            TearOp::DeleteLines { row, n } => {
                let _ = write!(out, "\x1b[{row};1H\x1b[{n}M");
            }
            TearOp::EchHole { row, col, n } => {
                let _ = write!(out, "\x1b[{row};{col}H\x1b[{n}X");
            }
            TearOp::Sprinkle {
                row,
                col,
                glyph,
                color,
            } => {
                let _ = write!(out, "\x1b[{row};{col}H");
                out.extend_from_slice(color.sgr().as_bytes());
                let mut buf = [0u8; 4];
                out.extend_from_slice(glyph.encode_utf8(&mut buf).as_bytes());
                if color != TearColor::None {
                    let _ = write!(out, "\x1b[0m");
                }
            }
            TearOp::ScrollRegion { top, bot, up, n } => {
                let _ = write!(out, "\x1b[{top};{bot}r");
                if up {
                    let _ = write!(out, "\x1b[{n}S");
                } else {
                    let _ = write!(out, "\x1b[{n}T");
                }
                let _ = write!(out, "\x1b[r");
            }
            TearOp::ShearBand {
                top,
                band_rows,
                col,
                n,
                step_n,
                insert,
            } => {
                for r in 0..band_rows {
                    let row_i = top + r;
                    let n_i = n + step_n * r;
                    let _ = write!(out, "\x1b[{row_i};{col}H");
                    if insert {
                        let _ = write!(out, "\x1b[{n_i}@");
                    } else {
                        let _ = write!(out, "\x1b[{n_i}P");
                    }
                }
            }
            TearOp::Stutter { row, n } => {
                let _ = write!(out, "\x1b[{row};1H\x1b[{n}L");
                let _ = write!(out, "\x1b[{row};1H\x1b[{n}M");
            }
        }
    }
    out
}

// ─────────────────────────────────── gating ──────────────────────────────────

/// Pure decision: should the tear run at all? Every input is a plain bool so
/// this is fully testable with no live terminal.
pub fn should_run_tear(
    boot_intro_setting: bool,
    no_intro_env: bool,
    is_tty: bool,
    vt_capable: bool,
) -> bool {
    boot_intro_setting && !no_intro_env && is_tty && vt_capable
}

/// The live probe half of [`should_run_tear`]'s `(is_tty, vt_capable)` pair:
/// is stdout a real terminal, and can it run VT/ANSI processing at all. Not
/// unit-testable itself (it reads the real console) — [`should_run_tear`]
/// carries the tested decision logic.
pub fn tty_and_vt_capable() -> bool {
    use bevy_ratatui::crossterm::tty::IsTty;
    std::io::stdout().is_tty() && crate::color::vt_capable()
}

// ──────────────────────────────── live driver ─────────────────────────────────

/// Run the tear against the REAL terminal: enable raw mode (to poll
/// keypresses without an Enter), loop the pure generator at `cfg.fps`,
/// writing each step straight to stdout, until `cfg.duration` elapses or any
/// key is pressed — then disable raw mode again. Touches no alternate screen
/// (writes land on the MAIN screen, exactly where the game was launched
/// from), so `bevy_ratatui`'s later `RatatuiPlugins` Startup (which enables
/// its OWN raw mode + alt screen) is unaffected either way.
///
/// The caller MUST have already checked [`should_run_tear`] — this function
/// performs no gating of its own besides bailing out if raw mode cannot even
/// be enabled (best-effort: if that fails we cannot safely poll keys or
/// guarantee clean output, so we do nothing rather than half-draw).
// `crossterm::event::Event` is workspace-`disallowed-types` (`clippy.toml`) to
// keep terminal event types confined to `re_present` — this crate's own
// declared exception, per that file's header comment: "False positives in
// re_present itself are expected and acceptable — they are the intentional
// use site." This function runs BEFORE any Bevy app / `bevy_ratatui` plugin
// exists, so there is no `KeyMessage` wrapper to read instead — raw
// `crossterm::event::read` is the only way to poll a keypress here.
#[allow(clippy::disallowed_types)]
pub fn run_screen_tear(cfg: &TearConfig) {
    use bevy_ratatui::crossterm::event::{poll, read, Event, KeyEventKind};
    use bevy_ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};

    // Observed live: immediately after a brand-new console host window is
    // created (e.g. Windows Terminal spawning a fresh tab/window), the very
    // first `enable_raw_mode()` call can transiently fail before the console
    // is fully attached to this process's stdio — a race with the host, not
    // a real capability gap. A silent one-shot bail here would skip the
    // ENTIRE tear phase for that launch with no visible symptom. A few short
    // retries costs at most a handful of milliseconds on the (common) first
    // try and turns a launch-order race into a non-issue.
    let mut enabled = enable_raw_mode().is_ok();
    for _ in 0..9u32 {
        if enabled {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
        enabled = enable_raw_mode().is_ok();
    }
    if !enabled {
        return;
    }

    let (cols, rows) = size().unwrap_or((80, 24));
    let fps = cfg.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / f64::from(fps));
    let total_steps = ((cfg.duration.as_secs_f64() * f64::from(fps)).ceil() as u32).max(1);

    let mut stdout = std::io::stdout();
    let start = std::time::Instant::now();

    'steps: for step in 0..total_steps {
        let ops = generate_tear_ops(cfg.seed, cols, rows, step, total_steps);
        let bytes = render_ops(&ops);
        if stdout.write_all(&bytes).is_err() {
            break;
        }
        let _ = stdout.flush();

        if step + 1 >= total_steps {
            break;
        }
        let target = frame_dur * (step + 1);
        let wait = target.saturating_sub(start.elapsed());
        if let Ok(true) = poll(wait) {
            if let Ok(Event::Key(k)) = read() {
                if k.kind == KeyEventKind::Press {
                    break 'steps;
                }
            }
        }
    }

    let _ = disable_raw_mode();
}

// ──────────────────────────────── exit teardown ───────────────────────────────

/// Force the terminal back to the MAIN screen, cooked (non-raw) mode, cursor
/// visible, colours reset, and CLEARED — see the module doc's "Exit" section
/// for the two call sites and why this must be idempotent rather than relying
/// on `Drop` ordering.
// `LeaveAlternateScreen` is workspace-`disallowed-types` for the same reason
// as `Event` above — this IS the confined, intentional use site `clippy.toml`
// documents. Called from the panic hook (before any `RatatuiContext` Drop has
// necessarily run) and after a normal `app.run()` return, so it cannot go
// through `RatatuiContext` either.
#[allow(clippy::disallowed_types)]
pub fn finalize_terminal_teardown() {
    use bevy_ratatui::crossterm::cursor::{MoveTo, Show};
    use bevy_ratatui::crossterm::execute;
    use bevy_ratatui::crossterm::style::ResetColor;
    use bevy_ratatui::crossterm::terminal::{
        disable_raw_mode, Clear, ClearType, LeaveAlternateScreen,
    };

    let mut stdout = std::io::stdout();
    let _ = execute!(stdout, LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = execute!(
        stdout,
        ResetColor,
        Clear(ClearType::All),
        MoveTo(0, 0),
        Show
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── determinism + bounds (generate_tear_ops) ──────────────────────────

    #[test]
    fn deterministic_given_a_seed() {
        // step 8 is a real BURST step for total_steps=60 (see
        // burst_steps_are_sparse_and_deterministic below) — a non-burst step
        // would trivially return empty for both calls and prove nothing.
        let a = generate_tear_ops(42, 80, 24, 8, 60);
        let b = generate_tear_ops(42, 80, 24, 8, 60);
        assert_eq!(a, b, "same inputs must produce the same ops, always");
        assert!(
            !a.is_empty(),
            "test setup must land on an actual burst step"
        );
    }

    #[test]
    fn different_seed_differs() {
        let a = generate_tear_ops(1, 80, 24, 8, 60);
        let b = generate_tear_ops(2, 80, 24, 8, 60);
        assert_ne!(a, b, "a different seed must (almost certainly) diverge");
    }

    #[test]
    fn empty_on_degenerate_grid() {
        assert!(generate_tear_ops(1, 0, 24, 8, 60).is_empty());
        assert!(generate_tear_ops(1, 80, 0, 8, 60).is_empty());
    }

    #[test]
    fn most_steps_are_calm_between_bursts() {
        // The whole point of the redesign: most of the run emits NOTHING, so
        // the screen holds still between bursts instead of glitching every
        // frame — "calm pauses between bursts".
        let total_steps = 60u32;
        let active = (0..total_steps)
            .filter(|&step| !generate_tear_ops(3, 100, 32, step, total_steps).is_empty())
            .count();
        assert_eq!(
            active as u32, BURST_COUNT,
            "exactly BURST_COUNT steps may ever emit ops"
        );
        assert!(
            (active as f64) < 0.2 * f64::from(total_steps),
            "active steps must be a small minority of the whole run: {active}/{total_steps}"
        );
    }

    #[test]
    fn burst_steps_are_sparse_and_deterministic() {
        let total_steps = 60u32;
        let mut burst_steps: Vec<u32> = (0..total_steps)
            .filter(|&s| burst_index_for_step(s, total_steps).is_some())
            .collect();
        burst_steps.sort_unstable();
        assert_eq!(burst_steps.len(), BURST_COUNT as usize);
        assert_ne!(
            burst_steps.first(),
            Some(&0),
            "the very first step must not be a burst (never opens mid-burst)"
        );
        assert_ne!(
            burst_steps.last(),
            Some(&(total_steps - 1)),
            "the very last step must not be a burst (never closes mid-burst)"
        );
        // Strictly increasing (no two bursts share a step).
        for w in burst_steps.windows(2) {
            assert!(
                w[0] < w[1],
                "burst steps must be distinct and ordered: {burst_steps:?}"
            );
        }
    }

    #[test]
    fn later_bursts_build_up_but_stay_small() {
        let total_steps = 60u32;
        let bursts: Vec<u32> = (0..total_steps)
            .filter(|&s| burst_index_for_step(s, total_steps).is_some())
            .collect();
        let counts: Vec<usize> = bursts
            .iter()
            .map(|&s| generate_tear_ops(11, 100, 32, s, total_steps).len())
            .collect();
        let first = *counts.first().unwrap();
        let last = *counts.last().unwrap();
        assert!(
            first <= last,
            "later bursts must not be sparser than earlier ones: {counts:?}"
        );
        for &c in &counts {
            assert!(
                (BURST_MIN_OPS as usize..=BURST_MAX_OPS as usize).contains(&c),
                "every burst must stay within the small BURST_MIN_OPS..=BURST_MAX_OPS band: {c} in {counts:?}"
            );
        }
    }

    #[test]
    fn erasure_is_confined_to_the_final_burst_and_stays_a_minority() {
        // EchHole (the only ERASURE op) must never appear at any burst
        // EXCEPT the final one, and even there stays a minority next to the
        // movement ops — "erasure should be the minority op, reserved for
        // the end".
        let cols = 100u16;
        let rows = 32u16;
        let total_steps = 60u32;
        let bursts: Vec<u32> = (0..total_steps)
            .filter(|&s| burst_index_for_step(s, total_steps).is_some())
            .collect();
        let final_step = *bursts.last().unwrap();
        let mut non_final_ech_holes = 0u32;
        let mut final_total = 0u32;
        let mut final_ech_holes = 0u32;
        for seed in 0..20u64 {
            for &step in &bursts {
                let ops = generate_tear_ops(seed, cols, rows, step, total_steps);
                if step == final_step {
                    final_total += ops.len() as u32;
                    final_ech_holes += ops
                        .iter()
                        .filter(|op| matches!(op, TearOp::EchHole { .. }))
                        .count() as u32;
                } else {
                    non_final_ech_holes += ops
                        .iter()
                        .filter(|op| matches!(op, TearOp::EchHole { .. }))
                        .count() as u32;
                }
            }
        }
        assert_eq!(
            non_final_ech_holes, 0,
            "EchHole must never appear before the final burst"
        );
        assert!(
            final_ech_holes > 0,
            "EchHole must be reachable at the final burst"
        );
        assert!(
            (final_ech_holes as f64) < 0.5 * (final_total as f64),
            "EchHole must stay a minority even at the final burst: {final_ech_holes}/{final_total}"
        );
    }

    #[test]
    fn sprinkle_is_a_minority_kind_across_the_whole_run() {
        // "sprinkle glyphs only sparingly" — across every burst, Sprinkle
        // must be clearly outnumbered by the movement kinds (shear/stutter/
        // scroll/shear-band).
        let total_steps = 60u32;
        let mut sprinkle = 0u32;
        let mut movement = 0u32;
        for seed in 0..30u64 {
            for step in 0..total_steps {
                for op in generate_tear_ops(seed, 100, 32, step, total_steps) {
                    match op {
                        TearOp::Sprinkle { .. } => sprinkle += 1,
                        TearOp::ShearBand { .. }
                        | TearOp::Stutter { .. }
                        | TearOp::ScrollRegion { .. } => movement += 1,
                        _ => {}
                    }
                }
            }
        }
        assert!(sprinkle > 0, "sprinkle must still be reachable at all");
        assert!(
            sprinkle < movement,
            "sprinkle must be a minority next to the movement kinds: sprinkle={sprinkle} movement={movement}"
        );
    }

    #[test]
    fn shear_and_stutter_appear_on_a_normal_grid() {
        // Movement ops must actually be reachable — not merely defined.
        let mut saw_shear = false;
        let mut saw_stutter = false;
        let mut saw_scroll = false;
        for seed in 0..30u64 {
            for step in 0..20u32 {
                for op in generate_tear_ops(seed, 100, 32, step, 20) {
                    match op {
                        TearOp::ShearBand { .. } => saw_shear = true,
                        TearOp::Stutter { .. } => saw_stutter = true,
                        TearOp::ScrollRegion { .. } => saw_scroll = true,
                        _ => {}
                    }
                }
            }
        }
        assert!(saw_shear, "ShearBand must be reachable on a normal grid");
        assert!(saw_stutter, "Stutter must be reachable on a normal grid");
        assert!(
            saw_scroll,
            "ScrollRegion must be reachable on a normal grid"
        );
    }

    #[test]
    fn every_op_is_bounded_to_the_grid() {
        let cols = 40u16;
        let rows = 20u16;
        for seed in 0..30u64 {
            for step in 0..40u32 {
                for op in generate_tear_ops(seed, cols, rows, step, 40) {
                    match op {
                        TearOp::DeleteChars { row, col, n }
                        | TearOp::InsertChars { row, col, n }
                        | TearOp::EchHole { row, col, n } => {
                            assert!((1..=rows).contains(&row), "{op:?} row out of bounds");
                            assert!((1..=cols).contains(&col), "{op:?} col out of bounds");
                            assert!(n >= 1, "{op:?} n must be at least 1");
                            assert!(
                                col as u32 + n as u32 - 1 <= cols as u32,
                                "{op:?} overruns the row"
                            );
                        }
                        TearOp::InsertLines { row, n } | TearOp::DeleteLines { row, n } => {
                            assert!((1..=rows).contains(&row), "{op:?} row out of bounds");
                            assert!(n >= 1);
                        }
                        TearOp::Sprinkle { row, col, .. } => {
                            assert!((1..=rows).contains(&row), "{op:?} row out of bounds");
                            assert!((1..=cols).contains(&col), "{op:?} col out of bounds");
                        }
                        TearOp::ScrollRegion { top, bot, n, .. } => {
                            assert!((1..=rows).contains(&top), "{op:?} top out of bounds");
                            assert!((1..=rows).contains(&bot), "{op:?} bot out of bounds");
                            assert!(top < bot, "{op:?} top must precede bot");
                            assert!(n >= 1);
                        }
                        TearOp::ShearBand {
                            top,
                            band_rows,
                            col,
                            n,
                            step_n,
                            ..
                        } => {
                            assert!((1..=rows).contains(&top), "{op:?} top out of bounds");
                            assert!(band_rows >= 2, "{op:?} a band must span at least 2 rows");
                            let bottom = top + band_rows - 1;
                            assert!(
                                bottom <= rows,
                                "{op:?} band overruns the grid: bottom={bottom} rows={rows}"
                            );
                            assert!((1..=cols).contains(&col), "{op:?} col out of bounds");
                            assert!(n >= 1, "{op:?} n must be at least 1");
                            let n_last = n + step_n * (band_rows - 1);
                            assert!(
                                col as u32 + n_last as u32 - 1 <= cols as u32,
                                "{op:?} last row of the band overruns: n_last={n_last} cols={cols}"
                            );
                        }
                        TearOp::Stutter { row, n } => {
                            assert!((1..=rows).contains(&row), "{op:?} row out of bounds");
                            assert!(n >= 1);
                        }
                    }
                }
            }
        }
    }

    // ── render_ops emits ONLY the allowed VT sequences ────────────────────

    /// A minimal byte-stream validator: walks `bytes` recognising exactly the
    /// closed CSI vocabulary this module emits, plus single-glyph UTF-8 text
    /// runs (the `Sprinkle` glyph). Any other byte sequence fails the scan.
    fn assert_only_allowed_sequences(bytes: &[u8]) {
        let s = std::str::from_utf8(bytes).expect("render_ops must emit valid UTF-8");
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                assert_eq!(
                    chars.next(),
                    Some('['),
                    "every ESC must be followed by CSI '['"
                );
                let mut params = String::new();
                loop {
                    match chars.peek().copied() {
                        Some(d) if d.is_ascii_digit() || d == ';' => {
                            params.push(d);
                            chars.next();
                        }
                        _ => break,
                    }
                }
                let final_byte = chars.next().expect("CSI sequence must have a final byte");
                assert!(
                    matches!(
                        final_byte,
                        'H' | 'P' | '@' | 'L' | 'M' | 'X' | 'r' | 'S' | 'T' | 'm'
                    ),
                    "disallowed CSI final byte {final_byte:?} (params {params:?})"
                );
            } else {
                // A plain glyph (the Sprinkle text) — any non-control char is fine;
                // it is never itself an escape byte since we already branched on it.
                assert!(
                    !c.is_control() || c == '\n',
                    "unexpected control byte {c:?}"
                );
            }
        }
    }

    #[test]
    fn render_ops_emits_only_the_allowed_vocabulary() {
        for seed in 0..10u64 {
            for step in 0..30u32 {
                let ops = generate_tear_ops(seed, 100, 32, step, 30);
                let bytes = render_ops(&ops);
                assert_only_allowed_sequences(&bytes);
            }
        }
    }

    #[test]
    fn render_ops_of_empty_is_empty() {
        assert!(render_ops(&[]).is_empty());
    }

    #[test]
    fn each_op_kind_renders_its_documented_shape() {
        assert_eq!(
            render_ops(&[TearOp::DeleteChars {
                row: 3,
                col: 5,
                n: 2
            }]),
            b"\x1b[3;5H\x1b[2P".to_vec()
        );
        assert_eq!(
            render_ops(&[TearOp::InsertChars {
                row: 1,
                col: 1,
                n: 4
            }]),
            b"\x1b[1;1H\x1b[4@".to_vec()
        );
        assert_eq!(
            render_ops(&[TearOp::InsertLines { row: 2, n: 1 }]),
            b"\x1b[2;1H\x1b[1L".to_vec()
        );
        assert_eq!(
            render_ops(&[TearOp::DeleteLines { row: 2, n: 1 }]),
            b"\x1b[2;1H\x1b[1M".to_vec()
        );
        assert_eq!(
            render_ops(&[TearOp::EchHole {
                row: 4,
                col: 4,
                n: 3
            }]),
            b"\x1b[4;4H\x1b[3X".to_vec()
        );
        assert_eq!(
            render_ops(&[TearOp::ScrollRegion {
                top: 1,
                bot: 10,
                up: true,
                n: 2
            }]),
            b"\x1b[1;10r\x1b[2S\x1b[r".to_vec()
        );
        let colored = render_ops(&[TearOp::Sprinkle {
            row: 1,
            col: 1,
            glyph: '#',
            color: TearColor::Red,
        }]);
        assert_eq!(colored, b"\x1b[1;1H\x1b[31m#\x1b[0m".to_vec());
        assert_eq!(
            render_ops(&[TearOp::ShearBand {
                top: 5,
                band_rows: 3,
                col: 2,
                n: 1,
                step_n: 2,
                insert: false
            }]),
            b"\x1b[5;2H\x1b[1P\x1b[6;2H\x1b[3P\x1b[7;2H\x1b[5P".to_vec(),
            "a delete-direction shear band walks n by step_n down each row of the band"
        );
        assert_eq!(
            render_ops(&[TearOp::ShearBand {
                top: 1,
                band_rows: 2,
                col: 1,
                n: 4,
                step_n: 0,
                insert: true
            }]),
            b"\x1b[1;1H\x1b[4@\x1b[2;1H\x1b[4@".to_vec(),
            "an insert-direction band uses @, and step_n=0 keeps every row's n identical"
        );
        assert_eq!(
            render_ops(&[TearOp::Stutter { row: 6, n: 2 }]),
            b"\x1b[6;1H\x1b[2L\x1b[6;1H\x1b[2M".to_vec(),
            "a stutter is IL immediately followed by DL at the same row -- a hiccup, not a net shift"
        );
        let plain = render_ops(&[TearOp::Sprinkle {
            row: 1,
            col: 1,
            glyph: '#',
            color: TearColor::None,
        }]);
        assert_eq!(plain, b"\x1b[1;1H#".to_vec());
    }

    // ── gating ─────────────────────────────────────────────────────────────

    #[test]
    fn should_run_tear_requires_every_condition() {
        assert!(should_run_tear(true, false, true, true));
        assert!(!should_run_tear(false, false, true, true), "settings off");
        assert!(!should_run_tear(true, true, true, true), "TS_NO_INTRO set");
        assert!(!should_run_tear(true, false, false, true), "not a tty");
        assert!(!should_run_tear(true, false, true, false), "not VT-capable");
    }
}
