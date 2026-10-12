//! Deterministic glyph-vine growth for the main menu.
//!
//! This is a scoped-down but REAL implementation of the space-colonization-lite
//! growth rule the spec describes: tips advance one cell per growth tick along
//! one of 8 headings, turn probabilistically (biased toward a hovered item and
//! away from every label rect), branch, die when blocked or too long, and the
//! whole field is capped at a coverage fraction — past the cap the single
//! oldest healthy cell withers glyph → `·` → `.` → (removed), freeing room for
//! growth elsewhere so the menu never finishes and never fills (the spec's
//! "coverage cap").
//!
//! **Deliberately out of scope**: the spec's F-channel Oklch luminance banding
//! and the "growth colour IS the progress display" shard-count → hue mapping
//! — both depend on the palette gate (`re_postfx`/`re_render`), which this
//! engine-pure module never touches (see [`AGE_BOLD_TICKS`] below: this
//! module hands back only an age in ticks, not a colour). The game's menu
//! paints fresh growth `BOLD` and aged stems `DIM` off that age band, which is
//! a real (not fake) approximation of the spec's "tip bright, stems dim after
//! 3s" intent without inventing a new colour pipeline here.
//!
//! Every random-looking decision is a deterministic hash of
//! `(seed, tip_id, step, salt)` — never a wall-clock or thread-local RNG — so
//! two runs with the same seed and the same repeller/attractor inputs produce
//! byte-identical growth, which is what keeps a menu golden/snapshot test
//! stable and is exercised directly by this module's `determinism` test.

use std::collections::BTreeMap;

// ─────────────────────────────── tuning ──────────────────────────────────
// Named per the design's own numbers — tune here, not scattered inline.

/// p(turn) per growth tick — the design calls for "turns ±1 heading with p = 0.3".
const TURN_PROBABILITY: f64 = 0.3;
/// p(branch) per growth tick — "branches with p = 0.08".
const BRANCH_PROBABILITY: f64 = 0.08;
/// "At most 24 live tips."
const MAX_LIVE_TIPS: usize = 24;
/// "A tip dies when blocked or longer than L = 40."
const MAX_TIP_LENGTH: u32 = 40;
/// "Coverage cap 30 %."
const COVERAGE_CAP_FRACTION: f64 = 0.30;
/// The wither sequence has 2 visible decay glyphs (`·`, `.`) before the cell
/// is removed (the 3rd step, `' '`, is "gone" — modeled as removal rather than
/// a literal blank cell, since a blank grown-cell and no grown-cell paint
/// identically).
const WITHER_MAX_STAGE: u8 = 2;
/// "Each tip advances 1 cell per 180 ms ± 40 %": a tip's per-tick speed is
/// hashed once at spawn into this range, and fractional `move_debt` carries
/// across ticks so a slow tip still averages one cell per its own period.
const SPEED_MIN: f64 = 0.6;
const SPEED_MAX: f64 = 1.4;
/// A grown cell younger than this many ticks paints `BOLD` (the "bright tip");
/// past it, paints `DIM` ("stems are dim") — the design's F-channel intent
/// without the Oklch banding this pass punts on (see module doc).
pub const AGE_BOLD_TICKS: u64 = 16;

/// The 8 headings, index order N, NE, E, SE, S, SW, W, NW — matches the angle
/// convention `heading_towards` computes with, so the two never disagree.
const HEADINGS: [(i32, i32); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

// Distinct hash salts so "should I turn this tick" and "should I branch this
// tick" and "which way do I turn" never accidentally consume the same hash
// stream and correlate with each other.
const SALT_TURN_ROLL: u64 = 1;
const SALT_TURN_DIR: u64 = 2;
const SALT_BRANCH_ROLL: u64 = 3;
const SALT_SPEED: u64 = 4;
const SALT_SPAWN_HEADING: u64 = 5;

// ───────────────────────────── deterministic hashing ─────────────────────
//
// A splitmix64-style mix — no wall clock, no thread-local RNG, just a pure
// function of its three integer inputs. Same inputs always produce the same
// output, on any machine, forever: that IS the determinism guarantee the
// spec asks for ("Tests drive the timer and pointer directly").

fn hash64(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// A value in `[0, 1)`, deterministic in `(a, b, c)`.
fn hash_unit(a: u64, b: u64, c: u64) -> f64 {
    // Top 53 bits -> an f64 mantissa's worth of precision, same trick every
    // splitmix-derived RNG uses to turn a u64 into a uniform float.
    ((hash64(a, b, c) >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
}

/// A signed unit in `[-1, 1)`, deterministic in `(a, b, c)`.
fn hash_signed(a: u64, b: u64, c: u64) -> f64 {
    hash_unit(a, b, c) * 2.0 - 1.0
}

// ───────────────────────────────── geometry ───────────────────────────────

/// A label's clickable box, inflated by margin at use — the thing growth must
/// never cover. The caller (the game's menu screen) builds these from its own
/// item-rect type; this module stays free of any ECS/menu-specific type so it
/// is testable in complete isolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Repeller {
    /// The single row this label occupies (labels are one cell tall).
    pub row: i32,
    /// Inclusive start column of the label text.
    pub col_start: i32,
    /// Exclusive end column of the label text.
    pub col_end: i32,
}

impl Repeller {
    /// `true` if `(col, row)` falls inside this label's box plus a 1-cell
    /// margin on every side.
    fn blocks(&self, col: i32, row: i32) -> bool {
        (row - self.row).abs() <= 1 && col >= self.col_start - 1 && col < self.col_end + 1
    }
}

/// The hovered item's box, used both as a turn-bias attractor and as the
/// target `OvergrowthState`'s hover-wrap state grows tendrils around.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoverRect {
    /// The single row this item occupies.
    pub row: i32,
    /// Inclusive start column of the item's label.
    pub col_start: i32,
    /// Exclusive end column of the item's label.
    pub col_end: i32,
}

impl HoverRect {
    fn center(&self) -> (i32, i32) {
        (
            self.col_start + (self.col_end - self.col_start) / 2,
            self.row,
        )
    }
}

/// The heading index (0..8, see [`HEADINGS`]) that points from `from` toward
/// `to`, rounded to the nearest 45°. Pure trig — no ECS.
fn heading_towards(from: (i32, i32), to: (i32, i32)) -> u8 {
    let dcol = f64::from(to.0 - from.0);
    let drow = f64::from(to.1 - from.1);
    if dcol == 0.0 && drow == 0.0 {
        return 0;
    }
    // North (0,-1) is angle 0; east (1,0) is 90° — atan2(dcol, -drow) matches
    // that convention directly.
    let mut deg = dcol.atan2(-drow).to_degrees();
    if deg < 0.0 {
        deg += 360.0;
    }
    (((deg / 45.0).round() as i64).rem_euclid(8)) as u8
}

/// Shortest signed turn (in `{-1, 0, +1}` steps of 45°) from `current` heading
/// toward `desired` heading.
fn shortest_turn(current: u8, desired: u8) -> i32 {
    let mut diff = i32::from(desired) - i32::from(current);
    diff = ((diff % 8) + 8) % 8; // 0..8
    if diff == 0 {
        0
    } else if diff <= 4 {
        1
    } else {
        -1
    }
}

// ──────────────────────────────── grown cells ─────────────────────────────

/// One planted cell of vine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowCell {
    /// The growth-tick this cell was planted on — the coverage cap withers
    /// the cell with the SMALLEST `planted_step` first ("the oldest branch"),
    /// and the age-band paint (bold vs dim) reads off it too.
    pub planted_step: u64,
    /// `0` = healthy (glyph chosen by connectivity). `1..=WITHER_MAX_STAGE` =
    /// mid-decay (`·`, `.`). Past `WITHER_MAX_STAGE` the cell is removed from
    /// the map outright, standing in for the spec's final `' '` stage.
    pub wither_stage: u8,
    /// Set only for `OvergrowthState`'s hover-wrap-planted perimeter cells —
    /// their glyph is the exact corner/side character the wrap walk chose,
    /// not whatever the generic connectivity mask would infer.
    pub forced_glyph: Option<char>,
}

/// One active growth tip.
#[derive(Clone, Debug, PartialEq)]
struct Tip {
    id: u64,
    col: i32,
    row: i32,
    heading: u8,
    length: u32,
    /// Ticks-per-cell speed, hashed once at spawn into `[SPEED_MIN, SPEED_MAX]`
    /// — the "±40 %" jitter. Fractional debt below carries the remainder.
    speed: f64,
    move_debt: f64,
    alive: bool,
}

/// Frozen once un-hovered ("these tendrils stop but stay" is the design intent).
/// While the SAME item stays hovered, `revealed` grows by 2 cells per tick
/// (one from each end of the perimeter walk) until the two ends meet.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HoverWrapState {
    rect: HoverRect,
    perimeter: Vec<(i32, i32, char)>,
    revealed: usize,
}

/// The whole growth field for one menu session. The caller seeds this from
/// its own per-session save data (the game's `Progress::session` counter) so
/// each session's menu grows differently but reproducibly.
#[derive(Clone, Debug, PartialEq)]
pub struct OvergrowthState {
    cells: BTreeMap<(i32, i32), GrowCell>,
    tips: Vec<Tip>,
    next_tip_id: u64,
    step: u64,
    seed: u64,
    hover_wrap: Option<HoverWrapState>,
}

impl OvergrowthState {
    /// Plant the 6 edge seeds (the design also calls for 2 additional
    /// eye-lower-lid seeds, skipped here: the eye does not live on
    /// the menu screen in this codebase, only on the separate Splash state,
    /// so there is no lower-lid geometry to seed from here).
    #[must_use]
    pub fn new(seed: u64, cols: usize, rows: usize) -> Self {
        let cols = cols as i32;
        let rows = rows as i32;
        let mut tips = Vec::new();
        let mut next_id = 0u64;
        let mut spawn = |col: i32, row: i32, heading: u8, seed: u64, next_id: &mut u64| {
            let speed = SPEED_MIN + hash_unit(seed, *next_id, SALT_SPEED) * (SPEED_MAX - SPEED_MIN);
            tips.push(Tip {
                id: *next_id,
                col,
                row,
                heading,
                length: 0,
                speed,
                move_debt: 0.0,
                alive: true,
            });
            *next_id += 1;
        };
        // North(0)/South(4)/East(2)/West(6) — see HEADINGS.
        spawn(cols / 4, 0, 4, seed, &mut next_id);
        spawn(3 * cols / 4, 0, 4, seed, &mut next_id);
        spawn(cols / 4, rows - 1, 0, seed, &mut next_id);
        spawn(3 * cols / 4, rows - 1, 0, seed, &mut next_id);
        spawn(0, rows / 2, 2, seed, &mut next_id);
        spawn(cols - 1, rows / 2, 6, seed, &mut next_id);
        OvergrowthState {
            cells: BTreeMap::new(),
            tips,
            next_tip_id: next_id,
            step: 0,
            seed,
            hover_wrap: None,
        }
    }

    /// Number of growth ticks elapsed since [`Self::new`].
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Number of cells currently planted (healthy or withering).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// The glyph to paint at `(col, row)`, if any growth occupies it.
    /// Connectivity-mask selection: straight
    /// runs are `│`/`─`, corners are `╭╮╰╯`, junctions are `├┤┬┴┼`, and a cell
    /// with no orthogonal neighbour but a grown diagonal neighbour reads as
    /// `╱`/`╲` — the two glyphs the spec lists for exactly that case.
    #[must_use]
    pub fn glyph_at(&self, col: i32, row: i32) -> Option<char> {
        let cell = self.cells.get(&(col, row))?;
        if let Some(g) = cell.forced_glyph {
            return Some(g);
        }
        match cell.wither_stage {
            0 => {}
            1 => return Some('·'),
            _ => return Some('.'),
        }
        let n = self.cells.contains_key(&(col, row - 1));
        let s = self.cells.contains_key(&(col, row + 1));
        let e = self.cells.contains_key(&(col + 1, row));
        let w = self.cells.contains_key(&(col - 1, row));
        Some(connectivity_glyph(n, e, s, w).unwrap_or_else(|| {
            let ne = self.cells.contains_key(&(col + 1, row - 1));
            let sw = self.cells.contains_key(&(col - 1, row + 1));
            let nw = self.cells.contains_key(&(col - 1, row - 1));
            let se = self.cells.contains_key(&(col + 1, row + 1));
            if ne || sw {
                '╱'
            } else if nw || se {
                '╲'
            } else {
                '·'
            }
        }))
    }

    /// Whether `(col, row)` is planted at all (any wither stage) — the
    /// painter uses this to know which cells to touch; `glyph_at` for what
    /// to draw there. Also `true` for the cell this exact tick planted, so a
    /// same-tick neighbour mask sees it immediately (no one-tick lag).
    #[must_use]
    pub fn planted_step_at(&self, col: i32, row: i32) -> Option<u64> {
        self.cells.get(&(col, row)).map(|c| c.planted_step)
    }

    /// Advance the whole field by one growth tick: decay withering cells,
    /// enforce the coverage cap, advance every live tip (move / turn /
    /// branch / die), and advance the active hover-wrap reveal, if any.
    /// Pure given `(cols, rows, repellers, attractor)` plus `self`'s own
    /// state — no wall clock, no ECS — so two identically-seeded states fed
    /// the identical call sequence produce byte-identical results.
    pub fn step(
        &mut self,
        cols: usize,
        rows: usize,
        repellers: &[Repeller],
        attractor: Option<HoverRect>,
    ) {
        self.step += 1;
        let cols_i = cols as i32;
        let rows_i = rows as i32;

        self.decay_wither();
        self.enforce_coverage_cap(cols, rows);
        self.advance_tips(cols_i, rows_i, repellers, attractor.map(|r| r.center()));
        self.advance_hover_wrap();
    }

    fn decay_wither(&mut self) {
        let mut remove = Vec::new();
        for (pos, cell) in &mut self.cells {
            if cell.wither_stage > 0 {
                if cell.wither_stage >= WITHER_MAX_STAGE {
                    remove.push(*pos);
                } else {
                    cell.wither_stage += 1;
                }
            }
        }
        for pos in remove {
            self.cells.remove(&pos);
        }
    }

    fn enforce_coverage_cap(&mut self, cols: usize, rows: usize) {
        let cap = ((cols * rows) as f64 * COVERAGE_CAP_FRACTION) as usize;
        if self.cells.len() <= cap {
            return;
        }
        // Already something withering from a previous tick or a click recoil
        // — let it finish before starting another, so at most one branch
        // withers "from tip to root" at a time.
        if self.cells.values().any(|c| c.wither_stage > 0) {
            return;
        }
        if let Some((&pos, _)) = self
            .cells
            .iter()
            .filter(|(_, c)| c.wither_stage == 0)
            .min_by_key(|(_, c)| c.planted_step)
        {
            self.cells.get_mut(&pos).unwrap().wither_stage = 1;
        }
    }

    fn advance_tips(
        &mut self,
        cols: i32,
        rows: i32,
        repellers: &[Repeller],
        attractor: Option<(i32, i32)>,
    ) {
        let live_count_before = self.tips.iter().filter(|t| t.alive).count();
        let mut spawned = Vec::new();
        let mut live_count = live_count_before;

        for tip in &mut self.tips {
            if !tip.alive {
                continue;
            }
            tip.move_debt += tip.speed;
            if tip.move_debt < 1.0 {
                continue;
            }
            tip.move_debt -= 1.0;

            if hash_unit(self.seed, tip.id, self.step + SALT_TURN_ROLL) < TURN_PROBABILITY {
                let desired = attractor
                    .map(|a| heading_towards((tip.col, tip.row), a))
                    .unwrap_or(tip.heading);
                let dir = match shortest_turn(tip.heading, desired) {
                    0 => {
                        if hash_signed(self.seed, tip.id, self.step + SALT_TURN_DIR) >= 0.0 {
                            1
                        } else {
                            -1
                        }
                    }
                    d => d,
                };
                tip.heading = ((i32::from(tip.heading) + dir).rem_euclid(8)) as u8;
            }

            let (dcol, drow) = HEADINGS[tip.heading as usize];
            let ncol = tip.col + dcol;
            let nrow = tip.row + drow;
            let blocked = ncol < 0
                || nrow < 0
                || ncol >= cols
                || nrow >= rows
                || repellers.iter().any(|r| r.blocks(ncol, nrow));
            if blocked {
                tip.alive = false;
                live_count -= 1;
                continue;
            }

            tip.col = ncol;
            tip.row = nrow;
            tip.length += 1;
            self.cells.entry((ncol, nrow)).or_insert(GrowCell {
                planted_step: self.step,
                wither_stage: 0,
                forced_glyph: None,
            });

            if tip.length > MAX_TIP_LENGTH {
                tip.alive = false;
                live_count -= 1;
                continue;
            }

            if live_count < MAX_LIVE_TIPS
                && hash_unit(self.seed, tip.id, self.step + SALT_BRANCH_ROLL) < BRANCH_PROBABILITY
            {
                let id = self.next_tip_id;
                self.next_tip_id += 1;
                let dir = if hash_signed(self.seed, id, self.step + SALT_SPAWN_HEADING) >= 0.0 {
                    2
                } else {
                    -2
                };
                let heading = ((i32::from(tip.heading) + dir).rem_euclid(8)) as u8;
                let speed =
                    SPEED_MIN + hash_unit(self.seed, id, SALT_SPEED) * (SPEED_MAX - SPEED_MIN);
                spawned.push(Tip {
                    id,
                    col: ncol,
                    row: nrow,
                    heading,
                    length: 0,
                    speed,
                    move_debt: 0.0,
                    alive: true,
                });
                live_count += 1;
            }
        }

        self.tips.retain(|t| t.alive);
        self.tips.extend(spawned);
    }

    /// Begin (or restart) wrapping `rect`'s perimeter — call once, on the
    /// tick hover ENTERS a new item: two new tips spawn and grow toward the
    /// label, then wrap its rect. A no-op if `rect` is already
    /// the item being wrapped (so re-calling every frame while the same item
    /// stays hovered does not restart the reveal).
    pub fn on_hover_start(&mut self, rect: HoverRect) {
        if self.hover_wrap.as_ref().is_some_and(|w| w.rect == rect) {
            return;
        }
        self.hover_wrap = Some(HoverWrapState {
            perimeter: perimeter_walk(rect),
            revealed: 0,
            rect,
        });
    }

    /// Freeze the current wrap where it stands (on un-hover these
    /// tendrils stop but stay): the already-revealed perimeter cells stay
    /// planted in `self.cells` (they were injected as they revealed — see
    /// `advance_hover_wrap`), and no more get added. Idempotent.
    pub fn on_hover_end(&mut self) {
        self.hover_wrap = None;
    }

    fn advance_hover_wrap(&mut self) {
        let Some(wrap) = &mut self.hover_wrap else {
            return;
        };
        let total = wrap.perimeter.len();
        if wrap.revealed >= total {
            return;
        }
        // Reveal one more cell from each end, meeting in the middle — the
        // "two new tips" from either side of the label.
        let front = wrap.revealed;
        let back = total - 1 - wrap.revealed;
        let step = self.step;
        for idx in [front, back] {
            if idx >= total {
                continue;
            }
            let (col, row, glyph) = wrap.perimeter[idx];
            self.cells
                .entry((col, row))
                .and_modify(|c| c.forced_glyph = Some(glyph))
                .or_insert(GrowCell {
                    planted_step: step,
                    wither_stage: 0,
                    forced_glyph: Some(glyph),
                });
        }
        wrap.revealed += 1;
    }

    /// Click recoil ("on click release, every growth cell within `radius`
    /// cells of the label recoils"): begin withering every healthy cell
    /// within `radius` (Chebyshev distance, matching the cell-grid's own
    /// 8-directional growth) of `center`. The existing per-tick decay pass
    /// (`decay_wither`) then carries them through `·` → `.` → removed exactly
    /// like organic coverage-cap withering — a real ripple, just simplified
    /// from "oldest-first over 250 ms" to "all at once, then decays over the
    /// normal 2-tick wither timeline" (see this module's doc for why).
    pub fn recoil(&mut self, center: (i32, i32), radius: i32) {
        for (&(col, row), cell) in &mut self.cells {
            if cell.wither_stage == 0 {
                let d = (col - center.0).abs().max((row - center.1).abs());
                if d <= radius {
                    cell.wither_stage = 1;
                }
            }
        }
    }
}

/// Neighbour-mask → connectivity glyph, minus the diagonal pair
/// which `glyph_at` supplies as a fallback when this returns `None`.
fn connectivity_glyph(n: bool, e: bool, s: bool, w: bool) -> Option<char> {
    Some(match (n, e, s, w) {
        (true, true, true, true) => '┼',
        (true, true, true, false) => '├',
        (true, false, true, true) => '┤',
        (false, true, true, true) => '┬',
        (true, true, false, true) => '┴',
        (true, false, true, false) => '│',
        (false, true, false, true) => '─',
        (true, true, false, false) => '╰',
        (true, false, false, true) => '╯',
        (false, true, true, false) => '╭',
        (false, false, true, true) => '╮',
        (true, false, false, false) | (false, false, true, false) => '│',
        (false, true, false, false) | (false, false, false, true) => '─',
        (false, false, false, false) => return None,
    })
}

/// The ordered perimeter cells of `rect` inflated by 1 cell of margin (so the
/// wrap never sits directly on the label it wraps), walked clockwise starting
/// at the top-left corner, each paired with the exact box-drawing glyph that
/// cell paints — corners `╭╮╰╯`, sides `─`/`│`. Pure geometry, so it is
/// itself directly unit-testable (closed loop, right length, right corners).
fn perimeter_walk(rect: HoverRect) -> Vec<(i32, i32, char)> {
    let top = rect.row - 1;
    let bottom = rect.row + 1;
    let left = rect.col_start - 1;
    let right = rect.col_end; // col_end is already exclusive => this IS the outer edge
    let mut out = Vec::new();
    if right <= left {
        return out;
    }
    // top edge, left -> right
    for col in left..=right {
        let glyph = if col == left {
            '╭'
        } else if col == right {
            '╮'
        } else {
            '─'
        };
        out.push((col, top, glyph));
    }
    // right edge, top+1 -> bottom-1
    for row in (top + 1)..bottom {
        out.push((right, row, '│'));
    }
    // bottom edge, right -> left
    for col in (left..=right).rev() {
        let glyph = if col == right {
            '╯'
        } else if col == left {
            '╰'
        } else {
            '─'
        };
        out.push((col, bottom, glyph));
    }
    // left edge, bottom-1 -> top+1 (open — does not repeat the top-left corner)
    for row in ((top + 1)..bottom).rev() {
        out.push((left, row, '│'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_repellers() -> Vec<Repeller> {
        Vec::new()
    }

    #[test]
    fn new_state_plants_no_cells_until_stepped() {
        let state = OvergrowthState::new(7, 80, 24);
        assert_eq!(
            state.cell_count(),
            0,
            "seeds are TIPS, not yet planted cells"
        );
        assert_eq!(state.step_count(), 0);
    }

    #[test]
    fn stepping_grows_at_least_one_cell_per_live_tip_eventually() {
        let mut state = OvergrowthState::new(1, 80, 24);
        for _ in 0..5 {
            state.step(80, 24, &no_repellers(), None);
        }
        assert!(
            state.cell_count() > 0,
            "6 edge tips advancing for 5 ticks must have planted SOMETHING"
        );
    }

    #[test]
    fn determinism_same_seed_same_inputs_same_result() {
        let repellers = vec![Repeller {
            row: 12,
            col_start: 30,
            col_end: 50,
        }];
        let run = |seed: u64| {
            let mut state = OvergrowthState::new(seed, 80, 24);
            for _ in 0..120 {
                state.step(80, 24, &repellers, None);
            }
            state
        };
        let a = run(42);
        let b = run(42);
        assert_eq!(
            a.cells, b.cells,
            "identical seed + identical inputs must replay identically"
        );
        assert_eq!(a.tips, b.tips);
    }

    #[test]
    fn different_seeds_diverge() {
        let repellers = no_repellers();
        let run = |seed: u64| {
            let mut state = OvergrowthState::new(seed, 80, 24);
            for _ in 0..120 {
                state.step(80, 24, &repellers, None);
            }
            state
        };
        let a = run(1);
        let b = run(2);
        assert_ne!(a.cells, b.cells, "different sessions must grow differently");
    }

    #[test]
    fn growth_never_enters_a_repellers_margin() {
        let repellers = vec![Repeller {
            row: 12,
            col_start: 30,
            col_end: 50,
        }];
        let mut state = OvergrowthState::new(99, 80, 24);
        for _ in 0..300 {
            state.step(80, 24, &repellers, None);
        }
        for &(col, row) in state.cells.keys() {
            assert!(
                !repellers[0].blocks(col, row),
                "a grown cell landed inside the label's blocked margin at ({col},{row})"
            );
        }
    }

    #[test]
    fn coverage_cap_is_respected_after_many_ticks() {
        let mut state = OvergrowthState::new(5, 40, 12);
        for _ in 0..2000 {
            state.step(40, 12, &no_repellers(), None);
        }
        let cap = ((40 * 12) as f64 * COVERAGE_CAP_FRACTION) as usize;
        // A small tolerance: the cap check runs once per tick BEFORE that
        // tick's moves, so growth can overshoot the cap by at most the
        // number of tips that move in a single tick before the next
        // enforcement pass catches it.
        assert!(
            state.cell_count() <= cap + MAX_LIVE_TIPS,
            "cell_count {} should stay near the {}-cell cap",
            state.cell_count(),
            cap
        );
    }

    #[test]
    fn withering_cells_decay_then_vanish() {
        let mut state = OvergrowthState::new(3, 80, 24);
        state.step(80, 24, &no_repellers(), None);
        let pos = *state
            .cells
            .keys()
            .next()
            .expect("at least one cell planted after a tick");
        state.cells.get_mut(&pos).unwrap().wither_stage = 1;
        assert_eq!(state.glyph_at(pos.0, pos.1), Some('·'));
        state.decay_wither();
        assert_eq!(state.glyph_at(pos.0, pos.1), Some('.'));
        state.decay_wither();
        assert_eq!(
            state.glyph_at(pos.0, pos.1),
            None,
            "past WITHER_MAX_STAGE the cell is gone"
        );
    }

    #[test]
    fn recoil_starts_withering_only_cells_in_radius() {
        let mut state = OvergrowthState::new(11, 80, 24);
        for _ in 0..40 {
            state.step(80, 24, &no_repellers(), None);
        }
        assert!(
            state.cell_count() > 1,
            "need at least 2 planted cells for a near/far split"
        );
        // Recoil is centered on an actually-planted cell so "near" is
        // guaranteed non-empty regardless of which edge tips happened to
        // reach where within 40 ticks.
        let center = *state.cells.keys().next().unwrap();
        let far_pos = *state
            .cells
            .keys()
            .max_by_key(|(c, r)| (c - center.0).abs() + (r - center.1).abs())
            .unwrap();
        state.recoil(center, 2);
        assert_eq!(
            state.cells.get(&center).map(|c| c.wither_stage),
            Some(1),
            "the recoil center itself must start withering"
        );
        let far_distance = (far_pos.0 - center.0)
            .abs()
            .max((far_pos.1 - center.1).abs());
        if far_distance > 2 {
            assert_eq!(
                state.cells.get(&far_pos).map(|c| c.wither_stage),
                Some(0),
                "a cell far outside the recoil radius must be untouched"
            );
        }
    }

    #[test]
    fn perimeter_walk_is_a_closed_loop_with_the_right_corners() {
        let rect = HoverRect {
            row: 10,
            col_start: 20,
            col_end: 26,
        };
        let cells = perimeter_walk(rect);
        assert!(cells.contains(&(19, 9, '╭')));
        assert!(cells.contains(&(26, 9, '╮')));
        assert!(cells.contains(&(26, 11, '╯')));
        assert!(cells.contains(&(19, 11, '╰')));
        // No repeated cell — a real perimeter walk, not overlapping segments.
        let mut positions: Vec<(i32, i32)> = cells.iter().map(|(c, r, _)| (*c, *r)).collect();
        positions.sort_unstable();
        let before = positions.len();
        positions.dedup();
        assert_eq!(
            positions.len(),
            before,
            "perimeter walk must not revisit a cell"
        );
    }

    #[test]
    fn hover_wrap_reveals_from_both_ends_and_freezes_on_unhover() {
        let rect = HoverRect {
            row: 10,
            col_start: 20,
            col_end: 26,
        };
        let mut state = OvergrowthState::new(1, 80, 24);
        state.on_hover_start(rect);
        for _ in 0..3 {
            state.advance_hover_wrap();
        }
        let revealed_while_hovered = state.cell_count();
        assert!(
            revealed_while_hovered > 0,
            "hovering must reveal SOME perimeter cells"
        );
        state.on_hover_end();
        // Un-hovering does not erase what was already revealed ("stop but
        // stay") — cell_count must not drop.
        assert_eq!(state.cell_count(), revealed_while_hovered);
        // And with no active wrap, further ticks add nothing new to it.
        state.advance_hover_wrap();
        assert_eq!(state.cell_count(), revealed_while_hovered);
    }

    #[test]
    fn hover_wrap_center_and_heading_towards_agree_on_cardinal_directions() {
        assert_eq!(heading_towards((0, 0), (0, -5)), 0, "straight up is North");
        assert_eq!(heading_towards((0, 0), (5, 0)), 2, "straight right is East");
        assert_eq!(heading_towards((0, 0), (0, 5)), 4, "straight down is South");
        assert_eq!(heading_towards((0, 0), (-5, 0)), 6, "straight left is West");
    }

    #[test]
    fn shortest_turn_picks_the_short_way_around() {
        assert_eq!(shortest_turn(0, 0), 0);
        assert_eq!(shortest_turn(0, 2), 1, "N -> E is a right turn");
        assert_eq!(
            shortest_turn(0, 6),
            -1,
            "N -> W is a left turn, not 3 rights"
        );
        assert_eq!(
            shortest_turn(0, 4),
            1,
            "a straight reversal picks a consistent (right) tie-break"
        );
    }
}
