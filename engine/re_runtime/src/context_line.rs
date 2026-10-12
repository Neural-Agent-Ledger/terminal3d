//! The faced-switch HUD context-line renderer.
//!
//! The reference technique this follows fills a panel for the nearest
//! interactable with a taxonomy label, a title, a live state, and what the
//! interact key will do — the key prompt lives in the copy. The only
//! faced-object feedback before this was the crosshair switching `+` → `◆`,
//! which tells the player *something* is faced but not *what*, *what state
//! it's in*, or *what `E` will do*.
//!
//! ## Scope: one real category today
//!
//! The design sketch names several possible categories (switch, puzzle node,
//! door, exit, note/terminal). TERMINAL STATE's engine exposes exactly ONE of
//! those as a discrete `E`-interactable today: the pipe **switch**
//! ([`crate::hud_probe::faced_switch`] / [`PuzzleState::pipe_at_switch`]). A
//! puzzle NODE is never itself pressed (only the pipes feeding it are); a door
//! is walked through, not `E`'d; notes/terminals are read-only wall text with
//! no interact verb. Building a broader "faced interactable" taxonomy spanning
//! those kinds is real future work, not invented here.
//!
//! ## No new raycast
//!
//! [`faced_context`] reuses [`crate::hud_probe::faced_switch`] — the EXACT
//! faced-cell probe a crosshair and an interact path already run every frame —
//! then one `O(pipes)` lookup via [`re_puzzle::PuzzleState::pipe_at_switch`] (a
//! handful of pipes per level, not a raycast).
//!
//! ## Per-entity labels: what the level format would need
//!
//! `re_level::Node::name` is the closest thing to an authored label today
//! (`R`, `G`, `mix`, `output`, …), so [`faced_context`] reads the pipe's
//! endpoint node names as the title (`R → MIX`) and falls back to
//! `PIPE #<index>` when a name is missing. There is no dedicated per-switch
//! `label =` / `desc =` field.
//!
//! ## The colour seam
//!
//! This module knows nothing about a host's own palette constants. Every
//! painting entry point takes a [`HudPalette`] the host fills from its own
//! `FG`/`BG` (or equivalent) — the exact split `re_asciifx::paint`'s
//! `put_str`/`put_center`/`fill_bg` already use for their own `fg`/`bg`
//! parameters, applied here to a whole panel instead of one string.
//!
//! ## Proximity-picked label (a second entry point)
//!
//! The reference floats a prompt beside the *nearest* interactable in a
//! short range+cone, not only the one under the reticle.
//! [`faced_context`]/[`draw_faced_context`] above are UNCHANGED by
//! this addition — they stay the facing-only rule the existing tests pin. The
//! proximity system is a parallel, self-contained addition:
//! [`nearest_candidate`] (stateless sweep), [`ProximityTracker`] (adds the
//! anti-flicker hysteresis a real per-frame HUD wants), and
//! [`draw_proximity_context`] (resolve + paint).

use bevy::prelude::Resource;
use re_core::{Cell, Framebuffer, Vec3};
use re_level::Level;
use re_puzzle::{Coord3, Pipe, PuzzleState};
use re_render::lines::{world_to_screen, ScreenPoint};
use re_render::Fov;
use re_world::CameraPose;

use crate::hud_probe::faced_switch;
use re_asciifx::paint::put_str;

/// The colours a host lends this module to paint with — the seam that keeps
/// this crate ignorant of any host's own palette constants.
/// `fg` is the HUD's default text
/// colour (a host's title text), `bg` is the HUD's flat background (the same
/// value every other HUD row is stamped with, so a panel renders consistently
/// however the active [`re_present::ColorMode`] quantises it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudPalette {
    /// Default text colour (titles, separators).
    pub fg: u32,
    /// Flat HUD background every row/strip is stamped with.
    pub bg: u32,
}

/// The taxonomy tag, in our own terminal voice — a category prefix, not a
/// literal reuse of a study reference's own strings.
const CATEGORY_SWITCH: &str = "::SWITCH";

/// What `E` does right now, normally.
const HINT_TOGGLE: &str = "E toggle";
/// What `E` does right now when the grid is at its `power_budget` cap and
/// this switch is OFF — pressing `E` would REFUSE, not toggle (mirrors the
/// wording a host's own `power_hint` HUD row already uses, so the two
/// readouts never disagree).
const HINT_REFUSED: &str = "E \u{2014} refused, grid full";

const CTX_FG_CATEGORY: u32 = 0x0080_A8FF;
const CTX_FG_ON: u32 = 0x0070_D8A0;
const CTX_FG_OFF: u32 = 0x0090_9090;
const CTX_FG_HINT: u32 = 0x0060_7888;

/// One faced-object context line, ready to paint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextLine {
    /// The taxonomy tag (e.g. [`CATEGORY_SWITCH`]).
    pub category: &'static str,
    /// The entity's label: the pipe's endpoint node names when the level
    /// names them (`R \u{2192} MIX`), else `PIPE #<index>`.
    pub title: String,
    /// The live state readout (`ON` / `OFF`).
    pub state: &'static str,
    /// True iff the pipe currently carries power (`state == "ON"`).
    pub on: bool,
    /// What `E` will do right now.
    pub hint: &'static str,
}

/// Build the context line for whatever the camera currently faces, or `None`
/// when nothing interactable is in range. See the module doc for the reuse
/// contract (no new raycast) and the label fallback.
pub fn faced_context(
    cam: &CameraPose,
    puzzle: &PuzzleState,
    level: Option<&Level>,
) -> Option<ContextLine> {
    let (x, y, z, active) = faced_switch(cam, puzzle);
    if !active {
        return None;
    }
    let idx = puzzle.pipe_at_switch(x, y, z)?;
    let pipe = puzzle.pipes.get(idx)?;
    let on = pipe.is_on();

    let node_name = |i: usize| -> Option<&str> {
        level
            .and_then(|lvl| lvl.nodes.get(i))
            .map(|n| n.name.as_str())
            .filter(|n| !n.is_empty())
    };
    let title = match (node_name(pipe.from), node_name(pipe.to)) {
        (Some(f), Some(t)) => format!("{f} \u{2192} {t}"),
        (Some(f), None) => format!("{f} \u{2192} ?"),
        (None, Some(t)) => format!("? \u{2192} {t}"),
        (None, None) => format!("PIPE #{idx}"),
    };

    let refused = !on && puzzle.has_power_budget() && !puzzle.power_available();
    Some(ContextLine {
        category: CATEGORY_SWITCH,
        title,
        state: if on { "ON" } else { "OFF" },
        on,
        hint: if refused { HINT_REFUSED } else { HINT_TOGGLE },
    })
}

/// Cells of padding the dim backing strip ([`paint_backing_strip`]) extends
/// beyond the text span on each side, so the panel reads as a small plate
/// sitting IN FRONT of a busy wall texture rather than a caption whose first
/// and last glyphs touch the scene directly.
const STRIP_PADDING: usize = 1;

/// Fill `[col_start - STRIP_PADDING, col_end + STRIP_PADDING)` on `row` with a
/// blank, `bg`-coloured cell — the dim backing strip. Painted BEFORE the text
/// segments (which then overwrite their own span with the SAME `bg` via
/// `put_str`), so this call only needs to darken the padding columns; using
/// one shared colour end to end keeps the strip visually seamless instead of
/// showing a colour seam at the text's own edges. Clipped to the frame, so it
/// never panics on a narrow/short grid.
fn paint_backing_strip(
    fb: &mut Framebuffer,
    col_start: usize,
    col_end: usize,
    row: usize,
    bg: u32,
) {
    if row >= fb.rows() {
        return;
    }
    let cols = fb.cols();
    let from = col_start.saturating_sub(STRIP_PADDING);
    let to = (col_end + STRIP_PADDING).min(cols);
    let blank = Cell {
        glyph: u32::from(' '),
        fg: bg,
        bg,
        ..Default::default()
    };
    for c in from..to {
        fb.set(c, row, blank);
    }
}

/// Paint `ctx` as one compact, centred, BG-stamped line just under the
/// crosshair row (`rows/2 + 2`) — close to the faced object's screen position
/// without sitting ON the reticle itself. A no-op on a frame too short to have
/// a row there.
///
/// Backed by a [`paint_backing_strip`] a little wider than the text itself: over
/// a plain menu background this is invisible (BG-on-BG), but over a textured
/// wall cell — exactly where the crosshair sits most of the time — it keeps the
/// line legible instead of dissolving into the scene right at its own edges.
fn paint_context_line(fb: &mut Framebuffer, ctx: &ContextLine, palette: HudPalette) {
    let cols = fb.cols();
    let row = fb.rows() / 2 + 2;
    if row >= fb.rows() || cols == 0 {
        return;
    }

    let state_fg = if ctx.on { CTX_FG_ON } else { CTX_FG_OFF };
    let segments: [(&str, u32); 5] = [
        (ctx.category, CTX_FG_CATEGORY),
        ("  ", palette.fg),
        (ctx.title.as_str(), palette.fg),
        ("  ", palette.fg),
        (ctx.state, state_fg),
    ];
    let core_width: usize = segments.iter().map(|(s, _)| s.chars().count()).sum();
    let tail = format!("  \u{00B7}  {}", ctx.hint);
    let total_width = core_width + tail.chars().count();

    let x0 = cols.saturating_sub(total_width) / 2;
    paint_backing_strip(fb, x0, x0 + total_width, row, palette.bg);

    let mut x = x0;
    for (text, fg) in segments {
        put_str(fb, x, row, text, fg, palette.bg);
        x += text.chars().count();
    }
    put_str(fb, x, row, &tail, CTX_FG_HINT, palette.bg);
}

/// The one call a host's play HUD makes: resolve the faced context (if any)
/// and paint it. A no-op — leaves the frame byte-identical — when nothing
/// interactable is faced, so a level with the reticle idle shows no panel at
/// all. `palette` supplies the host's own title/background colours (see
/// [`HudPalette`]).
pub fn draw_faced_context(
    fb: &mut Framebuffer,
    cam: &CameraPose,
    puzzle: &PuzzleState,
    level: Option<&Level>,
    palette: HudPalette,
) {
    if let Some(ctx) = faced_context(cam, puzzle, level) {
        paint_context_line(fb, &ctx, palette);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Proximity-picked label -- a verb + noun + skill-level prompt floats beside
// the *nearest* interactable in range, which is not necessarily the one under
// the reticle. The facing rule above stays
// the rule for ACTING (interact still requires `faced_switch`); this is a
// second, purely informational candidate that only LABELS the nearest switch
// in a short view cone, so a player discovers a switch without pixel-aiming
// at it first.
//
// Anchored with the render layer's own public projection
// (`re_render::lines::world_to_screen`, the exact basis
// `re_render::lines::LineViewBasis::project` casts and authored lines use),
// and occluded against the FRAMEBUFFER's own depth buffer already sitting in
// `fb`. No new `pub` surface was needed in `re_render` for either.
// ─────────────────────────────────────────────────────────────────────────

/// Candidate range, in world cells (Euclidean, camera to the switch cell's
/// centre) -- "within 2 cells".
const PROXIMITY_RANGE_CELLS: f32 = 2.0;

/// Half-angle of the horizontal view cone, in degrees -- "\u{00B1}35\u{00B0}".
/// Horizontal only (yaw), matching [`faced_switch`]'s own horizontal-only
/// forward probe; pitch is not part of either cone.
const PROXIMITY_HALF_ANGLE_DEG: f32 = 35.0;

/// Depth slack (world-perpendicular-distance cells) an occluding cell must
/// clear before a candidate is rejected as hidden. A candidate switch is
/// mounted ON a wall, so the framebuffer's own depth at its projected pixel
/// is the WALL FACE's depth, up to about half a cell nearer than the cell
/// CENTRE this module projects (depending on view angle) -- that gap must
/// not itself read as an occluder. A wall genuinely standing between the
/// camera and the switch sits at least a further full cell forward, clearing
/// this margin easily.
const PROXIMITY_OCCLUSION_SLACK: f32 = 0.75;

/// Screen rows a host's top-of-frame HUD reserves that a proximity label's
/// anchor must not paint over. A plain literal (not a cross-crate constant
/// reference), since this crate has no visibility into a host's own layout —
/// a host that reserves more rows passes a taller value in a future overload
/// if this ever needs to be configurable; today it matches the host's HUD.
const PROXIMITY_MIN_ROW: usize = 11;

/// One resolved proximity label: a noun (+ live state) anchored at its own
/// screen position, with the "\[E\] toggle" verb attached only when this same
/// object is ALSO the one [`faced_context`] would report (show the verb only
/// when the object is also faced).
#[derive(Debug, Clone, PartialEq)]
pub struct ProximityLabel {
    /// The taxonomy tag (e.g. [`CATEGORY_SWITCH`]).
    pub category: &'static str,
    /// The entity's label: the pipe's endpoint node names when the level
    /// names them (`R \u{2192} MIX`), else `PIPE #<index>`.
    pub title: String,
    /// The live state readout (`ON` / `OFF`).
    pub state: &'static str,
    /// True iff the pipe currently carries power (`state == "ON"`).
    pub on: bool,
    /// `Some("[E] toggle")`-style hint, present only when this same object is
    /// ALSO the one [`faced_context`] would report.
    pub verb: Option<&'static str>,
    /// Index into `PuzzleState::pipes` this label names -- stable for the
    /// lifetime of one loaded level's [`PuzzleState`], which is what
    /// [`ProximityTracker`] latches onto for hysteresis.
    pub pipe: usize,
    anchor_col: f32,
    anchor_row: f32,
    /// Euclidean distance from the camera to the switch cell's centre, in
    /// world cells -- kept only to compare candidates against each other in
    /// [`nearest_candidate`]; not part of the paint contract.
    dist: f32,
}

/// The forward probe [`faced_switch`] runs, reduced to "which pipe (if any)
/// is faced right now" -- the same rule [`faced_context`] uses, duplicated
/// here (four lines) rather than threaded through as a shared helper, so
/// this addition cannot change `faced_context`'s existing, already-tested
/// behaviour by so much as a refactor.
fn faced_pipe_index(cam: &CameraPose, puzzle: &PuzzleState) -> Option<usize> {
    let (x, y, z, active) = faced_switch(cam, puzzle);
    if !active {
        return None;
    }
    puzzle.pipe_at_switch(x, y, z)
}

/// What `E` will do for a pipe in state `on` right now -- the exact rule
/// [`faced_context`] uses for [`ContextLine::hint`], duplicated so the
/// proximity label can show identical wording when the labelled object is
/// also faced.
fn hint_for(on: bool, puzzle: &PuzzleState) -> &'static str {
    let refused = !on && puzzle.has_power_budget() && !puzzle.power_available();
    if refused {
        HINT_REFUSED
    } else {
        HINT_TOGGLE
    }
}

/// The pipe's endpoint-name title -- [`faced_context`]'s own fallback rule
/// (`R \u{2192} MIX` / `PIPE #<index>`), duplicated (not shared) for the
/// same reason as [`faced_pipe_index`].
fn pipe_title(idx: usize, pipe: &Pipe, level: Option<&Level>) -> String {
    let node_name = |i: usize| -> Option<&str> {
        level
            .and_then(|lvl| lvl.nodes.get(i))
            .map(|n| n.name.as_str())
            .filter(|n| !n.is_empty())
    };
    match (node_name(pipe.from), node_name(pipe.to)) {
        (Some(f), Some(t)) => format!("{f} \u{2192} {t}"),
        (Some(f), None) => format!("{f} \u{2192} ?"),
        (None, Some(t)) => format!("? \u{2192} {t}"),
        (None, None) => format!("PIPE #{idx}"),
    }
}

/// Is the framebuffer cell nearest `point`'s projected pixel already holding
/// something closer than `point` itself? A blank/undrawn cell defaults to
/// [`re_core::Cell::depth`] `INFINITY` (nothing there yet), which never
/// occludes. A point that projects off the frame (negative column/row) has
/// no depth cell to compare against and is treated as unoccluded too -- its
/// screen anchor gets clamped separately for painting.
fn is_occluded(fb: &Framebuffer, point: &ScreenPoint) -> bool {
    if point.col < 0.0 || point.row < 0.0 {
        return false;
    }
    let (c, r) = (point.col.round() as usize, point.row.round() as usize);
    match fb.get(c, r) {
        Some(cell) => cell.depth + PROXIMITY_OCCLUSION_SLACK < point.depth,
        None => false,
    }
}

/// Build the label for pipe `idx` at world switch cell `coord`, or `None`
/// when it fails the range, cone, projection, or occlusion test. The one
/// evaluation both [`nearest_candidate`] (sweeps every switchable pipe) and
/// [`ProximityTracker::resolve`] (re-checks last frame's single pick first)
/// call -- so "valid" means exactly the same thing in both places.
fn evaluate_candidate(
    idx: usize,
    coord: Coord3,
    cam: &CameraPose,
    puzzle: &PuzzleState,
    level: Option<&Level>,
    fb: &Framebuffer,
    fov: &Fov,
) -> Option<ProximityLabel> {
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return None;
    }

    let center = Vec3::new(
        coord.0 as f32 + 0.5,
        coord.1 as f32 + 0.5,
        coord.2 as f32 + 0.5,
    );
    let delta = center - cam.pos;
    let dist = delta.length();
    if dist > PROXIMITY_RANGE_CELLS || dist <= f32::EPSILON {
        return None;
    }

    let (fdx, fdy) = cam.dir();
    let horiz_len = (delta.x * delta.x + delta.y * delta.y).sqrt();
    if horiz_len > f32::EPSILON {
        let cos_angle = (delta.x * fdx + delta.y * fdy) / horiz_len;
        if cos_angle < PROXIMITY_HALF_ANGLE_DEG.to_radians().cos() {
            return None;
        }
    }

    let point = world_to_screen(cam, fov, cols, rows, center)?;
    if is_occluded(fb, &point) {
        return None;
    }

    let pipe = puzzle.pipes.get(idx)?;
    let on = pipe.is_on();
    let verb = if faced_pipe_index(cam, puzzle) == Some(idx) {
        Some(hint_for(on, puzzle))
    } else {
        None
    };
    Some(ProximityLabel {
        category: CATEGORY_SWITCH,
        title: pipe_title(idx, pipe, level),
        state: if on { "ON" } else { "OFF" },
        on,
        verb,
        pipe: idx,
        anchor_col: point.col.clamp(0.0, cols.saturating_sub(1) as f32),
        anchor_row: point
            .row
            .clamp(PROXIMITY_MIN_ROW as f32, rows.saturating_sub(1) as f32),
        dist,
    })
}

/// Sweep every switchable pipe in `puzzle`, returning the label for the
/// NEAREST one that passes [`evaluate_candidate`] (range, cone, occlusion),
/// or `None` when nothing qualifies. Stateless -- see [`ProximityTracker`]
/// for the hysteresis a real per-frame HUD wants on top of this.
pub fn nearest_candidate(
    cam: &CameraPose,
    puzzle: &PuzzleState,
    level: Option<&Level>,
    fb: &Framebuffer,
    fov: &Fov,
) -> Option<ProximityLabel> {
    let mut best: Option<ProximityLabel> = None;
    for (idx, pipe) in puzzle.pipes.iter().enumerate() {
        let Some(coord) = pipe.switch_coord() else {
            continue;
        };
        let Some(label) = evaluate_candidate(idx, coord, cam, puzzle, level, fb, fov) else {
            continue;
        };
        let replace = match &best {
            None => true,
            Some(b) => label.dist < b.dist,
        };
        if replace {
            best = Some(label);
        }
    }
    best
}

/// Per-level latch that keeps the SAME proximity label across frames as long
/// as it stays valid -- "stable ... hysteresis so the label doesn't flicker
/// between two candidates". Re-checks the previously-picked pipe FIRST; only
/// falls through to [`nearest_candidate`]'s fresh sweep when the previous
/// pick is gone (out of range/cone, occluded, or the pipe index no longer
/// exists).
///
/// **Reset on level (re)load.** `last` is a raw index into
/// `PuzzleState::pipes`, which is only stable for the lifetime of ONE loaded
/// level's puzzle. A host that keeps this across a level reload must replace
/// it with [`ProximityTracker::default`] at load time, exactly as other
/// per-level HUD state does.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct ProximityTracker {
    last: Option<usize>,
}

impl ProximityTracker {
    /// Resolve this frame's label: keep the previous pick if it is still
    /// valid, else re-sweep for the nearest one. Updates `self` in place.
    pub fn resolve(
        &mut self,
        cam: &CameraPose,
        puzzle: &PuzzleState,
        level: Option<&Level>,
        fb: &Framebuffer,
        fov: &Fov,
    ) -> Option<ProximityLabel> {
        if let Some(prev) = self.last {
            if let Some(coord) = puzzle.pipes.get(prev).and_then(Pipe::switch_coord) {
                if let Some(label) = evaluate_candidate(prev, coord, cam, puzzle, level, fb, fov) {
                    return Some(label);
                }
            }
        }
        let resolved = nearest_candidate(cam, puzzle, level, fb, fov);
        self.last = resolved.as_ref().map(|l| l.pipe);
        resolved
    }
}

/// Paint `label` at its own anchored screen position (as opposed to
/// [`paint_context_line`], which always paints at the fixed crosshair row).
/// Backed by the same dim strip as the faced-only line, for the same
/// wall-legibility reason. `palette` supplies the host's own title/background
/// colours (see [`HudPalette`]).
pub fn draw_proximity_label(fb: &mut Framebuffer, label: &ProximityLabel, palette: HudPalette) {
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let row = (label.anchor_row.round() as usize).min(rows - 1);

    let state_fg = if label.on { CTX_FG_ON } else { CTX_FG_OFF };
    let mut segments: Vec<(String, u32)> = vec![
        (label.category.to_string(), CTX_FG_CATEGORY),
        ("  ".to_string(), palette.fg),
        (label.title.clone(), palette.fg),
        ("  ".to_string(), palette.fg),
        (label.state.to_string(), state_fg),
    ];
    if let Some(v) = label.verb {
        segments.push((format!("  \u{00B7}  {v}"), CTX_FG_HINT));
    }
    let total_width: usize = segments.iter().map(|(s, _)| s.chars().count()).sum();
    let anchor_col = label.anchor_col.round() as usize;
    let x0 = anchor_col
        .saturating_sub(total_width / 2)
        .min(cols.saturating_sub(total_width));
    paint_backing_strip(fb, x0, x0 + total_width, row, palette.bg);

    let mut x = x0;
    for (text, fg) in &segments {
        put_str(fb, x, row, text, *fg, palette.bg);
        x += text.chars().count();
    }
}

/// The call a host's play HUD makes in place of [`draw_faced_context`] (see
/// this module's outer doc): resolve this frame's proximity label through
/// `tracker` and paint it. A no-op -- byte-identical `fb` -- when nothing
/// qualifies. `palette` supplies the host's own title/background colours (see
/// [`HudPalette`]).
pub fn draw_proximity_context(
    fb: &mut Framebuffer,
    cam: &CameraPose,
    puzzle: &PuzzleState,
    level: Option<&Level>,
    tracker: &mut ProximityTracker,
    fov: &Fov,
    palette: HudPalette,
) {
    if let Some(label) = tracker.resolve(cam, puzzle, level, fb, fov) {
        draw_proximity_label(fb, &label, palette);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::{Level, Node, PipeState};
    use re_puzzle::{Pipe, PuzzleNode, PuzzleState};
    use re_world::CameraPose;

    /// The colours these tests paint with — arbitrary but fixed, matching
    /// what any host would pass via [`HudPalette`].
    const TEST_PALETTE: HudPalette = HudPalette {
        fg: 0x00C8_F0FF,
        bg: 0x0000_0608,
    };

    /// The cell a camera at `(0.5, 0.5, 0.0)` facing `yaw = 0.0` lands on:
    /// `dir() == (sin 0, cos 0) == (0.0, 1.0)`, so `(floor(0.5+0), floor(0.5+1),
    /// floor(0.0)) == (0, 1, 0)`.
    const FACED_CELL: (i32, i32, i32) = (0, 1, 0);

    /// A minimal two-node, one-pipe puzzle with a switch at [`FACED_CELL`],
    /// pipe OFF, matching the shape [`faced_switch`]'s own tests use
    /// (`pipe_at_switch` keyed on the switch cell).
    fn puzzle_with_switch(on: bool) -> PuzzleState {
        PuzzleState {
            nodes: vec![
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 1,
                    is_input: false,
                    is_output: true,
                    activation: 0,
                },
            ],
            pipes: vec![Pipe {
                from: 0,
                to: 1,
                state: if on { PipeState::On } else { PipeState::Off },
                path: vec![FACED_CELL],
                switch_cell: Some(0),
            }],
            ..Default::default()
        }
    }

    /// Camera at `(0.5, 0.5, 0.0)` facing `yaw = 0.0`, landing on
    /// [`FACED_CELL`] (see its own doc comment for the arithmetic).
    fn facing_camera() -> CameraPose {
        CameraPose {
            pos: re_core::Vec3::new(0.5, 0.5, 0.0),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn level_with_names(from: &str, to: &str) -> Level {
        Level {
            nodes: vec![
                Node {
                    name: from.to_string(),
                    ..Default::default()
                },
                Node {
                    name: to.to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn far_away_camera() -> CameraPose {
        CameraPose {
            pos: re_core::Vec3::new(10.5, 10.5, 0.0), // nowhere near the pipe
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    #[test]
    fn nothing_faced_is_none() {
        let puzzle = puzzle_with_switch(false);
        let cam = far_away_camera();
        assert!(faced_context(&cam, &puzzle, None).is_none());
    }

    #[test]
    fn faced_switch_off_reports_off_and_toggle_hint() {
        let puzzle = puzzle_with_switch(false);
        let cam = facing_camera();
        let ctx = faced_context(&cam, &puzzle, None).expect("switch is faced");
        assert_eq!(ctx.category, CATEGORY_SWITCH);
        assert_eq!(ctx.state, "OFF");
        assert!(!ctx.on);
        assert_eq!(ctx.hint, HINT_TOGGLE);
        assert_eq!(ctx.title, "PIPE #0", "no level ⇒ kind+id fallback");
    }

    #[test]
    fn faced_switch_on_reports_on() {
        let puzzle = puzzle_with_switch(true);
        let cam = facing_camera();
        let ctx = faced_context(&cam, &puzzle, None).expect("switch is faced");
        assert_eq!(ctx.state, "ON");
        assert!(ctx.on);
        assert_eq!(ctx.hint, HINT_TOGGLE, "an ON switch is never power-refused");
    }

    #[test]
    fn title_uses_real_node_names_when_the_level_has_them() {
        let puzzle = puzzle_with_switch(false);
        let cam = facing_camera();
        let level = level_with_names("R", "MIX");
        let ctx = faced_context(&cam, &puzzle, Some(&level)).expect("switch is faced");
        assert_eq!(ctx.title, "R \u{2192} MIX");
    }

    #[test]
    fn title_falls_back_per_missing_endpoint_name() {
        let puzzle = puzzle_with_switch(false);
        let cam = facing_camera();
        let level = level_with_names("R", "");
        let ctx = faced_context(&cam, &puzzle, Some(&level)).expect("switch is faced");
        assert_eq!(ctx.title, "R \u{2192} ?");
    }

    #[test]
    fn power_full_and_off_reports_refused_hint() {
        let mut puzzle = puzzle_with_switch(false);
        puzzle.power_budget = 1;
        // Add a second, already-ON pipe elsewhere so the budget is AT the cap
        // (power_used == power_budget) while the FACED pipe is still OFF.
        puzzle.pipes.push(Pipe {
            from: 0,
            to: 1,
            state: PipeState::On,
            path: vec![(9, 9, 9)],
            switch_cell: Some(0),
        });
        assert_eq!(puzzle.power_used(), 1);
        assert!(!puzzle.power_available());
        let cam = facing_camera();
        let ctx = faced_context(&cam, &puzzle, None).expect("switch is faced");
        assert_eq!(ctx.state, "OFF");
        assert_eq!(ctx.hint, HINT_REFUSED);
    }

    #[test]
    fn draw_faced_context_is_a_noop_when_idle() {
        let puzzle = puzzle_with_switch(false);
        let cam = far_away_camera();
        let mut fb = Framebuffer::new(80, 24);
        let before = fb.cells().to_vec();
        draw_faced_context(&mut fb, &cam, &puzzle, None, TEST_PALETTE);
        assert_eq!(fb.cells(), before.as_slice(), "idle reticle paints nothing");
    }

    #[test]
    fn draw_faced_context_paints_something_when_active() {
        let puzzle = puzzle_with_switch(true);
        let cam = facing_camera();
        let mut fb = Framebuffer::new(80, 24);
        let before = fb.cells().to_vec();
        draw_faced_context(&mut fb, &cam, &puzzle, None, TEST_PALETTE);
        assert_ne!(
            fb.cells(),
            before.as_slice(),
            "an active context line must paint"
        );
    }

    #[test]
    fn the_backing_strip_darkens_padding_cells_past_the_text_span() {
        // A wide frame so the strip's 1-cell padding lands well inside it on
        // both sides, and starts from a NON-blank, NON-bg cell so we can tell
        // "the strip touched this cell" apart from "it was already this way".
        let puzzle = puzzle_with_switch(true);
        let cam = facing_camera();
        let mut fb = Framebuffer::new(120, 24);
        let busy = re_core::Cell {
            glyph: u32::from('#'),
            fg: 0x00FF_00FF,
            bg: 0x00FF_00FF,
            ..Default::default()
        };
        let row = fb.rows() / 2 + 2;
        for c in 0..fb.cols() {
            fb.set(c, row, busy);
        }

        draw_faced_context(&mut fb, &cam, &puzzle, None, TEST_PALETTE);

        // The STRIP (backing + text) is the whole run of cells no longer the busy
        // wall colour; the INK (actual glyphs) is the narrower run inside it whose
        // glyph is not a blank space. `paint_backing_strip` runs BEFORE the text
        // segments and is `STRIP_PADDING` cells wider on each side, so the gap
        // between "strip starts" and "ink starts" (and the mirror image on the
        // right) must be exactly `STRIP_PADDING`.
        let is_untouched = |c: usize| fb.get(c, row).unwrap().bg == 0x00FF_00FF;
        let strip_start = (0..fb.cols())
            .find(|&c| !is_untouched(c))
            .expect("something painted");
        let strip_end = 1
            + (0..fb.cols())
                .rev()
                .find(|&c| !is_untouched(c))
                .expect("something painted");
        assert!(
            strip_start > 0,
            "the strip must not run off the left edge in this wide frame"
        );
        assert!(
            strip_end < fb.cols(),
            "the strip must not run off the right edge in this wide frame"
        );

        let is_ink = |c: usize| fb.get(c, row).unwrap().glyph != u32::from(' ');
        let ink_start = (strip_start..strip_end)
            .find(|&c| is_ink(c))
            .expect("the context line renders visible glyphs");
        let ink_end = 1
            + (strip_start..strip_end)
                .rev()
                .find(|&c| is_ink(c))
                .expect("the context line renders visible glyphs");

        assert_eq!(
            ink_start - strip_start,
            STRIP_PADDING,
            "left padding must be exactly STRIP_PADDING cells wide"
        );
        assert_eq!(
            strip_end - ink_end,
            STRIP_PADDING,
            "right padding must be exactly STRIP_PADDING cells wide"
        );

        // Every padding-only cell must be repainted to the flat HUD background,
        // not merely "no longer '#'" (e.g. some other stray colour).
        for c in strip_start..ink_start {
            assert_eq!(
                fb.get(c, row).unwrap().bg,
                TEST_PALETTE.bg,
                "left padding cell {c} must be the HUD background"
            );
        }
        for c in ink_end..strip_end {
            assert_eq!(
                fb.get(c, row).unwrap().bg,
                TEST_PALETTE.bg,
                "right padding cell {c} must be the HUD background"
            );
        }
    }

    // ── Proximity-picked label ───────────────────────

    /// Two switchable pipes 1 cell apart along +Y: pipe0's switch at y=1,
    /// pipe1's at y=2 -- close enough that several cameras below put both
    /// inside [`PROXIMITY_RANGE_CELLS`] at once.
    fn two_switch_puzzle() -> PuzzleState {
        PuzzleState {
            nodes: vec![
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 1,
                    is_input: false,
                    is_output: true,
                    activation: 0,
                },
            ],
            pipes: vec![
                Pipe {
                    from: 0,
                    to: 1,
                    state: PipeState::Off,
                    path: vec![(0, 1, 0)],
                    switch_cell: Some(0),
                },
                Pipe {
                    from: 0,
                    to: 1,
                    state: PipeState::Off,
                    path: vec![(0, 2, 0)],
                    switch_cell: Some(0),
                },
            ],
            ..Default::default()
        }
    }

    fn cam_at(x: f32, y: f32, z: f32) -> CameraPose {
        CameraPose {
            pos: re_core::Vec3::new(x, y, z),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    #[test]
    fn nearest_in_cone_wins_over_a_farther_candidate_in_the_same_cone() {
        let puzzle = two_switch_puzzle();
        // pipe0 (y=1) is 0.5 cells ahead, pipe1 (y=2) is 1.5 -- both valid,
        // pipe0 strictly nearer.
        let cam = cam_at(0.5, 1.0, 0.5);
        let fb = Framebuffer::new(80, 24);
        let label = nearest_candidate(&cam, &puzzle, None, &fb, &Fov::default())
            .expect("a candidate is in range");
        assert_eq!(label.pipe, 0, "the nearer candidate must win");
    }

    #[test]
    fn candidate_out_of_range_is_rejected() {
        let puzzle = two_switch_puzzle();
        // pipe0 (y=1) is 3.5 cells away, pipe1 (y=2) is 4.5 -- both past
        // PROXIMITY_RANGE_CELLS (2.0), though both are dead ahead (in cone).
        let cam = cam_at(0.5, -2.5, 0.5);
        let fb = Framebuffer::new(80, 24);
        assert!(nearest_candidate(&cam, &puzzle, None, &fb, &Fov::default()).is_none());
    }

    #[test]
    fn candidate_out_of_cone_is_rejected() {
        let puzzle = two_switch_puzzle();
        // Forward is +Y; both switches (y=1, y=2) sit BEHIND this camera
        // (y=3.0), so both fail the \u{00B1}35\u{00B0} cone regardless of
        // being within PROXIMITY_RANGE_CELLS.
        let cam = cam_at(0.5, 3.0, 0.5);
        let fb = Framebuffer::new(80, 24);
        assert!(nearest_candidate(&cam, &puzzle, None, &fb, &Fov::default()).is_none());
    }

    #[test]
    fn occluded_candidate_is_rejected() {
        // A dedicated fixture at a realistic eye height (`re_world::STANDING_EYE_Z`
        // convention: eye above the floor by `PLAYER_EYE_HEIGHT`), with the switch
        // at the SAME world height as the eye, so the projected row lands near the
        // horizon regardless of FOV -- unlike the z=0.0 "floor-level" camera the
        // facing-only fixtures above use (fine for their `.floor()`-only cell probe,
        // not for a real screen projection).
        let puzzle = PuzzleState {
            nodes: vec![
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 1,
                    is_input: false,
                    is_output: true,
                    activation: 0,
                },
            ],
            pipes: vec![Pipe {
                from: 0,
                to: 1,
                state: PipeState::Off,
                path: vec![(0, 1, 1)],
                switch_cell: Some(0),
            }],
            ..Default::default()
        };
        let cam = CameraPose {
            pos: re_core::Vec3::new(0.5, 0.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let fov = Fov::default();
        let point = world_to_screen(&cam, &fov, 80, 24, re_core::Vec3::new(0.5, 1.5, 1.5))
            .expect("the switch projects in front of this camera");
        assert!(
            point.col >= 0.0
                && point.row >= 0.0
                && (point.col as usize) < 80
                && (point.row as usize) < 24,
            "premise: the switch must project on-screen for this test ({point:?})"
        );

        let clear_fb = Framebuffer::new(80, 24);
        assert!(
            nearest_candidate(&cam, &puzzle, None, &clear_fb, &fov).is_some(),
            "unoccluded baseline must find the candidate"
        );

        let mut blocked_fb = Framebuffer::new(80, 24);
        let (c, r) = (point.col.round() as usize, point.row.round() as usize);
        blocked_fb.set(
            c,
            r,
            Cell {
                depth: (point.depth * 0.1).max(0.01),
                ..Default::default()
            },
        );
        assert!(
            nearest_candidate(&cam, &puzzle, None, &blocked_fb, &fov).is_none(),
            "a closer cell at the same pixel must occlude the candidate"
        );
    }

    #[test]
    fn faced_candidate_shows_the_toggle_verb() {
        let puzzle = puzzle_with_switch(false);
        let cam = facing_camera();
        let fb = Framebuffer::new(80, 24);
        let label = nearest_candidate(&cam, &puzzle, None, &fb, &Fov::default())
            .expect("faced switch is also nearest");
        assert_eq!(label.verb, Some(HINT_TOGGLE));
    }

    #[test]
    fn unfaced_candidate_shows_no_verb() {
        // A single switch two cells directly ahead: within range/cone, but at
        // a different (x, y) than `faced_switch`'s one-cell-ahead probe, so
        // it is never the FACED object.
        let puzzle = PuzzleState {
            nodes: vec![
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 1,
                    is_input: false,
                    is_output: true,
                    activation: 0,
                },
            ],
            pipes: vec![Pipe {
                from: 0,
                to: 1,
                state: PipeState::Off,
                path: vec![(0, 2, 0)],
                switch_cell: Some(0),
            }],
            ..Default::default()
        };
        // forward probe lands on (0, 1, *), not (0, 2, *)
        let cam = cam_at(0.5, 0.6, 0.5);
        let fb = Framebuffer::new(80, 24);
        let label = nearest_candidate(&cam, &puzzle, None, &fb, &Fov::default())
            .expect("in range and in cone");
        assert_eq!(label.verb, None, "not the faced object \u{21D2} no verb");
    }

    #[test]
    fn hysteresis_keeps_the_previous_pick_while_it_stays_valid() {
        let puzzle = two_switch_puzzle();
        let fb = Framebuffer::new(80, 24);
        let mut tracker = ProximityTracker::default();

        // Frame 1: only pipe1 (y=2) is ahead of the camera; pipe0 (y=1) is
        // behind it.
        let cam1 = cam_at(0.5, 1.7, 0.5);
        let label1 = tracker
            .resolve(&cam1, &puzzle, None, &fb, &Fov::default())
            .expect("pipe1 is in range and ahead");
        assert_eq!(label1.pipe, 1);

        // Frame 2: camera moved so BOTH are ahead and pipe0 is now nearer --
        // a stateless sweep would flip to pipe0, but the tracker must not.
        let cam2 = cam_at(0.5, 1.0, 0.5);
        assert_eq!(
            nearest_candidate(&cam2, &puzzle, None, &fb, &Fov::default()).map(|l| l.pipe),
            Some(0),
            "sanity: the stateless sweep really would prefer pipe0 here"
        );
        let label2 = tracker
            .resolve(&cam2, &puzzle, None, &fb, &Fov::default())
            .expect("pipe1 is still valid");
        assert_eq!(
            label2.pipe, 1,
            "the tracker must not flicker to the marginally nearer pipe0"
        );

        // Frame 3: camera moved far enough that pipe1 falls out of range,
        // leaving only pipe0 -- NOW the tracker must switch.
        let cam3 = cam_at(0.5, -0.4, 0.5);
        let label3 = tracker
            .resolve(&cam3, &puzzle, None, &fb, &Fov::default())
            .expect("pipe0 is still in range");
        assert_eq!(
            label3.pipe, 0,
            "must switch once the latched pick is truly gone"
        );
    }

    #[test]
    fn draw_proximity_label_paints_something_when_active() {
        let puzzle = puzzle_with_switch(true);
        let cam = facing_camera();
        let fb_probe = Framebuffer::new(80, 24);
        let label = nearest_candidate(&cam, &puzzle, None, &fb_probe, &Fov::default())
            .expect("faced switch is nearest");

        let mut fb = Framebuffer::new(80, 24);
        let before = fb.cells().to_vec();
        draw_proximity_label(&mut fb, &label, TEST_PALETTE);
        assert_ne!(fb.cells(), before.as_slice(), "an active label must paint");
    }

    #[test]
    fn draw_proximity_context_is_a_noop_when_idle() {
        let puzzle = puzzle_with_switch(false);
        let cam = far_away_camera();
        let mut tracker = ProximityTracker::default();
        let mut fb = Framebuffer::new(80, 24);
        let before = fb.cells().to_vec();
        draw_proximity_context(
            &mut fb,
            &cam,
            &puzzle,
            None,
            &mut tracker,
            &Fov::default(),
            TEST_PALETTE,
        );
        assert_eq!(
            fb.cells(),
            before.as_slice(),
            "nothing in range paints nothing"
        );
    }

    /// End-to-end evidence (not just synthetic fixtures): the real shipped
    /// `levels/01.lvl` (archive/example fixture, not a campaign level) switch
    /// cell (`[pipe] switch_at (6,3,2)`, the same coordinate
    /// [`crate::hud_probe`]'s own fixture tests anchor on) is picked up as a
    /// proximity candidate and paints a label onto an actual rendered frame.
    #[test]
    fn draws_a_proximity_label_near_the_switch_on_a_real_shipped_level() {
        const SWITCH: (i32, i32, i32) = (6, 3, 2);
        const COLS: usize = 100;
        const ROWS: usize = 32;

        let lv = re_level::load("../../levels/01.lvl").expect("01.lvl loads");
        let puzzle = PuzzleState::from_level(&lv).expect("01.lvl's graph must build");

        // One cell off to the +X side of the switch, looking back at it --
        // within PROXIMITY_RANGE_CELLS regardless of exact aim.
        let cam = CameraPose {
            pos: re_core::Vec3::new(
                SWITCH.0 as f32 + 1.5,
                SWITCH.1 as f32 + 0.5,
                SWITCH.2 as f32 + 0.5,
            ),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };

        let map = lv.build_map();
        let registry = re_render::registry_for_map(&map);
        let mut fb = Framebuffer::new(COLS, ROWS);
        let fov = Fov::default();
        re_render::composite::composite_dump_once(&map, &cam, &registry, &fov, &[], &mut fb);

        let label = nearest_candidate(&cam, &puzzle, Some(&lv), &fb, &fov)
            .expect("01.lvl's switch must be picked up as a proximity candidate");
        assert_eq!(label.category, CATEGORY_SWITCH);

        draw_proximity_label(&mut fb, &label, TEST_PALETTE);

        // Text evidence: read the label's own painted row back off the frame.
        let row = (label.anchor_row.round() as usize).min(ROWS - 1);
        let text: String = (0..fb.cols())
            .map(|c| char::from_u32(fb.get(c, row).unwrap().glyph).unwrap_or(' '))
            .collect();
        println!("L1 proximity label (row {row}): {:?}", text.trim());
        assert!(
            text.contains(label.category) && text.contains(&label.title),
            "the painted row must carry the category and title: {text:?}"
        );
    }
}
