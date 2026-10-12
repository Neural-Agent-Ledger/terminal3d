#![warn(missing_docs)]
//! `re_lvlcheck` — a STATIC LEVEL-CONTENT linter for terminal3d `.lvl` files.
//!
//! This is the SUBSTANCE checker, not a renderer. It never draws a frame or
//! touches a terminal. For each level it analyses the *essence* — the elements,
//! the puzzle network's solvability, the physical reachability of its switches,
//! and a handful of layout-coherence heuristics — and reports PASS / WARN /
//! FAIL with specifics. It exists to replace eyeballing a rendered dump to judge
//! whether a level is a coherent, playable puzzle.
//!
//! The analysis reuses the *real* runtime: it builds the puzzle network through
//! [`re_puzzle::PuzzleState::from_level`] and solves it with the real
//! [`re_puzzle::solve_brute_force`] forward pass — it does not reimplement the
//! network math. Physical reachability is checked against the real
//! [`re_world::Map`] built by [`re_level::Level::build_map`] using the engine's
//! own jump/gravity constants.
//!
//! Pure, deterministic, allocation-light. No Bevy `App`, no rendering, no I/O
//! beyond the level file the caller hands in.

use std::path::Path;

use re_level::Level;
use re_puzzle::{solve_brute_force, PuzzleState, Solvability};
use re_world::{CellKind, Map, GRAVITY, JUMP_IMPULSE};

/// A single check's severity. Ordered so the worst across a report can be taken
/// with `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The check passed cleanly.
    Pass,
    /// A non-fatal concern — the level still works but reads poorly or has a
    /// soft defect worth flagging.
    Warn,
    /// A hard defect — the level is not a coherent, playable puzzle as-is.
    Fail,
}

impl Severity {
    /// A short uppercase tag for report lines.
    pub fn tag(self) -> &'static str {
        match self {
            Severity::Pass => "PASS",
            Severity::Warn => "WARN",
            Severity::Fail => "FAIL",
        }
    }
}

// ---------------------------------------------------------------------------
// WHERE a finding is
// ---------------------------------------------------------------------------
//
// A report line that says "3 of 12 switches are unreachable" is only half a
// diagnostic: the reader still has to find them. `ts_editor` shows this report
// verbatim in a drawer it cannot navigate, and `lvlcheck` prints it into a
// terminal that cannot jump. [`Location`] is the missing half.
//
// Three variants, because the linter's findings genuinely address three
// different kinds of thing and no two of them collapse into one:
//
//   * `Cell`       - the common case, and the one an editor cursor wants. The
//                    linter resolves it ITSELF (it holds the `Level`), so a
//                    consumer never looks a node name up to find its cube.
//   * `Element`    - the subject is an authored CLAUSE with no cell of its own:
//                    a `[texture]` whose file is missing, a `[pipe]` whose
//                    endpoint index is out of range (so it has no resolvable
//                    node coordinates at all).
//   * `SourceLine` - the subject was located by the PARSER while reading text
//                    and has no world position even in principle: an unknown
//                    key, a parse error.
//
// ## Why a `Vec`, not an `Option`
//
// One location per check is the wrong shape for THIS linter: a `Check` is one
// RULE, not one finding, and the rules that most need locating are exactly the
// ones that aggregate. `reachable` names every unreachable switch in one line;
// `emit_parts` names every eaten reward cell; `notes` names every mis-placed
// placard. An `Option` would point at the first and silently drop the rest -
// which is the same "it is in the report and nobody can get to it" defect this
// field exists to close, just one level down.
//
// So: [`Check::locations`] is a `Vec`, in the SAME ORDER as the offenders named
// in `detail` (an invariant the tests pin), and EMPTY is the first-class honest
// answer for a level-wide rule. [`Check::location`] gives back the `Option`
// ergonomics for a caller that only wants "the" place.

/// Which authored collection a [`Location::Element`] indexes into.
///
/// Only the kinds the linter can actually point at today. A finding whose
/// subject has a world position uses [`Location::Cell`] instead - the linter
/// resolves node/switch/placard coordinates itself rather than making the
/// consumer do the lookup, so there is deliberately no `Node` variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    /// A `[pipe A->B]` clause, indexed into `Level::pipes` in parse order.
    Pipe,
    /// A `[texture NAME]` clause, indexed into `Level::textures`.
    Texture,
    /// A `[sky]` image slot: `0` cylinder, `1` cap_top, `2` cap_bottom.
    Sky,
    /// A `[vox NAME]` clause, indexed into `Level::voxes`.
    Vox,
    /// An `[inset]` clause, indexed into `Level::overlay.insets`.
    Inset,
}

impl ElementKind {
    /// A short lowercase tag for report lines.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Pipe => "pipe",
            Self::Texture => "texture",
            Self::Sky => "sky",
            Self::Vox => "vox",
            Self::Inset => "inset",
        }
    }
}

impl From<re_level::AssetKind> for ElementKind {
    fn from(k: re_level::AssetKind) -> Self {
        match k {
            re_level::AssetKind::Texture => Self::Texture,
            re_level::AssetKind::Sky => Self::Sky,
            re_level::AssetKind::Vox => Self::Vox,
            re_level::AssetKind::Inset => Self::Inset,
        }
    }
}

/// Where one finding is. See the section comment above for the three-variant
/// rationale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    /// A cell in the level's own integer world coordinates - put the cursor here.
    Cell {
        /// World-space X.
        x: i32,
        /// World-space Y.
        y: i32,
        /// World-space Z.
        z: i32,
    },
    /// An authored clause with no cell of its own, addressed the way the `.lvl`
    /// addresses it: its collection, its index in parse order, and the name a
    /// human reads (`R->output`, `[texture plate]`).
    Element {
        /// Which authored collection (`[node]`, `[pipe]`, …) this is in.
        kind: ElementKind,
        /// Index within that collection, in parse order.
        index: usize,
        /// The human-readable name the `.lvl` gives this element.
        name: String,
    },
    /// A 1-based line in the source `.lvl` text.
    SourceLine(usize),
}

impl Location {
    /// A cell location from the `(x, y, z)` triples the level data uses.
    #[must_use]
    pub const fn cell((x, y, z): (i32, i32, i32)) -> Self {
        Self::Cell { x, y, z }
    }

    /// An element location.
    #[must_use]
    pub fn element(kind: ElementKind, index: usize, name: impl Into<String>) -> Self {
        Self::Element {
            kind,
            index,
            name: name.into(),
        }
    }

    /// The world cell this location names, when it names one. `Element` and
    /// `SourceLine` return `None` - deliberately: they have no coordinate, and
    /// synthesising one would move a cursor somewhere arbitrary.
    #[must_use]
    pub const fn as_cell(&self) -> Option<(i32, i32, i32)> {
        match self {
            Self::Cell { x, y, z } => Some((*x, *y, *z)),
            _ => None,
        }
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cell { x, y, z } => write!(f, "({x},{y},{z})"),
            Self::Element { kind, index, name } => {
                write!(f, "{} #{index} '{name}'", kind.tag())
            }
            Self::SourceLine(n) => write!(f, "line {n}"),
        }
    }
}

/// One named check result with a human-readable detail line and, where the rule
/// knew where the problem was, the places it is.
#[derive(Clone, Debug)]
pub struct Check {
    /// The check's stable identifier (e.g. `"solvable"`, `"reachable"`), used as
    /// both the report's row label and a lookup key for tooling.
    pub name: &'static str,
    /// The worst outcome this check reached.
    pub severity: Severity,
    /// The human-readable finding line shown in the report.
    pub detail: String,
    /// Every offender this check NAMED, in the SAME ORDER `detail` names them.
    ///
    /// Empty when the rule is level-wide - an inventory census, a solvability
    /// verdict, a power-budget arithmetic - because there is no single object to
    /// put a cursor on and inventing one would send a browser somewhere
    /// arbitrary. Empty is a first-class answer, not a stub.
    ///
    /// Severity does not decide this: `detail` naming something specific does.
    /// A clean PASS names nothing and carries nothing, but the `thresholds` PASS
    /// that declares a DEAD-BY-DESIGN node names it precisely so a reviewer goes
    /// and looks, and so it is located too.
    pub locations: Vec<Location>,
}

impl Check {
    fn new(name: &'static str, severity: Severity, detail: impl Into<String>) -> Self {
        Check {
            name,
            severity,
            detail: detail.into(),
            locations: Vec::new(),
        }
    }

    /// Builder: attach one location.
    #[must_use]
    fn at(mut self, loc: Location) -> Self {
        self.locations.push(loc);
        self
    }

    /// Builder: attach every location, in `detail` order.
    #[must_use]
    fn at_all(mut self, locs: Vec<Location>) -> Self {
        self.locations = locs;
        self
    }

    /// The PRIMARY location - the first offender - or `None` for a level-wide
    /// finding. The `Option<Location>` view of [`Check::locations`].
    #[must_use]
    pub fn location(&self) -> Option<&Location> {
        self.locations.first()
    }
}

/// Counts of every element kind in a level (the inventory check payload).
#[derive(Clone, Copy, Debug, Default)]
pub struct Inventory {
    /// `[node]`s marked as an input.
    pub inputs: usize,
    /// `[node]`s marked as an output.
    pub outputs: usize,
    /// `[node]`s that are neither input nor output.
    pub hidden: usize,
    /// Total `[node]` count (inputs + outputs + hidden).
    pub nodes_total: usize,
    /// `[pipe]` count.
    pub pipes: usize,
    /// Pipes with a resolvable `switch_at`/`switch_cell`.
    pub switches: usize,
    /// `[part]` count.
    pub parts: usize,
    /// `[emit]` clause count.
    pub emits: usize,
    /// Wall decal count.
    pub decals: usize,
    /// `[note]` placard count.
    pub notes: usize,
    /// `[holo]` object count.
    pub holos: usize,
    /// Whether the level authors a `[door]`.
    pub has_door: bool,
    /// Whether the level authors a `[player]` spawn.
    pub has_player: bool,
    /// Whether the level authors a `[pedestal]`.
    pub has_pedestal: bool,
}

/// The solvability verdict in report-friendly form. A thin re-shaping of
/// [`re_puzzle::Solvability`] so the report layer need not depend on the puzzle
/// crate's enum directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveVerdict {
    /// The brute-force search found a win; carries the search's own numbers.
    Solvable {
        /// The fewest pipe toggles (from the authored start state) that win.
        min_toggles: u32,
        /// How many pipes the search considered switchable.
        switchable: usize,
    },
    /// The authored start state already wins with zero toggles.
    TriviallySolved,
    /// The brute-force search covered every reachable state and none won.
    Unsolvable,
    /// Too many switchable pipes to brute-force within the search budget.
    TooManyToBruteForce {
        /// How many pipes the search considered switchable.
        switchable: usize,
    },
    /// The network is malformed (a cycle / bad edge) so the puzzle could not
    /// even be built — reported as a structural failure upstream, surfaced here
    /// so the summary table stays complete.
    NetworkError,
}

/// One unreachable / off-path switch, for the reachability report.
#[derive(Clone, Debug)]
pub struct SwitchIssue {
    /// Index of the offending `[pipe]` in parse order.
    pub pipe_index: usize,
    /// The pipe's source node name.
    pub from: String,
    /// The pipe's destination node name.
    pub to: String,
    /// The switch cell. `(0, 0, 0)` when the pipe's `switch_cell` index runs off
    /// the end of its path and there IS no cell - read [`SwitchIssue::location`]
    /// instead of trusting this in that case.
    pub cell: (i32, i32, i32),
    /// Where to send a cursor: the switch cell normally, or the `[pipe]` clause
    /// itself when the pipe has no resolvable switch cell to point at.
    pub location: Location,
    /// Why the switch is flagged (unreachable, off-path, no resolvable cell, …).
    pub reason: String,
}

/// The full per-level report.
#[derive(Clone, Debug)]
pub struct LevelReport {
    /// The path or label the level was loaded from, as given to `check_path`.
    pub source: String,
    /// The `[meta] name` field.
    pub level_name: String,
    /// The `[meta] title` field.
    pub title: String,
    /// The `[room] size` field (x, y, z cells).
    pub size: (usize, usize, usize),
    /// `Some` once the level parsed; `None` if it failed to load (then `checks`
    /// holds a single FAIL with the parse error).
    pub inventory: Option<Inventory>,
    /// The brute-force solvability verdict, when the network built cleanly.
    pub solve: Option<SolveVerdict>,
    /// Whether every switch is physically reachable, when reachability could be
    /// checked.
    pub reachable: Option<bool>,
    /// Whether the level clears every layout-coherence heuristic, when checked.
    pub coherent: Option<bool>,
    /// How many external asset references (`[texture]`/`[sky]`/`[vox]`/
    /// `[inset]`) failed to resolve, read or decode. `0` when the level's files
    /// are all there — which is the normal case, so a consumer can branch on it
    /// without scraping the check's detail text. See `check_assets`.
    pub broken_assets: usize,
    /// Every check this report ran, in run order.
    pub checks: Vec<Check>,
    /// The condensed one-line coherence verdict ("COHERENT" / "NEEDS LAYOUT
    /// WORK (…)"), or a load-failure note.
    pub verdict: String,
}

impl LevelReport {
    /// The worst severity across every check — drives the process exit code.
    pub fn worst(&self) -> Severity {
        self.checks
            .iter()
            .map(|c| c.severity)
            .max()
            .unwrap_or(Severity::Pass)
    }

    /// Did any check FAIL?
    pub fn failed(&self) -> bool {
        self.worst() == Severity::Fail
    }

    /// Every place worth jumping to, in report order: `(index into `checks`,
    /// location)`, one entry per located offender.
    ///
    /// This is the issue-browser list - "next finding" / "previous finding" is a
    /// step through this vector. PASS checks are skipped even when they carry
    /// locations (the dead-by-design declaration is documentation, not a defect
    /// to be walked), so a browser over this vector only ever lands on something
    /// that wants fixing.
    #[must_use]
    pub fn jump_targets(&self) -> Vec<(usize, &Location)> {
        self.checks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.severity > Severity::Pass)
            .flat_map(|(i, c)| c.locations.iter().map(move |l| (i, l)))
            .collect()
    }
}

/// Load and lint a level file from disk, returning a full [`LevelReport`].
/// A load/parse error yields a report carrying a single FAIL check — it never
/// panics.
pub fn check_path(path: impl AsRef<Path>) -> LevelReport {
    let path = path.as_ref();
    let source = path.display().to_string();
    match re_level::load(path) {
        Ok(level) => check_level(&level, &source),
        // `ParseError` knows the line it choked on; `line == 0` is the "no line"
        // sentinel the reader-error path uses (the file could not be opened at
        // all), and that stays unlocated rather than pointing at a fictional
        // line 0.
        Err(e) => load_failure_report(&source, &e.to_string(), e.line),
    }
}

/// Lint an already-parsed level. The pure entry point (used by tests on
/// synthetic levels and by [`check_path`] after a successful load).
pub fn check_level(level: &Level, source: &str) -> LevelReport {
    let mut checks: Vec<Check> = Vec::new();

    // --- 1. Element inventory --------------------------------------------
    let inv = inventory(level);
    checks.push(check_inventory(&inv));

    // --- 2. Structural validity ------------------------------------------
    checks.extend(check_structure(level));

    // --- Build the puzzle network (also the DAG / bad-edge gate) ---------
    let net = PuzzleState::from_level(level);

    // --- 3. Solvability ---------------------------------------------------
    let (solve_verdict, solve_check) = check_solvability(&net);
    checks.push(solve_check);

    // --- 4. Reachability --------------------------------------------------
    let (reachable, reach_check) = check_reachability(level);
    checks.push(reach_check);

    // --- 5. Coherence -----------------------------------------------------
    let (coherent, verdict_line, coh_checks) = check_coherence(level);
    checks.extend(coh_checks);

    // --- 6. `[note]` placard placement ------------------------------------
    checks.push(check_notes(level));

    // --- 7. threshold sanity (dead / always-on nodes) ---------------------
    checks.push(check_thresholds(level));

    // --- 8. power budget (the simultaneously-ON pipe cap) -----------------
    checks.push(check_power_budget(level, &net));

    // --- 9. unknown keys the parser read and discarded ---------------------
    checks.push(check_unknown_keys(level));

    // --- 10. `[emit]` reward footprint vs the pipe network -----------------
    checks.push(check_emit_parts(level));

    // --- 11. external assets (`[texture]`/`[sky]`/`[vox]`/`[inset]`) --------
    let (broken_assets, asset_check) = check_assets(level);
    checks.push(asset_check);

    // --- 12. per-level lighting (`[light]` / `[atmosphere]`) ---------------
    checks.push(check_lighting(level));

    // --- 12b. can the surfaces receive those lights? --------------------
    checks.push(check_lit_surfaces(level));

    // --- 13. `[prop]` set dressing never touches the puzzle ----------------
    checks.push(props_rule::check_props(level, max_jump_height()));

    // --- 14. every node is still a node cell in the built map ------------
    checks.push(check_node_cells(level));

    // --- 15. the player can never be trapped (real-physics flood, both power
    //         states) -------------------------------------------------------
    checks.push(no_stuck::check_no_stuck(level));

    // --- 16. sound fields ----------------------------------------------------
    checks.push(fields_rule::check_fields(level));

    LevelReport {
        source: source.to_string(),
        level_name: level.name.clone(),
        title: level.title.clone(),
        size: level.size,
        inventory: Some(inv),
        solve: Some(solve_verdict),
        reachable: Some(reachable),
        coherent: Some(coherent),
        broken_assets,
        checks,
        verdict: verdict_line,
    }
}

// ===========================================================================
// 14. Node cells
// ===========================================================================

/// Every `[node]` must still be an `Object` cell in the built map.
///
/// The loader stamps each node as an `Object` whose `side_data` is the node INDEX,
/// then lays a dense `[cells]` grid over it. A grid that paints the node's cell as
/// anything else (`/` trace, `#` wall) keeps the index in `side_data` but changes the
/// kind, and on a geometry cell `side_data` is a 1-based COLOUR id. The node then
/// renders as whatever palette material has that number — a `[note]` placard's text
/// when the level has no authored palette, a `redacted` bar or a prop material when
/// it does — and which one it is shifts whenever the palette grows (adding a single
/// `[prop]` appends the level's materials). That is how L10's junction `J` read as
/// the researcher placard until a prop was added (set-dressing 2026-09-24, item 17).
///
/// FAIL when the index is non-zero (the node draws as an unrelated material); WARN
/// for node 0 (its tag is 0, so it only loses its node look).
fn check_node_cells(level: &Level) -> Check {
    if level.nodes.is_empty() {
        return Check::new("node_cells", Severity::Pass, "no nodes");
    }
    let map = level.build_map_base();
    let mut fails = Vec::new();
    let mut warns = Vec::new();
    for (i, n) in level.nodes.iter().enumerate() {
        let kind = map.cell_at(n.x, n.y, n.z);
        if kind == CellKind::Object {
            continue;
        }
        let tag = map.side_data_at(n.x, n.y, n.z);
        let msg = format!(
            "node {} at ({},{},{}) is a {kind:?} cell in the built map (painted over by [cells]); its side_data {tag} is read as a colour id, so it renders as palette material {tag} instead of a node — paint it 'O' in [cells]",
            n.name, n.x, n.y, n.z
        );
        if i > 0 && tag != 0 {
            fails.push((msg, (n.x, n.y, n.z)));
        } else {
            warns.push((msg, (n.x, n.y, n.z)));
        }
    }
    let (sev, list) = if !fails.is_empty() {
        (Severity::Fail, fails)
    } else if !warns.is_empty() {
        (Severity::Warn, warns)
    } else {
        return Check::new(
            "node_cells",
            Severity::Pass,
            format!("{} node cell(s) intact", level.nodes.len()),
        );
    };
    let detail = list
        .iter()
        .map(|(m, _)| m.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    Check::new("node_cells", sev, detail)
        .at_all(list.into_iter().map(|(_, c)| Location::cell(c)).collect())
}

// ===========================================================================
// 9. Unknown keys
// ===========================================================================

/// Report every `key = value` line whose key the parser did not recognize
/// ([`re_level::UnknownKey`]).
///
/// **WARN, never FAIL.** A stray key must never make one of 63 hand-authored
/// levels unloadable or block a save — the point is to make it VISIBLE, not to
/// stop the work. The editor shows this on every save (`ts_editor`'s validation
/// drawer) and `lvlcheck` prints it in CI.
///
/// Why this check exists at all: `re_level` used to discard an unrecognized key in
/// complete silence, which is the ENABLER for this project's signature defect —
/// a key that is authored, parsed, tested and never on screen. A mistyped key and
/// a correctly-spelled-but-unwired key look identical from the outside, and both
/// look exactly like "it works". Six such bugs were found and fixed one at a time;
/// every one of them was a one-second find if the parser had simply said something.
fn check_unknown_keys(level: &Level) -> Check {
    if level.unknown_keys.is_empty() {
        return Check::new("unknown_keys", Severity::Pass, "no unrecognized keys");
    }
    let list = level
        .unknown_keys
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    Check::new(
        "unknown_keys",
        Severity::Warn,
        format!(
            "{} key(s) parsed and DISCARDED — a typo, or a real key nobody wired up: {list}",
            level.unknown_keys.len()
        ),
    )
    // The parser recorded the LINE and nothing else: an unrecognized key has no
    // world position even in principle, so this is the one rule whose honest
    // location is a text coordinate.
    .at_all(
        level
            .unknown_keys
            .iter()
            .map(|k| Location::SourceLine(k.line))
            .collect(),
    )
}

// ===========================================================================
// 10. `[emit]` reward footprint vs the pipe network
// ===========================================================================
//
// The `[emit]` reward is the payoff: a `[part]` voxel stamp that MATERIALISES on
// the pedestal when its source node fires. Its silhouette is authored intent —
// L7 is a flower, L8 is a cat — and the shape IS the reward.
//
// The shipped binder (`re_game::emit_parts::bind_emit_parts`) claims a part cell
// only when it is in-bounds, currently `CellKind::Empty`, AND carries no pipe tag.
// The pipe filter is correct and deliberate — a pipe path cell is spoken for by
// the wiring, and a reward stamped on top of it would bury the thing the player
// manipulates. But the refusal is SILENT: a part whose footprint a pipe crosses
// simply binds fewer cells, and what ships is a flower with no middle bar.
//
// That is this project's signature defect class — authored, parsed, tested, and
// not on screen — so it gets a rule rather than a reviewer's eye. Found on L7
// (`flower_head` lost its whole centre row to `growth->bloom`) and L4
// (`edge_glyph` lost three of five cells to `pivot->edge`), 2026-08-07.
//
// ## The rule
//
//   FAIL  a resolved part cell lies on a `[pipe] path`. Names the level's part,
//         the pipe, and every lost cell. Move the PIPE (its route between two
//         endpoints is usually free); move the part's `offset` only when the pipe
//         cannot go elsewhere.
//   WARN  a resolved part cell is out of bounds or lands on non-air geometry —
//         also a silent partial bind, but one an author can legitimately intend
//         (a figure tucked against a wall), so it is named and not failed.
//   PASS  every clause binds its whole authored mask.
//
// The part-cell resolution below MIRRORS the host game's `emit_parts::part_cells`
// exactly, including the pedestal anchor rule (x/y from the bbox MIN corner, z
// from the pedestal's DECLARED `cz`). `re_lvlcheck` cannot depend on the host
// game (a binary crate in a separate repository), so the two are hand-pinned —
// the same arrangement the `[note]` placement rule already lives under, and
// the host's own `emit_parts_render` test renders the result for real.

/// One `[emit]` part cell that the shipped binder will refuse, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitCellLoss {
    /// Index of the `[emit]` clause in parse order.
    pub clause: usize,
    /// The `[part]` name whose mask cell this is.
    pub part: String,
    /// The world cell the mask resolved to.
    pub cell: (i32, i32, i32),
    /// `Some(pipe index)` when a pipe path owns the cell; `None` for a geometry
    /// or out-of-bounds loss.
    pub pipe: Option<usize>,
    /// What blocks it, in report words.
    pub reason: String,
}

/// Resolve one `[part]`'s solid mask into absolute world cells, anchored on the
/// level's `[pedestal]`.
///
/// Hand-pinned mirror of `re_game::emit_parts::part_cells` — see the section
/// comment. X/Y take the pedestal bbox's MIN corner; Z takes the pedestal's
/// DECLARED `cz` (its standing surface), so `offset z = 0` lands on the deck.
fn emit_part_cells(level: &Level, part_idx: usize) -> Vec<(i32, i32, i32)> {
    let Some(part) = level.parts.get(part_idx) else {
        return Vec::new();
    };
    let (w, h, d) = (part.w.max(0), part.h.max(0), part.d.max(0));
    if w == 0 || h == 0 || d == 0 {
        return Vec::new();
    }
    let ped = &level.pedestal;
    let base_x = ped.cx - ped.w / 2 + part.off_x;
    let base_y = ped.cy - ped.h / 2 + part.off_y;
    let base_z = ped.cz + part.off_z;

    let mut out = Vec::new();
    for pz in 0..d {
        for py in 0..h {
            for px in 0..w {
                let idx = px as usize + (w as usize) * (py as usize + (h as usize) * pz as usize);
                if part.mask.get(idx).copied().unwrap_or(0) == 0 {
                    continue;
                }
                out.push((base_x + px, base_y + py, base_z + pz));
            }
        }
    }
    out
}

/// Every cell of `level`'s `[emit]` rewards that the shipped binder will refuse.
///
/// Public so tests (and a future editor drawer) can assert on the exact losses
/// rather than re-deriving the anchor rule a fourth time.
#[must_use]
pub fn emit_part_losses(level: &Level) -> Vec<EmitCellLoss> {
    if level.emits.is_empty() {
        return Vec::new();
    }
    let map = level.build_map();
    let (sx, sy, sz) = map.size();
    let mut out = Vec::new();

    for (ci, emit) in level.emits.iter().enumerate() {
        let part_name = level
            .parts
            .get(emit.part)
            .map_or_else(|| format!("#{}", emit.part), |p| p.name.clone());
        for cell in emit_part_cells(level, emit.part) {
            let (x, y, z) = cell;
            let loss = |pipe: Option<usize>, reason: String| EmitCellLoss {
                clause: ci,
                part: part_name.clone(),
                cell,
                pipe,
                reason,
            };
            if x < 0 || y < 0 || z < 0 || x as usize >= sx || y as usize >= sy || z as usize >= sz {
                out.push(loss(None, "outside the room".into()));
                continue;
            }
            // The PIPE tag is what the binder tests, not the cell kind: a path
            // cell keeps its tag even where a later `[cells]` pass demoted the
            // trace back to air.
            if let Some(pipe) = re_core::decode_pipe_tag(map.side_data_at(x, y, z)) {
                let route = level.pipes.get(pipe as usize).map_or_else(
                    || format!("pipe #{pipe}"),
                    |p| format!("[pipe {}->{}]", p.from, p.to),
                );
                out.push(loss(
                    Some(pipe as usize),
                    format!("{route} routes through it"),
                ));
                continue;
            }
            let kind = map.cell_at(x, y, z);
            if kind != CellKind::Empty {
                out.push(loss(None, format!("occupied by {kind:?} geometry")));
            }
        }
    }
    out
}

/// The `[emit]` footprint rule. See the section comment for the semantics.
fn check_emit_parts(level: &Level) -> Check {
    if level.emits.is_empty() {
        return Check::new("emit_parts", Severity::Pass, "no [emit] reward authored");
    }
    let authored: usize = level
        .emits
        .iter()
        .map(|e| emit_part_cells(level, e.part).len())
        .sum();
    let losses = emit_part_losses(level);
    let (pipe_losses, other_losses): (Vec<_>, Vec<_>) =
        losses.iter().cloned().partition(|l| l.pipe.is_some());
    let bound = authored.saturating_sub(losses.len());

    if !pipe_losses.is_empty() {
        // A level can lose cells to BOTH causes at once; `bound` counts every loss, so
        // say how the arithmetic splits rather than leaving the reader to guess why
        // `authored - eaten` does not equal it.
        let also = if other_losses.is_empty() {
            String::new()
        } else {
            format!(
                " ({} further cell(s) are lost to geometry — see below)",
                other_losses.len()
            )
        };
        // Locations follow the detail's own order: the pipe losses it lists
        // first, then the geometry losses after the `||`.
        let locs = pipe_losses
            .iter()
            .chain(other_losses.iter())
            .map(|l| Location::cell(l.cell))
            .collect();
        return Check::new(
            "emit_parts",
            Severity::Fail,
            format!(
                "{}/{authored} reward cell(s) are EATEN BY THE PIPE NETWORK — the [emit] figure \
                 binds {bound}{also} and ships with holes in its silhouette. Move the pipe's \
                 route (or, if the route is pinned, the [part] offset): {}{}",
                pipe_losses.len(),
                emit_loss_lines(&pipe_losses),
                if other_losses.is_empty() {
                    String::new()
                } else {
                    format!(" || {}", emit_loss_lines(&other_losses))
                }
            ),
        )
        .at_all(locs);
    }
    if !other_losses.is_empty() {
        return Check::new(
            "emit_parts",
            Severity::Warn,
            format!(
                "{}/{authored} reward cell(s) cannot bind (binds {bound}): {}",
                other_losses.len(),
                emit_loss_lines(&other_losses)
            ),
        )
        .at_all(
            other_losses
                .iter()
                .map(|l| Location::cell(l.cell))
                .collect(),
        );
    }
    Check::new(
        "emit_parts",
        Severity::Pass,
        format!(
            "all {authored} cell(s) of {} [emit] clause(s) bind — every reward keeps its whole \
             authored silhouette",
            level.emits.len()
        ),
    )
}

// ===========================================================================
// 11. External assets — every file the level names must actually be there
// ===========================================================================
//
// A `.lvl` names files it does not contain: `[texture] file`, the three `[sky]`
// images, `[vox] file`, `[inset] child`. `re_level` resolves each against the
// level's OWN location and reports a miss with the full list of candidate paths
// it probed — and until now that report reached nobody. `build_map` skips a
// broken asset best-effort (right for a player: an untextured wall beats no
// game), and `build_map_checked`, the one function that surfaced the error, had
// no caller outside `re_level`'s unit tests. Deleting a level's bound texture
// file produced an untextured level and ZERO output.
//
// This is where that becomes loud, because the linter is the one consumer whose
// entire job is "is this level valid": it already walks all of `levels/`, it runs
// on every editor save, and it runs in CI — so a removed asset is caught across
// every level that references it the moment it is removed, which no per-level
// content assertion can do.
//
// ## The rule
//
//   FAIL  the level came from a FILE and one of its assets does not resolve,
//         cannot be read, or cannot be decoded. Names every candidate probed.
//   WARN  the same, but the level has no `source_path` (parsed from a string: a
//         unit-test fixture, or a brand-new unsaved editor document). Without a
//         file to anchor to, only the CWD-relative candidate exists, so "not
//         found" genuinely means "cannot tell yet" — visible, never fatal.
//   PASS  every reference resolves AND decodes (or the level names no files).
//
// The linter uses `audit_assets` (resolve + read + DECODE), not the cheap
// `missing_assets` the game load path uses: paying a full image decode to answer
// "is this level valid" is exactly the trade a linter should make, and it catches
// a truncated PNG that mere existence cannot. Both collect EVERY issue —
// `build_map_checked` stops at the first, which is useless as a gate (fix one
// texture, fail again on the next).

/// Check every external file the level names (see the section comment).
/// Returns the number of BROKEN references alongside the check, so a consumer
/// (the editor's status line) can say "3 assets missing" without re-auditing or
/// scraping the detail text.
fn check_assets(level: &Level) -> (usize, Check) {
    let refs = level.asset_refs();
    if refs.is_empty() {
        return (
            0,
            Check::new("assets", Severity::Pass, "no external asset references"),
        );
    }
    let issues = level.audit_assets();
    if issues.is_empty() {
        return (
            0,
            Check::new(
                "assets",
                Severity::Pass,
                format!(
                    "all {} asset reference(s) resolve and decode ({})",
                    refs.len(),
                    asset_ref_summary(&refs)
                ),
            ),
        );
    }
    // No `source_path` ⇒ no anchor ⇒ only the CWD candidate was probed. That is a
    // "cannot tell", not a defect in the level — never sink a string-parsed
    // fixture (or an unsaved editor document) to FAIL over it.
    let anchored = level.source_path.is_some();
    let severity = if anchored {
        Severity::Fail
    } else {
        Severity::Warn
    };
    let caveat = if anchored {
        ""
    } else {
        " (this level has no file behind it, so only the working directory was \
         searched — save it and re-check)"
    };
    let lines = issues
        .iter()
        .map(|i| format!("\n    {i}"))
        .collect::<Vec<_>>()
        .join("");
    // A broken asset has no cell: the offender is the CLAUSE that names the file.
    // `AssetRef::index` addresses it inside its own collection, which is exactly
    // what an editor selects.
    let locs = issues
        .iter()
        .map(|i| Location::element(i.asset.kind.into(), i.asset.index, i.asset.label.clone()))
        .collect();
    (
        issues.len(),
        Check::new(
            "assets",
            severity,
            format!(
                "{} of {} asset reference(s) DO NOT LOAD{caveat} — the level still builds, with \
                 that surface falling back to untextured/procedural:{lines}",
                issues.len(),
                refs.len()
            ),
        )
        .at_all(locs),
    )
}

/// `2 texture, 1 sky` — the per-kind census for the PASS line, so a reader can
/// see at a glance what was actually checked.
fn asset_ref_summary(refs: &[re_level::AssetRef]) -> String {
    use re_level::AssetKind;
    [
        AssetKind::Texture,
        AssetKind::Sky,
        AssetKind::Vox,
        AssetKind::Inset,
    ]
    .into_iter()
    .filter_map(|k| {
        let n = refs.iter().filter(|r| r.kind == k).count();
        (n > 0).then(|| format!("{n} {}", k.tag()))
    })
    .collect::<Vec<_>>()
    .join(", ")
}

fn emit_loss_lines(losses: &[EmitCellLoss]) -> String {
    losses
        .iter()
        .map(|l| {
            format!(
                "clause {} part '{}' cell {:?}: {}",
                l.clause, l.part, l.cell, l.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

// ===========================================================================
// 1. Element inventory
// ===========================================================================

/// Tally every element kind in `level`.
pub fn inventory(level: &Level) -> Inventory {
    let inputs = level.nodes.iter().filter(|n| n.is_input).count();
    let outputs = level.nodes.iter().filter(|n| n.is_output).count();
    let nodes_total = level.nodes.len();
    let hidden = nodes_total.saturating_sub(inputs + outputs);
    let switches = level
        .pipes
        .iter()
        .filter(|p| p.switch_cell.is_some())
        .count();
    Inventory {
        inputs,
        outputs,
        hidden,
        nodes_total,
        pipes: level.pipes.len(),
        switches,
        parts: level.parts.len(),
        emits: level.emits.len(),
        decals: level.decals.len(),
        notes: level.notes.len(),
        holos: level.holos.len(),
        // The loader hard-requires door/pedestal/player, so these are always
        // present in a successfully-parsed level — but reported honestly.
        has_door: !level.door.target_parts.is_empty() || door_is_set(level),
        has_player: true,
        has_pedestal: level.pedestal.w > 0 && level.pedestal.h > 0 && level.pedestal.d > 0,
    }
}

/// A door is considered "set" if it has an anchor position or any target parts.
fn door_is_set(level: &Level) -> bool {
    level.door.x != 0
        || level.door.y != 0
        || level.door.z != 0
        || !level.door.target_parts.is_empty()
}

fn check_inventory(inv: &Inventory) -> Check {
    let detail = format!(
        "nodes={} (in {}, out {}, hidden {}) · pipes={} · switches={} · parts={} · emits={} · decals={} · notes={} · holos={} · door={} · player={} · pedestal={}",
        inv.nodes_total,
        inv.inputs,
        inv.outputs,
        inv.hidden,
        inv.pipes,
        inv.switches,
        inv.parts,
        inv.emits,
        inv.decals,
        inv.notes,
        inv.holos,
        yn(inv.has_door),
        yn(inv.has_player),
        yn(inv.has_pedestal),
    );
    // FAIL if there is no real puzzle: no nodes, or nothing to toggle.
    if inv.nodes_total == 0 {
        Check::new(
            "inventory",
            Severity::Fail,
            format!("no nodes — nothing to do. {detail}"),
        )
    } else if inv.switches == 0 {
        Check::new(
            "inventory",
            Severity::Fail,
            format!("0 switches — the player has nothing to toggle. {detail}"),
        )
    } else if inv.outputs == 0 {
        Check::new(
            "inventory",
            Severity::Warn,
            format!("0 output nodes — win depends on door target_parts only. {detail}"),
        )
    } else {
        Check::new("inventory", Severity::Pass, detail)
    }
}

// ===========================================================================
// 2. Structural validity
// ===========================================================================

fn check_structure(level: &Level) -> Vec<Check> {
    let mut checks = Vec::new();
    let n = level.nodes.len();

    // --- pipe endpoints reference valid node indices ---------------------
    let mut bad_pipes = Vec::new();
    for (i, p) in level.pipes.iter().enumerate() {
        if p.from_node >= n || p.to_node >= n {
            bad_pipes.push(i);
        }
    }
    if bad_pipes.is_empty() {
        checks.push(Check::new(
            "edges",
            Severity::Pass,
            format!(
                "all {} pipes reference valid node indices",
                level.pipes.len()
            ),
        ));
    } else {
        checks.push(
            Check::new(
                "edges",
                Severity::Fail,
                format!(
                    "{} pipe(s) reference an out-of-range node index: {bad_pipes:?}",
                    bad_pipes.len()
                ),
            )
            // The pipe's own endpoints are the broken thing, so they cannot be
            // resolved to node cells - the CLAUSE is the only honest address.
            .at_all(bad_pipes.iter().map(|&i| pipe_element(level, i)).collect()),
        );
    }

    // --- door target_parts reference real nodes --------------------------
    let bad_targets: Vec<usize> = level
        .door
        .target_parts
        .iter()
        .copied()
        .filter(|&t| t >= n)
        .collect();
    // Both failure modes are about the `[door]` clause itself, and the door has
    // an anchor cell - so the cursor goes to the door, not to a node index that
    // by definition names nothing.
    let door_cell = Location::cell((level.door.x, level.door.y, level.door.z));
    if level.door.target_parts.is_empty() {
        checks.push(
            Check::new(
                "door-targets",
                Severity::Fail,
                "door has no target_parts — the level can never be won",
            )
            .at(door_cell),
        );
    } else if bad_targets.is_empty() {
        checks.push(Check::new(
            "door-targets",
            Severity::Pass,
            format!(
                "door targets {} real node(s)",
                level.door.target_parts.len()
            ),
        ));
    } else {
        checks.push(
            Check::new(
                "door-targets",
                Severity::Fail,
                format!("door target_parts reference out-of-range node(s): {bad_targets:?}"),
            )
            .at(door_cell),
        );
    }

    // --- [inputs] reference real nodes (parser already validated, re-check) -
    let bad_inputs: Vec<&String> = level
        .inputs
        .iter()
        .filter(|(name, _)| level.node_index(name).is_none())
        .map(|(name, _)| name)
        .collect();
    if bad_inputs.is_empty() {
        checks.push(Check::new(
            "inputs",
            Severity::Pass,
            format!(
                "{} input binding(s) resolve to real nodes",
                level.inputs.len()
            ),
        ));
    } else {
        checks.push(Check::new(
            "inputs",
            Severity::Fail,
            format!("[inputs] reference unknown node(s): {bad_inputs:?}"),
        ));
    }

    // --- orphan nodes ----------------------------------------------------
    // A non-input node with no incoming pipe, or a non-output node with no
    // outgoing pipe, is dangling (can never participate in a win or be driven).
    let mut has_in = vec![false; n];
    let mut has_out = vec![false; n];
    for p in &level.pipes {
        if p.to_node < n {
            has_in[p.to_node] = true;
        }
        if p.from_node < n {
            has_out[p.from_node] = true;
        }
    }
    let mut orphans = Vec::new();
    let mut orphan_cells: Vec<Location> = Vec::new();
    for (i, node) in level.nodes.iter().enumerate() {
        // One entry (and one location) per REASON, so a node that is dangling at
        // both ends is listed twice and the 1:1 detail-to-location order holds.
        if !node.is_input && !has_in[i] {
            orphans.push(format!("{} (no incoming pipe)", node.name));
            orphan_cells.push(Location::cell((node.x, node.y, node.z)));
        }
        if !node.is_output && !has_out[i] {
            orphans.push(format!("{} (no outgoing pipe)", node.name));
            orphan_cells.push(Location::cell((node.x, node.y, node.z)));
        }
    }
    if orphans.is_empty() {
        checks.push(Check::new("orphans", Severity::Pass, "no orphan nodes"));
    } else {
        checks.push(
            Check::new(
                "orphans",
                Severity::Warn,
                format!("orphan node(s): {}", orphans.join(", ")),
            )
            .at_all(orphan_cells),
        );
    }

    // --- DAG check (matches the loader's forward-pass cycle gate) ---------
    match PuzzleState::from_level(level) {
        Ok(_) => checks.push(Check::new(
            "dag",
            Severity::Pass,
            "network is a DAG (acyclic)",
        )),
        Err(e) => {
            // `BadEdge` is the same defect `edges` located, so name the same
            // pipes. `Cycle` and `TooManyNodes` carry no node in the error and
            // there is no honest single culprit to invent - a cycle is a
            // PROPERTY of a set of edges, not a place.
            let locs: Vec<Location> = if e == re_puzzle::ForwardError::BadEdge {
                bad_pipes.iter().map(|&i| pipe_element(level, i)).collect()
            } else {
                Vec::new()
            };
            checks.push(
                Check::new(
                    "dag",
                    Severity::Fail,
                    format!("network is not a valid DAG: {e:?}"),
                )
                .at_all(locs),
            );
        }
    }

    checks
}

/// The `[pipe]` clause `index` as a [`Location::Element`], named the way the
/// `.lvl` names it (`R->output`).
fn pipe_element(level: &Level, index: usize) -> Location {
    let name = level
        .pipes
        .get(index)
        .map_or_else(|| format!("#{index}"), |p| format!("{}->{}", p.from, p.to));
    Location::element(ElementKind::Pipe, index, name)
}

// ===========================================================================
// 3. Solvability
// ===========================================================================

fn check_solvability(net: &Result<PuzzleState, re_puzzle::ForwardError>) -> (SolveVerdict, Check) {
    let mut state = match net {
        Ok(s) => s.clone(),
        Err(e) => {
            return (
                SolveVerdict::NetworkError,
                Check::new(
                    "solvable",
                    Severity::Fail,
                    format!("cannot build network ({e:?}) — unsolvable by construction"),
                ),
            );
        }
    };

    match solve_brute_force(&mut state) {
        Solvability::Solvable {
            min_toggles,
            switchable,
            ..
        } => (
            SolveVerdict::Solvable { min_toggles, switchable },
            Check::new(
                "solvable",
                Severity::Pass,
                format!(
                    "SOLVABLE — minimum {min_toggles} toggle(s) from the authored start ({switchable} switchable pipe(s))"
                ),
            ),
        ),
        Solvability::TriviallySolvedAtLoad => (
            SolveVerdict::TriviallySolved,
            Check::new(
                "solvable",
                Severity::Warn,
                "TRIVIALLY-SOLVED-AT-LOAD — already won before any toggle; there is no puzzle to perform",
            ),
        ),
        Solvability::Unsolvable => (
            SolveVerdict::Unsolvable,
            Check::new(
                "solvable",
                Severity::Fail,
                "UNSOLVABLE — no switch combination wins this level",
            ),
        ),
        Solvability::TooManyPipes { switchable } => (
            SolveVerdict::TooManyToBruteForce { switchable },
            Check::new(
                "solvable",
                Severity::Warn,
                format!(
                    "too many switchable pipes to brute-force ({switchable} > cap) — solvability not exhaustively verified"
                ),
            ),
        ),
    }
}

// ===========================================================================
// 8. Power budget
// ===========================================================================

/// Lint `[room] power_budget = N` — the cap on simultaneously-ON pipes.
///
/// A budget one unit too low ships an UNWINNABLE level, which is the worst
/// failure mode this linter exists to catch, so the check is deliberately
/// stronger than the ordinary solvability sweep:
///
/// * The solvability sweep ([`re_puzzle::solve_brute_force`]) is STATELESS. It
///   forward-passes isolated pipe assignments, which is simply wrong for a level
///   whose win is a state machine — on `L11-inner-circuit` the `[sequence]` lock
///   advances as a side effect of the sweep's own enumeration, so it reports a
///   one-toggle "solution" for a lock that needs three ordered output states.
/// * This check instead runs [`re_puzzle::win_reachable_under_budget`], a real
///   search over the reachable state space where every edge is an actual
///   `puzzle_interact` call — refusals, cycle rollback, the ordered lock and the
///   inverted / cycle-keep win modes all included.
///
/// Verdicts: `0` (absent) ⇒ PASS "unlimited" (the pre-rule default, and the
/// state of every non-campaign level). Unwinnable ⇒ FAIL. An authored start that
/// already exceeds its own cap ⇒ FAIL. A cap at or above the pipe count ⇒ WARN
/// (it can never bind — the level is not actually constrained). Otherwise PASS,
/// reporting how much slack sits above the level's true minimum.
fn check_power_budget(level: &Level, net: &Result<PuzzleState, re_puzzle::ForwardError>) -> Check {
    let budget = level.power_budget;
    if budget == 0 {
        return Check::new(
            "power_budget",
            Severity::Pass,
            "no power budget declared — pipes are UNLIMITED (the pre-rule default)",
        );
    }
    let Ok(state) = net else {
        return Check::new(
            "power_budget",
            Severity::Fail,
            format!("power_budget = {budget} declared, but the network does not build"),
        );
    };

    let pipes = state.pipes.len() as u32;
    let start_draw = state.power_used();
    if start_draw > budget {
        return Check::new(
            "power_budget",
            Severity::Fail,
            format!(
                "power_budget = {budget} but the AUTHORED start already draws {start_draw} \
                 (permanently-ON switchless pipes count) — the player begins in a state the \
                 rule forbids"
            ),
        );
    }

    match re_puzzle::win_reachable_under_budget(state, budget) {
        re_puzzle::BudgetFeasibility::Unwinnable => Check::new(
            "power_budget",
            Severity::Fail,
            format!(
                "UNWINNABLE at power_budget = {budget} — no reachable configuration within \
                 the cap wins. The level ships unsolvable; raise the budget or rewire."
            ),
        ),
        re_puzzle::BudgetFeasibility::TooManyPipes { switchable } => Check::new(
            "power_budget",
            Severity::Warn,
            format!(
                "power_budget = {budget} declared but there are {switchable} switchable pipes \
                 — too many to search exhaustively, so winnability under the cap is NOT verified"
            ),
        ),
        re_puzzle::BudgetFeasibility::Winnable => {
            if budget >= pipes {
                return Check::new(
                    "power_budget",
                    Severity::Warn,
                    format!(
                        "power_budget = {budget} with only {pipes} pipe(s) — the cap can never \
                         bind, so it constrains nothing (it still drives the HUD readout)"
                    ),
                );
            }
            let min = re_puzzle::min_power_budget(state);
            let slack = min.map_or_else(
                || "true minimum not determined".to_string(),
                |m| format!("true minimum {m}, slack {}", budget.saturating_sub(m)),
            );
            Check::new(
                "power_budget",
                Severity::Pass,
                format!(
                    "WINNABLE at power_budget = {budget} of {pipes} pipe(s) ({slack}; authored \
                     start draws {start_draw})"
                ),
            )
        }
    }
}

// ===========================================================================
// 4. Reachability
// ===========================================================================

/// A switch is reachable when it is (a) on its pipe's path (so it can be
/// interacted with at all) and (b) physically reachable by the player — there
/// is a standable floor/solid-top cell adjacent (in x/y) to the switch whose top
/// surface is within jump reach of the switch's z layer.
fn check_reachability(level: &Level) -> (bool, Check) {
    let map = level.build_map();
    let jump_reach = max_jump_height(); // ~1 cell

    let mut issues: Vec<SwitchIssue> = Vec::new();
    let mut total_switches = 0usize;

    for (i, pipe) in level.pipes.iter().enumerate() {
        let Some(sc) = pipe.switch_cell else { continue };
        total_switches += 1;
        let cell = match pipe.path.get(sc) {
            Some(&c) => c,
            None => {
                issues.push(SwitchIssue {
                    pipe_index: i,
                    from: pipe.from.clone(),
                    to: pipe.to.clone(),
                    cell: (0, 0, 0),
                    // There is no switch cell to go to - the `switch_at` index
                    // is the broken thing, so the clause is the honest address.
                    location: pipe_element(level, i),
                    reason: "switch_cell index is off the end of the pipe path".into(),
                });
                continue;
            }
        };

        // (a) on-path: switch_cell already indexes into path, so this holds by
        // construction once path.get succeeded. We still assert the cell is one
        // of the path's cells (defensive against a derived/mis-set switch_cell).
        if !pipe.path.contains(&cell) {
            issues.push(SwitchIssue {
                pipe_index: i,
                from: pipe.from.clone(),
                to: pipe.to.clone(),
                cell,
                location: Location::cell(cell),
                reason: "switch cell is not on the pipe path (not interactable)".into(),
            });
            continue;
        }

        // (b) physically reachable: is there a standable spot the player can be
        // in to interact with this switch?
        if !switch_physically_reachable(&map, cell, jump_reach) {
            issues.push(SwitchIssue {
                pipe_index: i,
                from: pipe.from.clone(),
                to: pipe.to.clone(),
                cell,
                location: Location::cell(cell),
                reason: "no standable floor within jump reach near the switch".into(),
            });
        }
    }

    if total_switches == 0 {
        return (
            false,
            Check::new("reachable", Severity::Fail, "no switches to reach"),
        );
    }

    if issues.is_empty() {
        (
            true,
            Check::new(
                "reachable",
                Severity::Pass,
                format!("all {total_switches} switch(es) are on-path and physically reachable"),
            ),
        )
    } else {
        let lines: Vec<String> = issues
            .iter()
            .map(|s| {
                format!(
                    "pipe {} ({}->{}) switch {:?}: {}",
                    s.pipe_index, s.from, s.to, s.cell, s.reason
                )
            })
            .collect();
        (
            false,
            Check::new(
                "reachable",
                Severity::Fail,
                format!(
                    "{}/{} switch(es) unreachable: {}",
                    issues.len(),
                    total_switches,
                    lines.join(" | ")
                ),
            )
            .at_all(issues.iter().map(|s| s.location.clone()).collect()),
        )
    }
}

/// The peak height (in cells) a jump from a standing start reaches, from the
/// engine's own constants: `h = v² / (2·g)`.
fn max_jump_height() -> f32 {
    (JUMP_IMPULSE * JUMP_IMPULSE) / (2.0 * GRAVITY)
}

/// Is `switch` interactable from some standable cell? The player stands in an
/// air cell whose floor (the cell below) is a solid/floor surface, and reaches
/// the switch by looking at it. We treat the switch as reachable if any of the
/// 4 horizontally-adjacent columns (plus the switch's own column) offers a
/// standable air cell whose standing-eye z is within jump reach of the switch z.
fn switch_physically_reachable(map: &Map, switch: (i32, i32, i32), jump_reach: f32) -> bool {
    let (sx, sy, sz) = switch;
    let (_, _, mz) = map.size();
    let candidates = [
        (sx, sy),
        (sx + 1, sy),
        (sx - 1, sy),
        (sx, sy + 1),
        (sx, sy - 1),
    ];
    for (cx, cy) in candidates {
        // Scan the column for a standable air cell: empty cell whose cell below
        // is a floor or solid surface.
        for z in 0..mz as i32 {
            let here = map.cell_at(cx, cy, z);
            if here.is_solid() {
                continue; // can't stand inside a wall/object
            }
            let below = map.cell_at(cx, cy, z - 1);
            let standable = below == CellKind::Floor || below.is_solid() || z == 0;
            if !standable {
                continue;
            }
            // Standing on cell z (feet at z), the player can reach switches from
            // the floor (z) up to z + jump_reach + a head/arm allowance of ~1.
            let reach_lo = z as f32 - 0.5;
            let reach_hi = z as f32 + jump_reach + 1.5;
            if (sz as f32) >= reach_lo && (sz as f32) <= reach_hi {
                return true;
            }
        }
    }
    false
}

// ===========================================================================
// 5. Coherence heuristics
// ===========================================================================

/// Combine the spawn-faces-puzzle, puzzle-in-reach, and elements-at-eye-level
/// heuristics into a single verdict plus the supporting checks.
fn check_coherence(level: &Level) -> (bool, String, Vec<Check>) {
    let mut checks = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    let spawn = (
        level.player.x as f32,
        level.player.y as f32,
        level.player.z as f32,
    );

    // The puzzle "points of interest" the player should be facing: every node
    // and every switch cell. The DOOR is deliberately excluded — it is the exit
    // and is conventionally placed BEHIND the spawn (you walk away from it into
    // the room), so including it would mis-flag a correctly-oriented spawn.
    // Kept as integer CELLS, not floats: every coherence finding points at one of
    // these, and rounding a float back to a cell to locate it would be a second
    // chance to be wrong.
    let mut puzzle_pois: Vec<(i32, i32, i32)> = Vec::new();
    for n in &level.nodes {
        puzzle_pois.push((n.x, n.y, n.z));
    }
    for p in &level.pipes {
        if let Some(c) = p.switch_cell.and_then(|i| p.path.get(i)) {
            puzzle_pois.push(*c);
        }
    }
    // The full POI set (puzzle elements + the door) for the distance check.
    let mut pois = puzzle_pois.clone();
    pois.push((level.door.x, level.door.y, level.door.z));

    // --- spawn-faces-puzzle ----------------------------------------------
    // Direction the player faces (xy plane): dir = (sin yaw, cos yaw),
    // yaw = facing_yaw_col / 480 * TAU.
    let yaw = (level.player.facing_yaw_col as f32 / 480.0) * std::f32::consts::TAU;
    let facing = (yaw.sin(), yaw.cos());

    // Nearest puzzle POI to spawn (by xy distance) — door excluded.
    let nearest = puzzle_pois.iter().copied().min_by(|a, b| {
        dist2_xy(spawn, as_f32(*a))
            .partial_cmp(&dist2_xy(spawn, as_f32(*b)))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(target_cell) = nearest {
        let target = as_f32(target_cell);
        let to = (target.0 - spawn.0, target.1 - spawn.1);
        let len = (to.0 * to.0 + to.1 * to.1).sqrt();
        if len < 1e-3 {
            // Spawn sits on a POI; facing is moot.
            checks.push(Check::new(
                "spawn-facing",
                Severity::Pass,
                "spawn coincides with the nearest puzzle element",
            ));
        } else {
            let to_n = (to.0 / len, to.1 / len);
            let dot = facing.0 * to_n.0 + facing.1 * to_n.1;
            // dot >= cos(90°) = 0 ⇒ within ±90° of the puzzle.
            if dot >= 0.0 {
                checks.push(Check::new(
                    "spawn-facing",
                    Severity::Pass,
                    format!("spawn faces toward the puzzle (alignment {dot:+.2})"),
                ));
            } else {
                reasons.push("spawn faces away from the puzzle".into());
                checks.push(
                    Check::new(
                        "spawn-facing",
                        Severity::Warn,
                        format!(
                            "spawn faces AWAY from the nearest puzzle element (alignment {dot:+.2})"
                        ),
                    )
                    // The SPAWN is what an author re-aims, so the cursor goes
                    // there; the element it should be facing follows second.
                    .at(Location::cell((
                        level.player.x,
                        level.player.y,
                        level.player.z,
                    )))
                    .at(Location::cell(target_cell)),
                );
            }
        }
    }

    // --- puzzle-in-reach (sane distance from spawn) ----------------------
    // The farthest POI should be within the room diagonal; flag a node that is
    // implausibly far (more than the room's own diagonal — which would mean a
    // coordinate error).
    let (rx, ry, rz) = level.size;
    let room_diag = ((rx * rx + ry * ry + rz * rz) as f32).sqrt();
    let farthest = pois
        .iter()
        .copied()
        .fold(None, |acc: Option<((i32, i32, i32), f32)>, p| {
            let d = dist3(spawn, as_f32(p));
            match acc {
                Some((_, best)) if best >= d => acc,
                _ => Some((p, d)),
            }
        });
    let max_d = farthest.map_or(0.0, |(_, d)| d);
    if max_d <= room_diag + 0.5 {
        checks.push(Check::new(
            "puzzle-reach",
            Severity::Pass,
            format!(
                "all puzzle elements within {max_d:.1} cells of spawn (room diag {room_diag:.1})"
            ),
        ));
    } else {
        reasons.push("a puzzle element sits beyond the room diagonal from spawn".into());
        checks.push(
            Check::new(
                "puzzle-reach",
                Severity::Warn,
                format!(
                    "farthest element is {max_d:.1} cells from spawn (> room diag {room_diag:.1})"
                ),
            )
            // The detail names ONE element - the farthest - so that is the one
            // location. `farthest` is always `Some` here (`pois` always holds at
            // least the door), but never unwrap on that.
            .at_all(
                farthest
                    .map(|(c, _)| Location::cell(c))
                    .into_iter()
                    .collect(),
            ),
        );
    }

    // --- elements-at-eye-level (switches near a standable z) -------------
    // Compare every switch z to the spawn z. A switch buried more than ~1 jump
    // above the spawn plane with no nearer standing surface reads as out of
    // initial reach.
    let map = level.build_map();
    let jump_reach = max_jump_height();
    let mut buried = Vec::new();
    let mut buried_cells: Vec<Location> = Vec::new();
    for p in &level.pipes {
        if let Some(c) = p.switch_cell.and_then(|i| p.path.get(i)) {
            // Within jump reach of the spawn z plane?
            let dz = c.2 as f32 - level.player.z as f32;
            let near_spawn_plane = dz <= jump_reach + 1.0 && dz >= -2.0;
            // …or independently reachable (a local standing surface).
            let locally = switch_physically_reachable(&map, *c, jump_reach);
            if !near_spawn_plane && !locally {
                buried.push(format!("{}->{} @ z={}", p.from, p.to, c.2));
                buried_cells.push(Location::cell(*c));
            }
        }
    }
    if buried.is_empty() {
        checks.push(Check::new(
            "eye-level",
            Severity::Pass,
            "switches sit near a standable z (in initial reach)",
        ));
    } else {
        reasons.push("a switch is buried out of initial reach".into());
        checks.push(
            Check::new(
                "eye-level",
                Severity::Warn,
                format!(
                    "switch(es) buried out of initial view/reach: {}",
                    buried.join(", ")
                ),
            )
            .at_all(buried_cells),
        );
    }

    let coherent = reasons.is_empty();
    let verdict = if coherent {
        "COHERENT".to_string()
    } else {
        format!("NEEDS LAYOUT WORK ({})", reasons.join("; "))
    };
    (coherent, verdict, checks)
}

// ===========================================================================
// 6. `[note]` placard placement
// ===========================================================================
//
// Every `[note]` must resolve to a real wall FACE, or its authored text is parsed
// and never seen. The rule below mirrors the SHIPPED binder semantics exactly
// (the reader's-eye-layer z-snap). THREE hand-kept copies of this algorithm
// exist, pinned together:
//
//   * the host game's `level_text.rs` (separate repository) — the shipped
//     binder, the source of truth
//   * `editor/re_imgqa/src/text_bind.rs`                     — the QA host's replica
//   * this section                                           — the linter's replica
//
// If the shipped semantics change, ALL THREE move. The binder resolves a placard
// in three stages (`level_text::resolve_note_wall`):
//
//   1. WHICH WALL, in the AUTHORED z-plane: the authored `pos` when it is itself
//      masonry (cell KIND only — paintability is the slide step's decision),
//      otherwise the first masonry cell along the reader's gaze (opposite the
//      `facing` normal) within `NOTE_WALL_SEARCH_MAX` cells (`wall_in_plane`).
//   2. WHICH LAYER: that wall column is taken at the READER'S EYE LAYER — the one
//      z-slice the 2.5D caster samples for the storey the authored `z` falls in
//      (`reader_eye_layer`, clamped into the map's habitable band) — sliding up to
//      ±`NOTE_WALL_SLIDE_MAX` along the wall run (offset order 0, +1, −1, +2, …)
//      to the nearest paintable, unclaimed cell whose face is actually viewable
//      (`slide_to_free_wall` + `face_is_visible`).
//   3. FALLBACK: if the authored plane names no usable wall, the gaze search is
//      re-run IN the eye layer itself.
//
// Verdict mapping:
//
//   PASS  the binder places the placard at `pos`'s own resolved cell — `pos` is
//         the wall column and nothing had to slide. (The eye-layer z-snap is the
//         binder's DEFINED semantics, not a tolerance: the authored `z` selects
//         the STOREY, and the visible layer is derived from it.)
//   WARN  placed via a tolerance — gaze search, wall-run slide, and/or the
//         eye-plane fallback. It renders, but the level is leaning on the
//         binder's tolerances rather than authoring the wall cell.
//   FAIL  binder-unplaceable: no paintable, viewable, unclaimed wall cell in
//         reach in either plane — that placard's text is parsed and never seen.
//
// DUPLICATION, on purpose: `re_lvlcheck` must depend on neither the host game (a
// binary crate in a separate repository) nor `re_imgqa` (the dependency direction
// is linter → data crates only), and the logic is ~60 lines against `Map`. The
// three copies are pinned together by the constants below, by the host's
// `level_text` unit tests over the same cases, and by this crate's own mirror
// tests. Hoisting ONE shared resolver
// into `re_level` was considered and not done: the eye-layer arithmetic derives
// from `re_render::gameplay_eye_ref_z`, and `re_level` does not depend on
// `re_render` — that hoist is an architecture decision, not a linter-side fix.

/// How far the shipped binder searches for the placard's wall when `pos` is not
/// itself masonry. MUST match `re_game::level_text::NOTE_WALL_SEARCH_MAX` (and
/// the `re_imgqa::text_bind` copy).
pub const NOTE_WALL_SEARCH_MAX: i32 = 24;

/// How far the shipped binder slides a placard ALONG the wall run.
///
/// The slide hunts for a paintable, viewable, unclaimed cell before giving up.
/// MUST match `re_game::level_text::NOTE_WALL_SLIDE_MAX` (and the
/// `re_imgqa::text_bind` copy).
pub const NOTE_WALL_SLIDE_MAX: i32 = 4;

/// Columns in the `.lvl` `facing` yaw encoding (`+y = 0`, `+x = 120`, `−y = 240`,
/// `−x = 360`). MUST match `re_core::YAW_COLS`.
const YAW_COLS: i32 = 480;

/// One placard that could not be placed cleanly, for the report line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteIssue {
    /// Index of the `[note]` in parse order (its `NoteWall { note: N }` id).
    pub index: usize,
    /// The authored `pos`.
    pub pos: (i32, i32, i32),
    /// The authored `facing`, as a compass token.
    pub facing: &'static str,
    /// What went wrong (or, for a search hit, what was tolerated).
    pub reason: String,
}

/// How one placard resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NotePlacement {
    /// `pos` itself names the wall column and nothing slid — authored exactly
    /// per the pinned semantics (the eye-layer z-snap still applies).
    AtPos,
    /// Placed, but only via the binder's tolerances.
    Tolerated {
        /// Cells walked along the reader's gaze to find the wall column
        /// (0 = `pos` was the column).
        search: i32,
        /// Offset slid along the wall run (0 = the resolved column itself).
        slid: i32,
        /// The authored plane named no usable wall; the eye-layer re-search
        /// placed it.
        eye_fallback: bool,
    },
    /// No paintable, viewable, unclaimed wall cell in reach — CANNOT render.
    Unplaced,
}

/// The `[note]` placement rule: every placard must resolve to a wall face the
/// shipped binder can bind (see the section comment for the pinned semantics).
fn check_notes(level: &Level) -> Check {
    if level.notes.is_empty() {
        return Check::new("notes", Severity::Pass, "no [note] placards authored");
    }
    let map = level.build_map();
    let storey_height = level.storey_height();
    let mut claimed: Vec<(i32, i32, i32)> = Vec::new();
    let mut at_pos = 0usize;
    let mut tolerated: Vec<NoteIssue> = Vec::new();
    let mut broken: Vec<NoteIssue> = Vec::new();

    for (i, note) in level.notes.iter().enumerate() {
        let step = facing_step(note.facing);
        let (placement, cell) = resolve_note_wall(&map, note, step, storey_height, &claimed);
        if let Some(c) = cell {
            claimed.push(c);
        }
        let issue = |reason: String| NoteIssue {
            index: i,
            pos: (note.x, note.y, note.z),
            facing: facing_token(note.facing),
            reason,
        };
        match placement {
            NotePlacement::AtPos => at_pos += 1,
            NotePlacement::Tolerated {
                search,
                slid,
                eye_fallback,
            } => {
                let mut how: Vec<String> = Vec::new();
                if search > 0 {
                    how.push(format!(
                        "the wall sits {search} cell(s) along the reader's gaze"
                    ));
                }
                if slid != 0 {
                    how.push(format!("slid {slid:+} cell(s) along the wall run"));
                }
                if eye_fallback {
                    how.push(
                        "placed by the eye-layer re-search (the authored plane had no usable wall)"
                            .to_string(),
                    );
                }
                tolerated.push(issue(format!(
                    "{} — placed at {:?}",
                    how.join("; "),
                    cell.unwrap_or_default()
                )));
            }
            NotePlacement::Unplaced => broken.push(issue(format!(
                "no paintable, viewable, unclaimed wall cell in reach (gaze <= {NOTE_WALL_SEARCH_MAX}, slide <= {NOTE_WALL_SLIDE_MAX}) in the authored plane or the reader's eye layer — this placard CANNOT render"
            ))),
        }
    }

    let total = level.notes.len();
    if !broken.is_empty() {
        // The AUTHORED `pos` is what a fix moves - not the cell the binder slid
        // to, which for a broken placard does not exist anyway.
        let locs = broken.iter().map(|n| Location::cell(n.pos)).collect();
        return Check::new(
            "notes",
            Severity::Fail,
            format!(
                "{}/{total} placard(s) are binder-unplaceable: {}",
                broken.len(),
                note_issue_lines(&broken)
            ),
        )
        .at_all(locs);
    }
    if !tolerated.is_empty() {
        let locs = tolerated.iter().map(|n| Location::cell(n.pos)).collect();
        return Check::new(
            "notes",
            Severity::Warn,
            format!(
                "{total} placard(s) render, but {} rely on the binder's placement tolerances (pos should be the WALL cell): {}",
                tolerated.len(),
                note_issue_lines(&tolerated)
            ),
        )
        .at_all(locs);
    }
    Check::new(
        "notes",
        Severity::Pass,
        format!(
            "all {at_pos} placard(s) name their wall cell directly (bound at the reader's eye layer)"
        ),
    )
}

fn note_issue_lines(issues: &[NoteIssue]) -> String {
    issues
        .iter()
        .map(|n| {
            format!(
                "note {} @ {:?} facing {}: {}",
                n.index, n.pos, n.facing, n.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Resolve one placard, mirroring `re_game::level_text::resolve_note_wall`
/// exactly (including the eye-layer z-snap). Returns how it resolved and the
/// wall cell it landed on (`None` when nothing was found).
fn resolve_note_wall(
    map: &Map,
    note: &re_level::LevelNote,
    step: (i32, i32),
    storey_height: u32,
    claimed: &[(i32, i32, i32)],
) -> (NotePlacement, Option<(i32, i32, i32)>) {
    let eye_z = reader_eye_layer(map, note.z, storey_height);
    // Stage 1+2: the wall named in the AUTHORED plane, taken at the eye layer.
    if let Some((wx, wy, search)) = wall_in_plane(map, note.x, note.y, note.z, step) {
        if let Some((cell, slid)) = slide_to_free_wall(map, wx, wy, eye_z, step, claimed) {
            let placement = if search == 0 && slid == 0 {
                NotePlacement::AtPos
            } else {
                NotePlacement::Tolerated {
                    search,
                    slid,
                    eye_fallback: false,
                }
            };
            return (placement, Some(cell));
        }
    }
    // Stage 3: re-run the gaze search IN the eye layer.
    if eye_z != note.z {
        if let Some((wx, wy, search)) = wall_in_plane(map, note.x, note.y, eye_z, step) {
            if let Some((cell, slid)) = slide_to_free_wall(map, wx, wy, eye_z, step, claimed) {
                return (
                    NotePlacement::Tolerated {
                        search,
                        slid,
                        eye_fallback: true,
                    },
                    Some(cell),
                );
            }
        }
    }
    (NotePlacement::Unplaced, None)
}

/// The z-LAYER the reader's eye sits in for the storey the authored `z` belongs
/// to — the one horizontal slice the 2.5D caster walks, and therefore the only
/// layer a placard can be seen on. Mirrors `re_game::level_text::reader_eye_layer`
/// (and the `re_imgqa::text_bind` copy).
///
/// `storey = z / storey_height`; the layer is that storey's grounded eye layer. A
/// storey whose eye layer would land on/above the ceiling slab is not standable,
/// so it steps down a storey until it fits, clamped into the habitable band
/// `[ground, sz-2]`.
fn reader_eye_layer(map: &Map, authored_z: i32, storey_height: u32) -> i32 {
    let h = storey_height.max(1) as i32;
    let (_, _, sz) = map.size();
    let ground = storey_eye_layer(0, h);
    let top = (sz as i32 - 2).max(ground);
    let mut storey = authored_z.max(0) / h;
    while storey > 0 && storey_eye_layer(storey, h) > top {
        storey -= 1;
    }
    storey_eye_layer(storey, h).clamp(ground, top)
}

/// The grounded eye LAYER (integer z-cell) of `storey` on a level of storey
/// height `h` — exactly what the caster computes for a player standing there.
///
/// The two-line eye-plane formula is a hand-kept mirror of
/// `re_render::gameplay_eye_ref_z` (which the host game and `re_imgqa` call
/// directly; `re_lvlcheck` does not depend on `re_render`), built on the same
/// `re_world` constants so there is no hand-copied literal.
fn storey_eye_layer(storey: i32, h: i32) -> i32 {
    let eye = re_world::STANDING_EYE_Z + (storey * h) as f32;
    // Mirror of `gameplay_eye_ref_z(eye)`: the slice plane the eye is grounded
    // over, clamped at the ground storey, plus the eye height back on top.
    let plane = (eye - re_world::PLAYER_EYE_HEIGHT).floor();
    let ground_plane = (re_world::STANDING_EYE_Z - re_world::PLAYER_EYE_HEIGHT).floor();
    (plane.max(ground_plane) + re_world::PLAYER_EYE_HEIGHT).floor() as i32
}

/// WHICH WALL, within one z-plane: the authored `(x, y)` when it is masonry,
/// otherwise the first masonry cell along the reader's gaze (opposite the facing
/// normal) within [`NOTE_WALL_SEARCH_MAX`]. Returns the wall column plus how many
/// cells the gaze walked (0 = the authored point itself). Tests the cell KIND
/// only — paintability is [`slide_to_free_wall`]'s decision. Mirrors
/// `re_game::level_text::wall_in_plane`.
fn wall_in_plane(map: &Map, x: i32, y: i32, z: i32, step: (i32, i32)) -> Option<(i32, i32, i32)> {
    if is_wall_kind(map, x, y, z) {
        return Some((x, y, 0));
    }
    // The reader stands on the `facing` side and looks along `-facing`; the wall
    // is somewhere along `-facing` from the authored point.
    let (dx, dy) = step;
    for n in 1..=NOTE_WALL_SEARCH_MAX {
        let (cx, cy) = (x - dx * n, y - dy * n);
        if !map.in_bounds(cx, cy, z) {
            break;
        }
        if is_wall_kind(map, cx, cy, z) {
            return Some((cx, cy, n));
        }
    }
    None
}

/// Slide along the wall RUN from `(wx, wy)` at layer `z` to the nearest
/// paintable, unclaimed cell whose face the reader can actually see, within
/// [`NOTE_WALL_SLIDE_MAX`]. Offset order `0, +1, -1, +2, -2, …` — nearest first
/// and deterministic. Returns the cell plus the offset it slid. Mirrors
/// `re_game::level_text::slide_to_free_wall`.
fn slide_to_free_wall(
    map: &Map,
    wx: i32,
    wy: i32,
    z: i32,
    step: (i32, i32),
    claimed: &[(i32, i32, i32)],
) -> Option<((i32, i32, i32), i32)> {
    // The wall run is perpendicular to the face normal: an x-facing placard
    // slides along y, a y-facing one along x.
    let (tx, ty) = (step.1.abs(), step.0.abs());
    for d in 0..=NOTE_WALL_SLIDE_MAX {
        for sign in [1, -1] {
            if d == 0 && sign < 0 {
                continue; // offset 0 is tried once
            }
            let off = d * sign;
            let cell = (wx + tx * off, wy + ty * off, z);
            if is_paintable_wall(map, cell.0, cell.1, cell.2)
                && face_is_visible(map, cell.0, cell.1, cell.2, step)
                && !claimed.contains(&cell)
            {
                return Some((cell, off));
            }
        }
    }
    None
}

/// Whether the reader can SEE this face: the cell one step along the face's
/// OUTWARD normal must be in-bounds, non-solid air to look from. A placard
/// pressed against a node/door/wall renders nothing however well it is laid out.
/// Mirrors `re_game::level_text::face_is_visible`.
fn face_is_visible(map: &Map, x: i32, y: i32, z: i32, step: (i32, i32)) -> bool {
    let (fx, fy) = (x + step.0, y + step.1);
    map.in_bounds(fx, fy, z) && !map.cell_at(fx, fy, z).is_solid()
}

/// Whether a cell is architectural MASONRY — one of the three WALL kinds (the
/// shell's kind follows `[room] ambient`). An `Object` (node/door) cell is not.
/// Mirrors `re_game::level_text::is_wall_kind`.
fn is_wall_kind(map: &Map, x: i32, y: i32, z: i32) -> bool {
    map.in_bounds(x, y, z)
        && matches!(
            map.cell_at(x, y, z),
            CellKind::WallSolid | CellKind::WallTrace | CellKind::WallMatrix
        )
}

/// Whether a cell can CARRY a placard: masonry that is also UNTAGGED — mirrors
/// `re_game::level_text::is_paintable_wall`. A tagged cell already means something
/// (a pipe tag, an authored colour id) and a puzzle `Object` cell's `side_data` is
/// its node/door tag, so neither may be repainted.
fn is_paintable_wall(map: &Map, x: i32, y: i32, z: i32) -> bool {
    is_wall_kind(map, x, y, z) && map.side_data_at(x, y, z) == 0
}

/// The OUTWARD normal of the painted face as a cell delta, from a `.lvl` `facing`
/// yaw column. Mirrors `re_core::FaceDir::from_yaw_col(..).step()`: quadrant
/// centres at 0/120/240/360 with boundaries half-way between.
fn facing_step(yaw_col: i32) -> (i32, i32) {
    match yaw_col.rem_euclid(YAW_COLS) {
        60..=179 => (1, 0),   // +x
        180..=299 => (0, -1), // -y
        300..=419 => (-1, 0), // -x
        _ => (0, 1),          // +y
    }
}

/// The compass token for a `facing` yaw column (report text only).
fn facing_token(yaw_col: i32) -> &'static str {
    match facing_step(yaw_col) {
        (1, 0) => "+x",
        (-1, 0) => "-x",
        (0, -1) => "-y",
        _ => "+y",
    }
}

// ===========================================================================
// 7. Threshold sanity — dead, always-on, and dead-by-design nodes
// ===========================================================================
//
// A binary neuron fires iff `sum(active incoming activations) >= threshold`
// (`re_puzzle::network_forward`). Two authoring mistakes are invisible to every
// other check in this linter, and both are one mistyped digit away:
//
//   * `threshold <= 0` on a non-input node — the sum of an empty set is 0, so
//     `0 >= 0` holds and the node fires with NO input at all. It looks like a
//     neuron and behaves like a constant. `solvable` cannot catch this: it makes
//     the level EASIER, never unwinnable.
//   * `threshold > in_degree` — the node can never reach threshold no matter what
//     the player toggles. Sometimes that is the mistake; sometimes it is the whole
//     point of the level (L8's `face_07`).
//
// ## The rule
//
// For every node that is NOT an input (an input's activation comes from
// `[inputs]` and its threshold is unused by the forward pass — reported, never
// judged):
//
//   PASS   `0 < threshold <= in_degree`      — satisfiable, as intended.
//   PASS   `threshold > in_degree` AND the node is NOT a door target
//          ⇒ **DEAD BY DESIGN**. Reported BY NAME so a reviewer sees the scar
//            declared rather than discovering it. This is the marker the plan
//            picked (W2 2.2): a permanently-unreachable node that the win does
//            not depend on is route-around content, which is exactly the L8
//            mechanic, and there is no `dead = true` key in `re_level` to say so
//            more directly.
//   FAIL   `threshold < 0`                   — nonsensical; no sum is ever below it.
//   FAIL   `threshold > in_degree` AND the node IS a door target
//          ⇒ the level can never be won: the door waits on a node that cannot
//            fire. (`solvable` also FAILs here; this names the reason.)
//   WARN   `threshold == 0` on a non-input   — fires unconditionally.
//   WARN   `threshold > in_degree` AND the node is an INVERTED target
//          (`[door] target_off`) ⇒ the inverse goal is trivially satisfied,
//          because the node it wants dark can never be lit anyway.
//
// ## Why `threshold == 0` is a WARN and not a FAIL
//
// `0` is the PARSER DEFAULT (`re_level` initialises `Node::threshold` to 0 and only
// overwrites it when the key is present), so "threshold 0" and "the author omitted
// the key" are the same state on disk and cannot be told apart here. Every
// technique-demo level under `levels/` leans on that default for the one-node
// scaffold puzzle it needs to satisfy the loader — those levels already report
// `solvable = TRIVIALLY-SOLVED-AT-LOAD`, which is the honest verdict for them.
// FAILing the default would turn a dozen shipped demos red for doing something
// they meant to do, so the omission is a WARN that names the node, and the real
// nonsense (a NEGATIVE threshold — never a default, never intentional) is the FAIL.
//
// A typo-dead node is still caught, from one side or the other: either the win
// depends on it (FAIL above, plus `solvable` UNSOLVABLE) or it does not, in which
// case it is inert content and this check names it in the report for a human to
// confirm. The pair "named dead node" + "SOLVABLE with an exact minimum" is what
// makes a dead-by-design level reviewable.

/// One node whose threshold is not plainly satisfiable, for the report line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThresholdIssue {
    /// The node's authored name.
    pub node: String,
    /// The node's authored `pos` - where a reader has to go to see it.
    pub cell: (i32, i32, i32),
    /// Its authored `threshold`.
    pub threshold: i32,
    /// How many pipes terminate on it.
    pub in_degree: usize,
    /// What the rule concluded.
    pub reason: String,
}

/// The threshold-sanity rule. See the module section above for the semantics.
fn check_thresholds(level: &Level) -> Check {
    let n = level.nodes.len();
    if n == 0 {
        return Check::new("thresholds", Severity::Pass, "no nodes to check");
    }

    let mut in_deg = vec![0usize; n];
    for p in &level.pipes {
        if p.to_node < n && p.from_node < n {
            in_deg[p.to_node] += 1;
        }
    }
    let is_door_target = |i: usize| level.door.target_parts.contains(&i);
    let is_inverted_target = |i: usize| level.door.target_off.contains(&i);

    let mut fails: Vec<ThresholdIssue> = Vec::new();
    let mut warns: Vec<ThresholdIssue> = Vec::new();
    let mut dead_by_design: Vec<ThresholdIssue> = Vec::new();
    let mut ok = 0usize;
    let mut inputs_skipped = 0usize;

    for (i, node) in level.nodes.iter().enumerate() {
        if node.is_input {
            // The forward pass reads an input's value from `[inputs]`; its
            // threshold never participates. Counted, not judged.
            inputs_skipped += 1;
            continue;
        }
        let t = node.threshold;
        let d = in_deg[i];
        let issue = |reason: String| ThresholdIssue {
            node: node.name.clone(),
            cell: (node.x, node.y, node.z),
            threshold: t,
            in_degree: d,
            reason,
        };
        if t < 0 {
            fails.push(issue(format!(
                "threshold {t} is negative — no sum is ever below it, so this node is a constant, not a neuron"
            )));
        } else if t == 0 {
            warns.push(issue(
                "threshold 0 (the parser default — the key may simply be missing) — an empty sum already satisfies it, so this node fires with no input at all".into(),
            ));
        } else if (t as usize) <= d {
            ok += 1;
        } else if is_door_target(i) {
            fails.push(issue(
                "threshold exceeds in_degree AND the node is a door target — the door waits on a node that can never fire, so the level can never be won".into(),
            ));
        } else if is_inverted_target(i) {
            warns.push(issue(
                "threshold exceeds in_degree AND the node is an INVERTED target ([door] target_off) — the inverse goal is trivially satisfied, because this node can never be lit in the first place".into(),
            ));
        } else {
            dead_by_design.push(issue(
                "threshold exceeds in_degree and the node is not a door target — DEAD BY DESIGN (permanently unreachable route-around content)".into(),
            ));
        }
    }

    let tail = format!(
        "{ok} satisfiable, {} dead-by-design, {inputs_skipped} input node(s) skipped (threshold unused)",
        dead_by_design.len()
    );

    if !fails.is_empty() {
        return Check::new(
            "thresholds",
            Severity::Fail,
            format!("{} — {tail}", threshold_issue_lines(&fails)),
        )
        .at_all(threshold_locations(&fails));
    }
    if !warns.is_empty() {
        return Check::new(
            "thresholds",
            Severity::Warn,
            format!("{} — {tail}", threshold_issue_lines(&warns)),
        )
        .at_all(threshold_locations(&warns));
    }
    if dead_by_design.is_empty() {
        Check::new(
            "thresholds",
            Severity::Pass,
            format!("every node satisfies 0 < threshold <= in_degree ({tail})"),
        )
    } else {
        // A PASS that carries locations, deliberately: this line exists to be
        // REVIEWED ("is that scar declared or accidental?"), and a reviewer who
        // cannot walk to the node cannot answer. `jump_targets` still skips it -
        // it is documentation, not a defect.
        Check::new(
            "thresholds",
            Severity::Pass,
            format!(
                "DEAD BY DESIGN: {} — {tail}",
                threshold_issue_lines(&dead_by_design)
            ),
        )
        .at_all(threshold_locations(&dead_by_design))
    }
}

fn threshold_locations(issues: &[ThresholdIssue]) -> Vec<Location> {
    issues.iter().map(|t| Location::cell(t.cell)).collect()
}

fn threshold_issue_lines(issues: &[ThresholdIssue]) -> String {
    issues
        .iter()
        .map(|t| {
            format!(
                "node '{}' (threshold {}, in_degree {}): {}",
                t.node, t.threshold, t.in_degree, t.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

// ===========================================================================
// 12. Per-level lighting (`[light]` + `[atmosphere]`)
// ===========================================================================
//
// The lighting spec (§1.3, 2026-09-02) gave a level four new ways to author
// light: repeatable `[light]` point lights, an `[atmosphere]` key direction /
// colour, an ambient floor, and height fog. Every one of them is a value the
// PARSER accepts happily and the RENDERER may then quietly ignore — which is
// this project's signature defect class (authored, parsed, tested, never on
// screen), so it gets a rule rather than an author's eye.
//
// Three ways an authored light does nothing, none of which the parser can see:
//
//   * THE CAP. `re_render`'s `PointLights::from_world` bakes at most
//     `MAX_POINT_LIGHTS` (4) lights and then breaks out of the loop. A fifth
//     `[light]` section is dropped in total silence — the level file says six
//     lights, the frame shows four, and nothing anywhere says so.
//   * INERT VALUES. `from_world` also skips a light whose radius is not `> 0`,
//     whose intensity is not `> 0`, or whose position is not finite. `intensity
//     = 0` in particular parses fine (`re_level` only requires `>= 0`) and is a
//     perfectly ordinary way to accidentally author darkness.
//   * PLACEMENT. A light outside the room footprint, below the floor, above the
//     ceiling, or buried inside a wall cell is baked and costs its slot, but it
//     is not lighting the room the player is standing in.
//
// ## The rule
//
//   PASS  no `[light]` and no `[atmosphere]` override — "engine defaults", said
//         once and quietly. That is every shipped level today, and a linter that
//         shouts at the normal case is a linter people learn to skip.
//   PASS  lighting is authored and every light is live, in-room and within the
//         cap. Names the count and what `[atmosphere]` overrides.
//   WARN  anything above. NEVER FAIL: a lighting mistake is cosmetic — the level
//         still loads, still solves, and still wins. Refusing a level over a
//         mis-placed lamp would be a worse defect than the lamp.
//
// The inert/cap arithmetic MIRRORS `PointLights::from_world` exactly, including
// the detail that an inert light does NOT consume a cap slot (it is skipped
// before the counter moves), so "the 5th light is dropped" here means the same
// lights the renderer drops. Hand-pinned, like the `[note]` binder constants:
// `re_lvlcheck` must not depend on `re_render` (the dependency direction is
// linter → data crates only).

/// The renderer's punctual-light cap.
///
/// `re_render::ascii_shader::MAX_POINT_LIGHTS` is the AUTHORITY; this is a
/// hand-pinned copy, because the linter depends only on the data crates
/// (`re_level` / `re_world` / `re_puzzle`) and never on the renderer. If the
/// renderer's cap moves, this constant moves with it.
pub const MAX_POINT_LIGHTS: usize = 4;

/// One authored `[light]` that the renderer will drop, or that is baked
/// somewhere it cannot light the room.
#[derive(Clone, Debug, PartialEq)]
pub struct LightIssue {
    /// Index of the `[light]` section in parse order.
    pub index: usize,
    /// The authored `pos`, exactly as written (`x`, `y` floor plane, `z` height).
    pub pos: (f32, f32, f32),
    /// The cell that position falls in (`floor` per axis) — where a cursor goes.
    /// `None` only when `pos` is not finite and therefore names no cell at all;
    /// the parser refuses such a light, so that case is reachable from a
    /// synthetic / editor-built `Level` only.
    pub cell: Option<(i32, i32, i32)>,
    /// What is wrong, in report words.
    pub reason: String,
}

/// The cell a light's position falls in, or `None` for a non-finite position.
/// `floor` per axis — the same world-to-cell mapping the camera and the DDA use.
fn light_cell(l: &re_world::PointLight) -> Option<(i32, i32, i32)> {
    l.pos.is_finite().then(|| {
        (
            l.pos.x.floor() as i32,
            l.pos.y.floor() as i32,
            l.pos.z.floor() as i32,
        )
    })
}

/// Audit every `[light]`: the issues, and how many lights actually reach the
/// renderer's light set. See the section comment for the mirrored semantics.
fn audit_lights(level: &Level) -> (Vec<LightIssue>, usize) {
    if level.lights.is_empty() {
        return (Vec::new(), 0);
    }
    let map = level.build_map();
    let (sx, sy, sz) = map.size();
    let storey = level.storey_height();
    let mut issues: Vec<LightIssue> = Vec::new();
    let mut live = 0usize;

    for (index, l) in level.lights.iter().enumerate() {
        let pos = (l.pos.x, l.pos.y, l.pos.z);
        let cell = light_cell(l);
        let mut push = |reason: String| {
            issues.push(LightIssue {
                index,
                pos,
                cell,
                reason,
            });
        };

        // (a) INERT — the same filter, in the same order, as `from_world`. An
        // inert light is skipped BEFORE the cap counter moves, so it costs no
        // slot; it simply does not exist at render time.
        let (radius, intensity) = (l.radius, l.intensity);
        if !(radius.is_finite() && radius > 0.0) {
            push(format!(
                "radius {radius} is not > 0 — INERT: the renderer drops this light and it never reaches a surface"
            ));
            continue;
        }
        if !(intensity.is_finite() && intensity > 0.0) {
            push(format!(
                "intensity {intensity} is not > 0 — INERT: the renderer drops this light, so the authored colour never appears"
            ));
            continue;
        }
        let Some((cx, cy, cz)) = cell else {
            push("pos is not finite — INERT: the renderer drops this light".to_string());
            continue;
        };

        // (b) THE CAP — past it the light is dropped whatever else is right
        // about it, so that is the whole finding.
        if live >= MAX_POINT_LIGHTS {
            push(format!(
                "past the renderer's {MAX_POINT_LIGHTS}-light cap (only the first {MAX_POINT_LIGHTS} live lights are baked) — SILENTLY DROPPED; merge or remove a light"
            ));
            continue;
        }
        live += 1;

        // (c) PLACEMENT — baked, costs a slot, lights the wrong part of the world.
        if cx < 0 || cy < 0 || cx as usize >= sx || cy as usize >= sy {
            push(format!(
                "(x,y) is outside the room footprint ({sx}x{sy} cells) — it burns a light slot and lights nothing inside the level"
            ));
        } else if cz < 0 || cz as usize >= sz {
            let side = if cz < 0 {
                "below the floor"
            } else {
                "above the ceiling"
            };
            push(format!(
                "z is {side} (the room is {sz} cell(s) tall, storey height {storey}) — it burns a light slot outside the habitable band"
            ));
        } else {
            let kind = map.cell_at(cx, cy, cz);
            if kind.is_solid() {
                push(format!(
                    "buried inside {kind:?} geometry at cell ({cx},{cy},{cz}) — lighting is UNSHADOWED, so it bleeds through the wall it sits in instead of lighting the room"
                ));
            }
        }
    }
    (issues, live)
}

/// Every authored `[light]` the renderer will drop or bake somewhere useless.
///
/// Public so tests (and a future editor drawer) can assert on the exact findings
/// instead of scraping the report text — the same arrangement
/// [`emit_part_losses`] has.
#[must_use]
pub fn light_issues(level: &Level) -> Vec<LightIssue> {
    audit_lights(level).0
}

/// `key_dir, ambient 0.18, postfx 'clean'` — what `[atmosphere]` actually
/// overrides, or an empty string when it overrides nothing.
fn atmosphere_summary(atmos: &re_level::LevelAtmosphere) -> String {
    let mut parts: Vec<String> = Vec::new();
    if atmos.key_dir.is_some() {
        parts.push("key_dir".to_string());
    }
    if atmos.key_color.is_some() {
        parts.push("key_color".to_string());
    }
    if let Some(a) = atmos.ambient {
        parts.push(format!("ambient {a}"));
    }
    if let Some(f) = atmos.fog_height {
        parts.push(format!(
            "fog_height (base {}, falloff {})",
            f.base, f.falloff
        ));
    }
    if let Some(p) = &atmos.postfx {
        parts.push(format!("postfx '{p}'"));
    }
    parts.join(", ")
}

// ── 12b. Can the level's SURFACES even receive its lamps? ───────────────────
//
// A `[light]` is only half the authoring. The other half is the surfaces it
// falls on, and one whole class of recipe cannot receive light at all:
//
//   `re_render`'s relief pass returns EARLY for a SELF-LIT recipe — projected
//   light, phosphor, hologram sheen — before `light_surface` is ever called.
//   A wall skinned with `scanline` / `truchet` / `fractal` / `hologram` /
//   `codewall` / a bare `image` therefore ignores every lamp in the room.
//
// This is the project's signature defect class in its purest form: the lamps
// parse, bake, pass every unit test, and light nothing a player can see. It cost
// the lighting demo level a whole authoring pass before anyone noticed, which is
// exactly the evidence that a human eye is the wrong instrument for it.
//
// WARN-only, and only when the level actually authors a `[light]` — a level with
// no lamps is free to skin every surface with phosphor.

/// Does this recipe's shader IGNORE the scene light?
///
/// Hand-pinned mirror of `re_render::ascii_shader::TextureRecipe::props().self_lit`
/// (`re_lvlcheck` depends only on the data crates, never on the renderer, so the
/// authority cannot be imported). The match is EXHAUSTIVE on purpose: a new
/// `MaterialRecipe` variant breaks this build until someone declares which side
/// it is on, and `ts_editor`'s `lighting_self_lit_agrees` test fails if a pinned
/// answer drifts from the renderer's.
pub fn recipe_is_self_lit(r: &re_world::MaterialRecipe) -> bool {
    use re_world::MaterialRecipe as R;
    match r {
        R::Hologram { .. }
        | R::CodeWall { .. }
        | R::NoteWall { .. }
        | R::Scanline { .. }
        | R::Truchet { .. }
        | R::Fractal { .. } => true,
        // An image is a self-lit decal UNTIL it authors `relief =`, which is
        // precisely what turns it into a surface the light falls on.
        R::Image { relief, .. } => relief.is_none(),
        R::Brick { .. }
        | R::Circuit { .. }
        | R::Grain { .. }
        | R::Sky { .. }
        | R::Substrate { .. }
        | R::Voice { .. }
        // The window grid is a physical wall (glass + mullion in the room's
        // light), not a self-lit projection — see `re_render::ascii_shader`'s
        // `TextureRecipe::Facade` arm of `props()`, which this mirrors.
        | R::Facade { .. } => false,
    }
}

/// The five surface slots, paired with the material name each is bound to.
fn surface_bindings(level: &Level) -> [(&'static str, Option<&String>); 5] {
    let s = &level.surface_materials;
    [
        ("wall_solid", s.wall_solid.as_ref()),
        ("wall_trace", s.wall_trace.as_ref()),
        ("wall_matrix", s.wall_matrix.as_ref()),
        ("floor", s.floor.as_ref()),
        ("ceiling", s.ceiling.as_ref()),
    ]
}

/// Surfaces bound to a self-lit material, as `(surface, material, recipe)`.
/// A slot with no binding, or one whose material takes its recipe from a
/// `preset =` (which only the renderer can resolve) or from the surface built-in,
/// is skipped rather than guessed at.
pub fn light_deaf_surfaces(level: &Level) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for (surface, bound) in surface_bindings(level) {
        let Some(name) = bound else { continue };
        let Some(m) = level.materials.iter().find(|m| &m.name == name) else {
            continue;
        };
        let Some(recipe) = m.recipe.as_ref() else {
            continue;
        };
        // An `Image` recipe here is the parser's placeholder (relief always None):
        // the authored `relief =` lives on `m.image_relief` and is only folded in
        // by `Level::material_to_def` at build time, which this lint never runs.
        let self_lit = if matches!(recipe, re_world::MaterialRecipe::Image { .. }) {
            m.image_relief.is_none()
        } else {
            recipe_is_self_lit(recipe)
        };
        if self_lit {
            let kind = format!("{recipe:?}");
            let kind = kind
                .split(&[' ', '{'][..])
                .next()
                .unwrap_or("?")
                .to_string();
            out.push((surface.to_string(), name.clone(), kind.to_lowercase()));
        }
    }
    out
}

/// WARN when a level authors lamps its own surfaces cannot receive.
fn check_lit_surfaces(level: &Level) -> Check {
    if level.lights.is_empty() {
        return Check::new(
            "lit-surfaces",
            Severity::Pass,
            "no [light] authored — surface recipes are unconstrained",
        );
    }
    let deaf = light_deaf_surfaces(level);
    if deaf.is_empty() {
        return Check::new(
            "lit-surfaces",
            Severity::Pass,
            "every bound surface uses a recipe that receives light",
        );
    }
    let list = deaf
        .iter()
        .map(|(s, m, k)| format!("{s} → [material {m}] ({k})"))
        .collect::<Vec<_>>()
        .join(", ");
    Check::new(
        "lit-surfaces",
        Severity::Warn,
        format!(
            "{} of 5 surface slot(s) use a SELF-LIT recipe and will ignore this \
             level's {} lamp(s): {list}. A self-lit recipe (hologram, codewall, \
             notewall, scanline, truchet, fractal, or an image without `relief =`) \
             returns before the lighting pass. Use brick / circuit / grain / sky / \
             substrate / voice, or give the image a `relief =`",
            deaf.len(),
            level.lights.len(),
        ),
    )
}

/// The per-level lighting rule. WARN-only — see the section comment.
fn check_lighting(level: &Level) -> Check {
    let atmo = atmosphere_summary(&level.atmos);
    let authored = level.lights.len();

    if authored == 0 {
        if atmo.is_empty() {
            // The normal case, said once and quietly: a level that authors no
            // lighting is not making a mistake, it is taking the engine default.
            return Check::new(
                "lighting",
                Severity::Pass,
                "no authored lighting (engine defaults)",
            );
        }
        return Check::new(
            "lighting",
            Severity::Pass,
            format!("no [light] point lights; [atmosphere] overrides {atmo}"),
        );
    }

    let (issues, live) = audit_lights(level);
    let atmo_tail = if atmo.is_empty() {
        "no [atmosphere] key/ambient override".to_string()
    } else {
        format!("[atmosphere] overrides {atmo}")
    };

    if issues.is_empty() {
        return Check::new(
            "lighting",
            Severity::Pass,
            format!(
                "{authored} point light(s), all live, in-room and within the {MAX_POINT_LIGHTS}-light cap; {atmo_tail}"
            ),
        );
    }
    Check::new(
        "lighting",
        Severity::Warn,
        format!(
            "{} of {authored} authored light(s) do not light the room ({live} of {authored} reach the renderer's light set; {atmo_tail}): {}",
            issues.len(),
            light_issue_lines(&issues)
        ),
    )
    // A light with a non-finite `pos` names no cell and so contributes no
    // location; every other finding is located at the cell its `pos` falls in,
    // in the order `detail` names them.
    .at_all(
        issues
            .iter()
            .filter_map(|i| i.cell.map(Location::cell))
            .collect(),
    )
}

fn light_issue_lines(issues: &[LightIssue]) -> String {
    issues
        .iter()
        .map(|l| {
            format!(
                "light {} at ({:.2},{:.2},{:.2}): {}",
                l.index, l.pos.0, l.pos.1, l.pos.2, l.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

// ===========================================================================
// helpers
// ===========================================================================

/// A cell triple as floats, for the coherence distance math.
fn as_f32((x, y, z): (i32, i32, i32)) -> (f32, f32, f32) {
    (x as f32, y as f32, z as f32)
}

fn dist2_xy(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

fn dist3(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn yn(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn load_failure_report(source: &str, msg: &str, line: usize) -> LevelReport {
    LevelReport {
        source: source.to_string(),
        level_name: String::new(),
        title: String::new(),
        size: (0, 0, 0),
        inventory: None,
        solve: None,
        reachable: None,
        coherent: None,
        // A level that did not parse has no asset references to speak of.
        broken_assets: 0,
        checks: vec![
            Check::new("load", Severity::Fail, format!("failed to parse: {msg}")).at_all(
                if line > 0 {
                    vec![Location::SourceLine(line)]
                } else {
                    Vec::new()
                },
            ),
        ],
        verdict: format!("LOAD FAILED ({msg})"),
    }
}

/// True if the report's solve verdict represents a level that at least *can* be
/// solved by toggling (Solvable or TriviallySolved — both have a winning state).
/// Used by the integration sweep over the shipped levels.
pub fn solve_is_winnable(v: SolveVerdict) -> bool {
    matches!(
        v,
        SolveVerdict::Solvable { .. } | SolveVerdict::TriviallySolved
    )
}

mod fields_rule;
mod no_stuck;
mod props_rule;

#[cfg(test)]
mod tests;
