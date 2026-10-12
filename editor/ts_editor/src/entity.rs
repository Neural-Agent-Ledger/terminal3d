//! Object-selector entity placement / editing / deletion.
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal. Entities live in
//! typed `re_level::Level` fields (`door`, `player`, `pedestal`, `nodes`), which
//! `re_voximport::to_lvl_text` already serializes, so placing them here and
//! saving via `io::save_lvl_text` round-trips them with no extra format work.
//!
//! Validate-before-mutate: every path either applies a complete, well-defined
//! change (and flips `dirty`) or leaves the level untouched. There is no panic,
//! no half-applied entity, and every cursor position has a defined result.
use crate::state::{
    cycle_facing, entity_field_default, step_entity_field, EditorState, EmitterField, EntityField,
    EntityKind,
};
use re_world::{BoldMode, ColorMode, EmitterBlend, EmitterKind, EmitterOrigin, EmitterSpec};
// Scalable Material Overlay newtypes — NOT re-exported by re_level, so imported
// from re_core directly (ts_editor depends on both crates).
use re_core::{InsetId, LineMaterialId, MaterialId};

/// Preset explicit glyph-set overrides the `glyph_set` field cycles. Index 0 is
/// `None` ("use the family"); the rest are small bespoke sequences. Cycling lets
/// an author bypass the family registry entirely without typing characters.
pub(crate) const GLYPH_SETS: [Option<&[char]>; 5] = [
    None,
    Some(&['.', 'o', 'O', '@']),
    Some(&['`', '\'', '"', '*']),
    Some(&[':', ';', '+', '#']),
    Some(&['-', '=', '+', '*', '#']),
];

/// Short label for the current `glyph_set` (the joined chars, or `none`).
fn glyph_set_label(e: &EmitterSpec) -> String {
    glyph_set_option_label(e.glyph_set.as_ref())
}

/// Step `color` to the next/prev palette swatch. Delegates to
/// [`crate::material::step_palette`] so the editor has ONE colour stepping rule:
/// on-palette values wrap ±1, off-palette values land on the NEAREST swatch
/// instead of snapping to white (which destroyed the authored hue of nearly
/// every object in the shipped corpus).
fn step_stop(color: u32, delta: i32) -> u32 {
    crate::material::step_palette(color, delta)
}

/// Short label for an `EmitterKind`.
fn kind_label(k: EmitterKind) -> &'static str {
    match k {
        EmitterKind::Standard => "Standard",
        EmitterKind::GodRays => "GodRays",
    }
}

/// Short label for an `EmitterBlend`.
fn blend_label(b: EmitterBlend) -> &'static str {
    match b {
        EmitterBlend::KeepBg => "KeepBg",
        EmitterBlend::Opaque => "Opaque",
    }
}

/// Read the emitter origin as an `(x, y, z)` triple. A `Screen` origin reports
/// its `(col, row, 0)` so the origin nudges still have a defined value to show /
/// edit (the nudges operate on a `World` origin; a `Screen` one is left as-is).
fn origin_xyz(e: &EmitterSpec) -> (f32, f32, f32) {
    match e.origin {
        EmitterOrigin::World(p) => (p.x, p.y, p.z),
        EmitterOrigin::Screen { col, row } => (col, row, 0.0),
    }
}

/// Nudge one axis of a `World` origin by `d` (cells). A `Screen` origin is left
/// unchanged (origin nudges are a world-space fine-tune). `axis`: 0=x,1=y,2=z.
fn nudge_origin(e: &mut EmitterSpec, axis: u8, d: f32) {
    if let EmitterOrigin::World(mut p) = e.origin {
        match axis {
            0 => p.x += d,
            1 => p.y += d,
            _ => p.z += d,
        }
        e.origin = EmitterOrigin::World(p);
    }
}

/// Default pedestal half-extent volume when first placed (1×1×1 footprint).
const PEDESTAL_DEFAULT_VOL: i32 = 1;
/// Maximum pedestal extent the grow key allows (keeps it inside small rooms).
const PEDESTAL_MAX_VOL: i32 = 9;

/// Candidate top-down glyphs the `NodeGlyph` field cycles through (`c`). Plain
/// ASCII so they survive the `[node] glyph` round-trip (one char). `N` is the
/// default the placer assigns; the rest are common circuit/neuron marks.
pub(crate) const NODE_GLYPHS: [char; 10] = ['N', 'O', '*', '#', '+', '@', 'X', '%', '=', '&'];

/// The list a node-glyph `c` press walks: [`NODE_GLYPHS`] plus every off-list
/// glyph this session has seen, plus `current` — see [`crate::vocab`].
pub fn node_glyph_options(state: &EditorState, current: char) -> Vec<char> {
    crate::vocab::options_at(&NODE_GLYPHS, &state.vocab.node_glyph, &current).0
}

/// Step a node's glyph `delta` places along [`node_glyph_options`], wrapping.
///
/// A glyph the canonical list does not offer is REMEMBERED first, so it stays in
/// the cycle and a full lap returns to it. It used to restart at index 0 (`N`),
/// which destroyed 168 of the 209 `glyph =` values the shipped corpus authors —
/// `R`, `A`, `◉`… — with no key able to get any of them back.
fn step_node_glyph(state: &mut EditorState, current: char, delta: i32) -> char {
    crate::vocab::remember(&NODE_GLYPHS, &mut state.vocab.node_glyph, &current);
    crate::vocab::step(&NODE_GLYPHS, &state.vocab.node_glyph, &current, delta)
}

/// Step a node `color_rgb` to the next/prev shared-palette swatch (the same
/// palette the emitter colour stops and material fg use, so the editor has one
/// colour vocabulary).
fn step_node_color(color: u32, delta: i32) -> u32 {
    step_stop(color, delta)
}

/// The six cardinal emission directions the `DirectionAxis` enum-field cycles
/// through, as `(unit_vector, label)`. `+Z` (straight up) is the default.
pub(crate) const DIRECTION_AXES: [([f32; 3], &str); 6] = [
    ([0.0, 0.0, 1.0], "+Z"),
    ([0.0, 0.0, -1.0], "-Z"),
    ([1.0, 0.0, 0.0], "+X"),
    ([-1.0, 0.0, 0.0], "-X"),
    ([0.0, 1.0, 0.0], "+Y"),
    ([0.0, -1.0, 0.0], "-Y"),
];

/// World cell occupied by an emitter's `World` origin (floor of each axis). A
/// `Screen`-anchored emitter has no world cell (`None`).
fn emitter_cell(e: &EmitterSpec) -> Option<(i32, i32, i32)> {
    match e.origin {
        EmitterOrigin::World(p) => {
            Some((p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32))
        }
        EmitterOrigin::Screen { .. } => None,
    }
}

/// Index of the emitter whose world cell is `(x, y, z)`, if any (the editor uses
/// this to SELECT rather than duplicate when an emitter already sits on the cell;
/// it also drives the 2D marker overlay and the `x`-key spatial delete).
///
/// SYNTHESIZED emitters (`EmitterSpec::synthesized`, e.g. the `[effects] snow`
/// lattice) are never returned: they are not editor objects — there is nothing
/// to select, mark, duplicate, or delete, since they are regenerated wholesale
/// from a knob on every load and are never persisted by `to_lvl_text`. Without
/// this filter, standing the cursor on a snow-lattice cell and pressing `x`
/// would delete a live emitter that reappears (fully, not incrementally) on the
/// very next load — a confusing, ineffective delete.
pub fn emitter_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    state
        .level
        .emitters
        .iter()
        .position(|e| !e.synthesized && emitter_cell(e) == Some((x, y, z)))
}

/// A short label for the emitter's current emission axis: the cardinal preset's
/// name when it is one of the six, else the vector itself.
///
/// It used to render every non-cardinal direction as the single word `custom`,
/// which made two different authored directions indistinguishable in the panel
/// AND in the enum popup — the popup then had no row it could highlight. Showing
/// the vector keeps every direction its own, nameable option.
fn direction_label(e: &EmitterSpec) -> String {
    direction_axis_label(&[e.direction.x, e.direction.y, e.direction.z])
}

/// [`direction_label`] over a bare vector — shared with the option list so the
/// popup rows and the panel readout are literally the same text.
pub fn direction_axis_label(v: &[f32; 3]) -> String {
    DIRECTION_AXES
        .iter()
        .find(|&&(a, _)| crate::vocab::dir_eq(&a, v))
        .map_or_else(
            || format!("{:.3} {:.3} {:.3}", v[0], v[1], v[2]),
            |&(_, l)| l.to_string(),
        )
}

/// The six cardinal vectors of [`DIRECTION_AXES`], for [`crate::vocab`].
pub fn direction_canon() -> Vec<[f32; 3]> {
    DIRECTION_AXES.iter().map(|&(v, _)| v).collect()
}

/// The [`GLYPH_SETS`] presets as owned values, for [`crate::vocab`].
pub fn glyph_set_canon() -> Vec<Option<Vec<char>>> {
    GLYPH_SETS.iter().map(|g| g.map(<[char]>::to_vec)).collect()
}

/// Label one `glyph_set` option (the joined chars, or `none`) — shared by the
/// panel readout and the option list.
pub fn glyph_set_option_label(g: Option<&Vec<char>>) -> String {
    g.map_or_else(|| "none".to_string(), |cs| cs.iter().collect::<String>())
}

/// Short label for a `ColorMode`.
fn color_mode_label(m: ColorMode) -> &'static str {
    match m {
        ColorMode::RampOverAge => "RampOverAge",
        ColorMode::RandomPick => "RandomPick",
        ColorMode::Fixed => "Fixed",
    }
}

/// Short label for a `BoldMode`.
fn bold_label(b: BoldMode) -> &'static str {
    match b {
        BoldMode::Off => "Off",
        BoldMode::On => "On",
        BoldMode::ByProximity => "ByProximity",
        BoldMode::ByAge => "ByAge",
    }
}

/// Read a trigger's region as `(min, max)` inclusive corners. The only region
/// shape is [`re_core::Region::Box`]; a future analytic region (non-exhaustive
/// enum) reports a degenerate point so the editor never panics.
fn trigger_box(t: &re_core::Trigger) -> ((i32, i32, i32), (i32, i32, i32)) {
    match t.region {
        re_core::Region::Box { min, max } => (min, max),
        // `Region` is `#[non_exhaustive]`; any future variant is shown as an
        // origin point (inert) rather than panicking.
        _ => ((0, 0, 0), (0, 0, 0)),
    }
}

/// Replace a trigger's box region with new inclusive corners (normalised so
/// `min <= max` per axis). Other region shapes are overwritten with a Box (the
/// editor only authors boxes).
fn set_trigger_box(t: &mut re_core::Trigger, min: (i32, i32, i32), max: (i32, i32, i32)) {
    let lo = (min.0.min(max.0), min.1.min(max.1), min.2.min(max.2));
    let hi = (min.0.max(max.0), min.1.max(max.1), min.2.max(max.2));
    t.region = re_core::Region::Box { min: lo, max: hi };
}

/// Index of the trigger whose region CONTAINS cell `(x, y, z)`, if any (the editor
/// SELECTS rather than duplicates when a trigger already covers the cursor). The
/// SELECTED trigger is preferred when several overlap, so re-placing on a covered
/// cell keeps editing the same one; otherwise the first containing trigger wins.
pub fn trigger_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_trigger {
        if state
            .level
            .triggers
            .get(sel)
            .is_some_and(|t| t.contains(x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .triggers
        .iter()
        .position(|t| t.contains(x, y, z))
}

/// A compact label for a trigger's action (`activate NODE` / `event LABEL`),
/// resolving an `ActivateNode` index to a node name where possible.
pub fn trigger_action_label(state: &EditorState, t: &re_core::Trigger) -> String {
    match &t.action {
        re_core::TriggerAction::ActivateNode { name, node_index } => {
            // Prefer the live node name at the resolved index (the editor keeps the
            // name in sync), falling back to the authored `name` string.
            let shown = state
                .level
                .nodes
                .get(*node_index)
                .map(|n| n.name.as_str())
                .unwrap_or(name.as_str());
            format!("activate {shown}")
        }
        re_core::TriggerAction::Event { label } => format!("event {label}"),
        // `TriggerAction` is `#[non_exhaustive]`; render unknown variants safely.
        _ => "?".to_string(),
    }
}

// ── ZONE INPUT helpers ───────────────────────────────────────────────────────

/// A zone input's inclusive box corners `((minx,miny,minz),(maxx,maxy,maxz))`.
/// The corners are stored normalised by the editor's `set_zone_box`.
fn zone_box(z: &re_level::ZoneInput) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((z.x0, z.y0, z.z0), (z.x1, z.y1, z.z1))
}

/// Write a zone input's box corners (normalised so `min <= max` per axis).
fn set_zone_box(z: &mut re_level::ZoneInput, min: (i32, i32, i32), max: (i32, i32, i32)) {
    z.x0 = min.0.min(max.0);
    z.y0 = min.1.min(max.1);
    z.z0 = min.2.min(max.2);
    z.x1 = min.0.max(max.0);
    z.y1 = min.1.max(max.1);
    z.z1 = min.2.max(max.2);
}

/// Index of the zone input whose box CONTAINS cell `(x, y, z)`, if any, with the
/// SELECTED zone preferred when several overlap (so re-placing keeps editing the
/// same one). Mirrors [`trigger_at`].
pub fn zone_input_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_zone_input {
        if state
            .level
            .zone_inputs
            .get(sel)
            .is_some_and(|zi| zi.contains(x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .zone_inputs
        .iter()
        .position(|zi| zi.contains(x, y, z))
}

/// The level's INPUT node indices, in node order — the candidates a zone's
/// `ZoneNode` field cycles. A zone MUST reference an input node (re_level finalize
/// enforces it), so cycling is restricted to these.
fn input_node_indices(state: &EditorState) -> Vec<usize> {
    state
        .level
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.is_input)
        .map(|(i, _)| i)
        .collect()
}

// ── NOTE helpers ─────────────────────────────────────────────────────────────

/// The default body a freshly-placed note carries (a valid non-empty body so the
/// note round-trips before the author types anything).
const NOTE_DEFAULT_TEXT: &str = "note";

// ── WALL DECAL helpers ───────────────────────────────────────────────────────

/// A wall decal's inclusive box corners `((minx,miny,minz),(maxx,maxy,maxz))`.
/// The corners are stored normalised by the editor's `set_decal_box`.
fn decal_box(d: &re_level::WallDecal) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((d.x0, d.y0, d.z0), (d.x1, d.y1, d.z1))
}

/// Write a wall decal's box corners (normalised so `min <= max` per axis).
fn set_decal_box(d: &mut re_level::WallDecal, min: (i32, i32, i32), max: (i32, i32, i32)) {
    d.x0 = min.0.min(max.0);
    d.y0 = min.1.min(max.1);
    d.z0 = min.2.min(max.2);
    d.x1 = min.0.max(max.0);
    d.y1 = min.1.max(max.1);
    d.z1 = min.2.max(max.2);
}

/// `true` when cell `(x, y, z)` lies inside the decal's inclusive box.
fn decal_contains(d: &re_level::WallDecal, x: i32, y: i32, z: i32) -> bool {
    let ((lx, ly, lz), (hx, hy, hz)) = decal_box(d);
    x >= lx && x <= hx && y >= ly && y <= hy && z >= lz && z <= hz
}

/// Index of the wall decal whose box CONTAINS cell `(x, y, z)`, if any, with the
/// SELECTED decal preferred when several overlap (so re-placing keeps editing the
/// same one). Mirrors [`zone_input_at`] / [`trigger_at`].
pub fn decal_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_decal {
        if state
            .level
            .decals
            .get(sel)
            .is_some_and(|d| decal_contains(d, x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .decals
        .iter()
        .position(|d| decal_contains(d, x, y, z))
}

/// A unique auto name (`decal1`, `decal2`, …) for a freshly-placed decal, skipping
/// any name already taken so a `[wall_decal]` never collides on its key.
fn unique_decal_name(state: &EditorState) -> String {
    let mut k = state.level.decals.len() + 1;
    loop {
        let name = format!("decal{k}");
        if !state.level.decals.iter().any(|d| d.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Default head/tail colours a freshly-placed decal carries (a bright amber head
/// fading to a dark amber tail — readable distinct hues, valid `#RRGGBB`).
const DECAL_DEFAULT_HEAD: u32 = 0x00_FF_A0_00;
const DECAL_DEFAULT_TAIL: u32 = 0x00_40_28_00;

/// Helper: render the SELECTED decal's box corners via `f`, or `"-"` when none is
/// selected. Mirrors [`zone_field_str`].
fn decal_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state
        .selected_decal
        .filter(|&i| i < state.level.decals.len())
    {
        Some(i) => f(decal_box(&state.level.decals[i])),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED decal via `f`, or `"-"` when none is
/// selected.
fn decal_value(state: &EditorState, f: impl Fn(&re_level::WallDecal) -> String) -> String {
    match state
        .selected_decal
        .filter(|&i| i < state.level.decals.len())
    {
        Some(i) => f(&state.level.decals[i]),
        None => "-".to_string(),
    }
}

// ── HOLOGRAM helpers ─────────────────────────────────────────────────────────

/// The `style` tag a freshly-placed hologram carries (0 = default blue look).
const HOLO_DEFAULT_STYLE: u16 = 0;

/// A hologram's inclusive box corners `((minx,miny,minz),(maxx,maxy,maxz))`. The
/// corners are stored normalised by the editor's `set_holo_box`.
fn holo_box(h: &re_level::LevelHolo) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((h.x0, h.y0, h.z0), (h.x1, h.y1, h.z1))
}

/// Write a hologram's box corners (normalised so `min <= max` per axis).
fn set_holo_box(h: &mut re_level::LevelHolo, min: (i32, i32, i32), max: (i32, i32, i32)) {
    h.x0 = min.0.min(max.0);
    h.y0 = min.1.min(max.1);
    h.z0 = min.2.min(max.2);
    h.x1 = min.0.max(max.0);
    h.y1 = min.1.max(max.1);
    h.z1 = min.2.max(max.2);
}

/// `true` when cell `(x, y, z)` lies inside the hologram's inclusive box.
fn holo_contains(h: &re_level::LevelHolo, x: i32, y: i32, z: i32) -> bool {
    let ((lx, ly, lz), (hx, hy, hz)) = holo_box(h);
    x >= lx && x <= hx && y >= ly && y <= hy && z >= lz && z <= hz
}

/// Index of the hologram whose box CONTAINS cell `(x, y, z)`, if any, with the
/// SELECTED hologram preferred when several overlap (so re-placing keeps editing
/// the same one). Mirrors [`decal_at`].
pub fn holo_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_holo {
        if state
            .level
            .holos
            .get(sel)
            .is_some_and(|h| holo_contains(h, x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .holos
        .iter()
        .position(|h| holo_contains(h, x, y, z))
}

/// A unique auto name (`holo1`, `holo2`, …) for a freshly-placed hologram, skipping
/// any name already taken so a `[holo]` never collides on its key.
fn unique_holo_name(state: &EditorState) -> String {
    let mut k = state.level.holos.len() + 1;
    loop {
        let name = format!("holo{k}");
        if !state.level.holos.iter().any(|h| h.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Helper: render the SELECTED hologram's box corners via `f`, or `"-"` when none
/// is selected. Mirrors [`decal_field_str`].
fn holo_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state.selected_holo.filter(|&i| i < state.level.holos.len()) {
        Some(i) => f(holo_box(&state.level.holos[i])),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED hologram via `f`, or `"-"` when none is
/// selected.
fn holo_value(state: &EditorState, f: impl Fn(&re_level::LevelHolo) -> String) -> String {
    match state.selected_holo.filter(|&i| i < state.level.holos.len()) {
        Some(i) => f(&state.level.holos[i]),
        None => "-".to_string(),
    }
}

// ── WARP PORTAL helpers ──────────────────────────────────────────────────────
//
// A portal is a PAIR of linked box regions (A and B). SSOT for paint + hit-test:
// `portal_a_box`/`portal_b_box` read the corners, `set_portal_a_box`/
// `set_portal_b_box` write them (normalised), `portal_contains` is true when a
// cell lies in EITHER region, `portal_at` selects the portal covering the cursor,
// and `unique_portal_name` mints `portal1`, `portal2`, … keys. Mirrors the
// HOLOGRAM helpers block above but with two regions per entity.

/// Portal Region A's inclusive box corners `((minx,miny,minz),(maxx,maxy,maxz))`.
/// Stored normalised by [`set_portal_a_box`].
fn portal_a_box(p: &re_level::LevelPortal) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((p.ax0, p.ay0, p.az0), (p.ax1, p.ay1, p.az1))
}

/// Portal Region B's inclusive box corners. Stored normalised by [`set_portal_b_box`].
fn portal_b_box(p: &re_level::LevelPortal) -> ((i32, i32, i32), (i32, i32, i32)) {
    ((p.bx0, p.by0, p.bz0), (p.bx1, p.by1, p.bz1))
}

/// Write Portal Region A's box corners (normalised so `min <= max` per axis).
fn set_portal_a_box(p: &mut re_level::LevelPortal, min: (i32, i32, i32), max: (i32, i32, i32)) {
    p.ax0 = min.0.min(max.0);
    p.ay0 = min.1.min(max.1);
    p.az0 = min.2.min(max.2);
    p.ax1 = min.0.max(max.0);
    p.ay1 = min.1.max(max.1);
    p.az1 = min.2.max(max.2);
}

/// Write Portal Region B's box corners (normalised so `min <= max` per axis).
fn set_portal_b_box(p: &mut re_level::LevelPortal, min: (i32, i32, i32), max: (i32, i32, i32)) {
    p.bx0 = min.0.min(max.0);
    p.by0 = min.1.min(max.1);
    p.bz0 = min.2.min(max.2);
    p.bx1 = min.0.max(max.0);
    p.by1 = min.1.max(max.1);
    p.bz1 = min.2.max(max.2);
}

/// `true` when cell `(x, y, z)` lies inside EITHER of the portal's two regions.
fn portal_contains(p: &re_level::LevelPortal, x: i32, y: i32, z: i32) -> bool {
    let inside = |((lx, ly, lz), (hx, hy, hz)): ((i32, i32, i32), (i32, i32, i32))| {
        x >= lx && x <= hx && y >= ly && y <= hy && z >= lz && z <= hz
    };
    inside(portal_a_box(p)) || inside(portal_b_box(p))
}

/// Index of the portal whose EITHER region CONTAINS cell `(x, y, z)`, if any, with
/// the SELECTED portal preferred when several overlap (so re-placing keeps editing
/// the same one). Mirrors [`holo_at`].
pub fn portal_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_portal {
        if state
            .level
            .portals
            .get(sel)
            .is_some_and(|p| portal_contains(p, x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .portals
        .iter()
        .position(|p| portal_contains(p, x, y, z))
}

/// A unique auto name (`portal1`, `portal2`, …) for a freshly-placed portal,
/// skipping any name already taken so a `[portal]` never collides on its key.
fn unique_portal_name(state: &EditorState) -> String {
    let mut k = state.level.portals.len() + 1;
    loop {
        let name = format!("portal{k}");
        if !state.level.portals.iter().any(|p| p.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Helper: render the SELECTED portal's box corners (region A or B, chosen by the
/// caller's `f` reading from one of [`portal_a_box`]/[`portal_b_box`]) via `f`, or
/// `"-"` when none is selected. Mirrors [`holo_field_str`].
fn portal_field_str(state: &EditorState, f: impl Fn(&re_level::LevelPortal) -> String) -> String {
    match state
        .selected_portal
        .filter(|&i| i < state.level.portals.len())
    {
        Some(i) => f(&state.level.portals[i]),
        None => "-".to_string(),
    }
}

/// One portal region as inclusive `(min, max)` cell corners.
type PortalFace = ((i32, i32, i32), (i32, i32, i32));

/// `true` when `region` is a flat wall face on a room boundary — the SAME rule
/// `re_level::finalize`/`validate_portal_face` enforces: EXACTLY one axis has
/// equal endpoints (the flat axis), that flat axis sits on a grid boundary
/// (`0` or `size-1`), and every cell is in bounds. Kept in lock-step with the
/// loader so the editor can warn BEFORE a save produces an unreloadable level.
fn is_valid_portal_face(region: PortalFace, size: (usize, usize, usize)) -> bool {
    let ((x0, y0, z0), (x1, y1, z1)) = region;
    let (sx, sy, sz) = (size.0 as i32, size.1 as i32, size.2 as i32);
    let flat = [x0 == x1, y0 == y1, z0 == z1];
    if flat.iter().filter(|&&f| f).count() != 1 {
        return false;
    }
    let on_boundary = if flat[0] {
        x0 == 0 || x0 == sx - 1
    } else if flat[1] {
        y0 == 0 || y0 == sy - 1
    } else {
        z0 == 0 || z0 == sz - 1
    };
    if !on_boundary {
        return false;
    }
    let in_range = |c: i32, hi: i32| (0..hi).contains(&c);
    in_range(x0.min(x1), sx)
        && in_range(x0.max(x1), sx)
        && in_range(y0.min(y1), sy)
        && in_range(y0.max(y1), sy)
        && in_range(z0.min(z1), sz)
        && in_range(z0.max(z1), sz)
}

/// `true` when BOTH of a portal's regions are valid wall faces with matching
/// opening areas — i.e. the portal would survive `re_level::finalize`. Used by
/// the authoring guard to warn when an edit leaves a portal unreloadable.
fn portal_is_reloadable(p: &re_level::LevelPortal, size: (usize, usize, usize)) -> bool {
    let a = portal_a_box(p);
    let b = portal_b_box(p);
    let area = |((x0, y0, z0), (x1, y1, z1)): PortalFace| {
        let mut s = [(x1 - x0).abs(), (y1 - y0).abs(), (z1 - z0).abs()];
        s.sort_unstable();
        (s[1], s[2])
    };
    is_valid_portal_face(a, size) && is_valid_portal_face(b, size) && area(a) == area(b)
}

/// Build a DEFAULT pair of valid, reloadable portal wall faces for a freshly
/// placed portal in a `size` room, anchored near the cursor `(_x, y, z)`.
///
/// Both faces are flat-X openings on OPPOSITE X room boundaries (Region A on the
/// nearer X wall, Region B on the far one), each spanning a small in-bounds
/// extent on Y and Z centred on the cursor. Identical Y/Z spans guarantee the two
/// openings have the SAME area (the loader rejects a mismatch), and a flat-X axis
/// on a boundary is a valid wall face — so the default round-trips through
/// `re_level::load_string` with no manual fix-up.
fn default_portal_faces(
    cursor: (i32, i32, i32),
    size: (usize, usize, usize),
) -> (PortalFace, PortalFace) {
    let (cx, cy, cz) = cursor;
    let (sx, sy, sz) = (size.0 as i32, size.1 as i32, size.2 as i32);
    // Region A on the X wall nearest the cursor; Region B on the opposite wall.
    let near_x = if cx <= (sx - 1) / 2 { 0 } else { sx - 1 };
    let far_x = if near_x == 0 { sx - 1 } else { 0 };
    // A ≤2-cell opening on each in-plane axis, clamped to fit the room. `span`
    // returns an inclusive `(lo, hi)` of `len` cells centred on `c` inside `0..ext`.
    let span = |c: i32, ext: i32| -> (i32, i32) {
        let len = ext.min(2); // 1 or 2 cells depending on room depth
        let lo = c.clamp(0, (ext - len).max(0));
        (lo, lo + len - 1)
    };
    let (y0, y1) = span(cy, sy);
    let (z0, z1) = span(cz, sz);
    (
        ((near_x, y0, z0), (near_x, y1, z1)),
        ((far_x, y0, z0), (far_x, y1, z1)),
    )
}

/// The skin / bound-material cycle vocabulary: `(none)` (the empty string)
/// followed by every material NAME the level defines, then every DANGLING name
/// this session has seen (`state.vocab.material_name`), then `current` itself.
///
/// The tail matters: a zone / line / portal may name a material the level does
/// not define — an import, a rename in progress, a hand-authored forward
/// reference. Cycling one used to snap it to `(none)`, silently deleting the
/// binding with no key able to type the name back. Now it is an option like any
/// other. See [`crate::vocab`].
pub fn material_name_options(state: &EditorState, current: &str) -> Vec<String> {
    let mut canon = vec!["(none)".to_string()];
    canon.extend(state.level.materials.iter().map(|m| m.name.clone()));
    let cur = if current.is_empty() {
        "(none)".to_string()
    } else {
        current.to_string()
    };
    crate::vocab::options_at(&canon, &state.vocab.material_name, &cur).0
}

/// Step a skin / bound-material name one place along [`material_name_options`],
/// REMEMBERING an off-list current so the lap closes. Returns the new LABEL
/// (`(none)` for "unbound").
fn step_material_name(state: &mut EditorState, current: &str, delta: i32) -> String {
    let mut canon = vec!["(none)".to_string()];
    canon.extend(state.level.materials.iter().map(|m| m.name.clone()));
    let cur = if current.is_empty() {
        "(none)".to_string()
    } else {
        current.to_string()
    };
    crate::vocab::remember(&canon, &mut state.vocab.material_name, &cur);
    crate::vocab::step(&canon, &state.vocab.material_name, &cur, delta)
}

// ── SCALABLE MATERIAL OVERLAY: ZONE helpers ──────────────────────────────────
//
// SSOT for the zone's BOX region (the only `Region` variant the editor authors):
// `zone_box` reads the corners, `set_zone_overlay_box` writes them (normalised,
// always producing a `Region::Box`), `zone_overlay_contains` hit-tests, and
// `zone_overlay_at` selects the zone covering the cursor. `unique_zone_name` mints
// `zone1`, `zone2`, … keys.

/// An overlay zone's inclusive box corners. `Region` is `#[non_exhaustive]`; a
/// future analytic variant reports a degenerate origin point (inert) so the editor
/// never panics (mirrors [`trigger_box`]).
fn overlay_zone_box(z: &re_level::OverlayZone) -> ((i32, i32, i32), (i32, i32, i32)) {
    match z.region {
        re_core::Region::Box { min, max } => (min, max),
        _ => ((0, 0, 0), (0, 0, 0)),
    }
}

/// Write an overlay zone's box region with new inclusive corners (normalised so
/// `min <= max` per axis). Always produces a `Region::Box` (the only authored shape).
fn set_zone_overlay_box(z: &mut re_level::OverlayZone, min: (i32, i32, i32), max: (i32, i32, i32)) {
    let lo = (min.0.min(max.0), min.1.min(max.1), min.2.min(max.2));
    let hi = (min.0.max(max.0), min.1.max(max.1), min.2.max(max.2));
    z.region = re_core::Region::Box { min: lo, max: hi };
}

/// `true` when cell `(x, y, z)` lies inside the overlay zone's region.
fn zone_overlay_contains(z: &re_level::OverlayZone, x: i32, y: i32, zc: i32) -> bool {
    z.region.contains(x, y, zc)
}

/// Index of the overlay zone whose region CONTAINS cell `(x, y, z)`, if any, with
/// the SELECTED zone preferred when several overlap (so re-placing keeps editing the
/// same one). Mirrors [`holo_at`].
pub fn zone_overlay_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_zone {
        if state
            .level
            .overlay
            .zones
            .get(sel)
            .is_some_and(|zn| zone_overlay_contains(zn, x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .overlay
        .zones
        .iter()
        .position(|zn| zone_overlay_contains(zn, x, y, z))
}

/// A unique auto name (`zone1`, `zone2`, …) for a freshly-placed overlay zone,
/// skipping any name already taken so a `[zone]` never collides on its key.
fn unique_zone_name(state: &EditorState) -> String {
    let mut k = state.level.overlay.zones.len() + 1;
    loop {
        let name = format!("zone{k}");
        if !state.level.overlay.zones.iter().any(|z| z.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Read the SelfSimilar params carried by a [`re_level::OverlayScaleRule`], or the
/// inert default when the rule is not SelfSimilar (so the arms/warp/twist readouts
/// have a defined value to show).
fn scale_params(rule: &re_level::OverlayScaleRule) -> re_level::OverlaySelfSimilarParams {
    match rule {
        re_level::OverlayScaleRule::SelfSimilar(p) => *p,
        _ => re_level::OverlaySelfSimilarParams::default(),
    }
}

/// Helper: render the SELECTED zone's box corners via `f`, or `"-"` when none is
/// selected. Mirrors [`holo_field_str`].
fn zone_overlay_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state
        .selected_zone
        .filter(|&i| i < state.level.overlay.zones.len())
    {
        Some(i) => f(overlay_zone_box(&state.level.overlay.zones[i])),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED zone via `f`, or `"-"` when none.
fn zone_overlay_value(state: &EditorState, f: impl Fn(&re_level::OverlayZone) -> String) -> String {
    match state
        .selected_zone
        .filter(|&i| i < state.level.overlay.zones.len())
    {
        Some(i) => f(&state.level.overlay.zones[i]),
        None => "-".to_string(),
    }
}

/// The selected zone's inset-membership readout: the cursor inset's name, whether
/// the zone references it, and the cursor position within the inset list. `"-"` when
/// no zone is selected; a hint when the level has no insets. Mirrors
/// [`door_target_value`].
fn zone_insets_value(state: &EditorState) -> String {
    if state
        .selected_zone
        .filter(|&i| i < state.level.overlay.zones.len())
        .is_none()
    {
        return "-".to_string();
    }
    let insets = &state.level.overlay.insets;
    if insets.is_empty() {
        return "no insets".to_string();
    }
    let Some(zi) = state.selected_zone else {
        return "-".to_string();
    };
    let idx = state.zone_inset_cursor.min(insets.len() - 1);
    let name = &insets[idx].name;
    let on = state.level.overlay.zones[zi]
        .insets
        .iter()
        .any(|i| i.0 == *name);
    let mark = if on { "ON" } else { "-" };
    format!("[{}/{}] '{name}' {mark}", idx + 1, insets.len())
}

// ── SCALABLE MATERIAL OVERLAY: LINE helpers ──────────────────────────────────
//
// A line is a polyline (`LinePath { points: Vec<Vec3> }`). SSOT: `line_at` selects
// the line whose path passes through the cursor cell (mirrors `pipe_at`),
// `unique_line_name` mints keys. Waypoint append/remove + render live in the
// `overlay` module (REUSING the pipe path model where the storage differs).

/// Index of the overlay line whose path passes through cell `(x, y, z)` (floor of
/// each waypoint), if any, with the SELECTED line preferred when several overlap.
/// Mirrors [`crate::pipe::pipe_at`].
pub fn line_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    let on_path = |l: &re_level::OverlayLine| {
        l.path
            .points
            .iter()
            .any(|p| p.x.floor() as i32 == x && p.y.floor() as i32 == y && p.z.floor() as i32 == z)
    };
    if let Some(sel) = state.selected_line {
        if state.level.overlay.lines.get(sel).is_some_and(on_path) {
            return Some(sel);
        }
    }
    state.level.overlay.lines.iter().position(on_path)
}

/// Helper: render a field of the SELECTED line via `f`, or `"-"` when none.
fn line_value(state: &EditorState, f: impl Fn(&re_level::OverlayLine) -> String) -> String {
    match state
        .selected_line
        .filter(|&i| i < state.level.overlay.lines.len())
    {
        Some(i) => f(&state.level.overlay.lines[i]),
        None => "-".to_string(),
    }
}

// ── SCALABLE MATERIAL OVERLAY: INSET helpers ─────────────────────────────────
//
// An inset is a box region at an octave range carrying a child-scene reference.
// SSOT mirrors the zone helpers.

/// An overlay inset's inclusive box corners (the only authored `Region` variant).
fn inset_box(ins: &re_level::OverlayInset) -> ((i32, i32, i32), (i32, i32, i32)) {
    match ins.region {
        re_core::Region::Box { min, max } => (min, max),
        _ => ((0, 0, 0), (0, 0, 0)),
    }
}

/// Write an overlay inset's box region (normalised so `min <= max` per axis).
fn set_inset_box(ins: &mut re_level::OverlayInset, min: (i32, i32, i32), max: (i32, i32, i32)) {
    let lo = (min.0.min(max.0), min.1.min(max.1), min.2.min(max.2));
    let hi = (min.0.max(max.0), min.1.max(max.1), min.2.max(max.2));
    ins.region = re_core::Region::Box { min: lo, max: hi };
}

/// `true` when cell `(x, y, z)` lies inside the inset's region.
fn inset_contains(ins: &re_level::OverlayInset, x: i32, y: i32, z: i32) -> bool {
    ins.region.contains(x, y, z)
}

/// Index of the overlay inset whose region CONTAINS cell `(x, y, z)`, if any, with
/// the SELECTED inset preferred when several overlap. Mirrors [`zone_overlay_at`].
pub fn inset_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_inset {
        if state
            .level
            .overlay
            .insets
            .get(sel)
            .is_some_and(|ins| inset_contains(ins, x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .overlay
        .insets
        .iter()
        .position(|ins| inset_contains(ins, x, y, z))
}

/// A unique auto name (`inset1`, `inset2`, …) for a freshly-placed overlay inset.
fn unique_inset_name(state: &EditorState) -> String {
    let mut k = state.level.overlay.insets.len() + 1;
    loop {
        let name = format!("inset{k}");
        if !state.level.overlay.insets.iter().any(|i| i.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Helper: render the SELECTED inset's box corners via `f`, or `"-"` when none.
fn inset_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state
        .selected_inset
        .filter(|&i| i < state.level.overlay.insets.len())
    {
        Some(i) => f(inset_box(&state.level.overlay.insets[i])),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED inset via `f`, or `"-"` when none.
fn inset_value(state: &EditorState, f: impl Fn(&re_level::OverlayInset) -> String) -> String {
    match state
        .selected_inset
        .filter(|&i| i < state.level.overlay.insets.len())
    {
        Some(i) => f(&state.level.overlay.insets[i]),
        None => "-".to_string(),
    }
}

// ── PART helpers ─────────────────────────────────────────────────────────────

/// A unique auto name (`part1`, `part2`, …) for a freshly-placed part, skipping
/// any name already taken so a `[part]` never collides on its key.
fn unique_part_name(state: &EditorState) -> String {
    let mut k = state.level.parts.len() + 1;
    loop {
        let name = format!("part{k}");
        if !state.level.parts.iter().any(|p| p.name == name) {
            return name;
        }
        k += 1;
    }
}

/// Index of the part whose ANCHOR cell (`off_x, off_y, off_z`) is exactly
/// `(x, y, z)`, if any. The editor SELECTS rather than duplicates when a part is
/// already anchored on the cursor; the SELECTED part is preferred when several
/// stack. Mirrors [`note_at`].
pub fn part_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_part {
        if state
            .level
            .parts
            .get(sel)
            .is_some_and(|p| (p.off_x, p.off_y, p.off_z) == (x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .parts
        .iter()
        .position(|p| (p.off_x, p.off_y, p.off_z) == (x, y, z))
}

/// The selected node's emit readout: the cursor part's name, whether the node
/// currently lights it, and the cursor position within the part list. `"-"` when
/// no node is selected; a hint when the level has no parts. Mirrors
/// [`door_target_value`].
fn node_emits_value(state: &EditorState) -> String {
    if state
        .selected_node
        .filter(|&i| i < state.level.nodes.len())
        .is_none()
    {
        return "-".to_string();
    }
    let parts = &state.level.parts;
    if parts.is_empty() {
        return "no parts".to_string();
    }
    let Some(node) = state.selected_node else {
        return "-".to_string();
    };
    let idx = state.emit_part_cursor.min(parts.len() - 1);
    let name = &parts[idx].name;
    let on = state
        .level
        .emits
        .iter()
        .any(|e| e.source_node == node && e.part == idx);
    let mark = if on { "ON" } else { "-" };
    format!("[{}/{}] '{name}' {mark}", idx + 1, parts.len())
}

/// Index of the note whose cell is exactly `(x, y, z)`, if any (the editor SELECTS
/// rather than duplicates when a note already sits on the cursor). The SELECTED
/// note is preferred when several stack.
pub fn note_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_note {
        if state
            .level
            .notes
            .get(sel)
            .is_some_and(|n| (n.x, n.y, n.z) == (x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .notes
        .iter()
        .position(|n| (n.x, n.y, n.z) == (x, y, z))
}

/// Enter the object selector (if off) or cycle to the next entity kind (if on).
/// Toggling the selector clears any per-entity selection so a stale node edit
/// doesn't leak across kinds.
pub fn cycle_selector(state: &mut EditorState) {
    let kind = match state.place {
        None => EntityKind::Door,
        Some(k) => k.next(),
    };
    state.place = Some(kind);
    // The object selector and material mode are mutually exclusive.
    state.material_mode = false;
    state.selected_node = None;
    state.selected_emitter = None;
    state.selected_trigger = None;
    state.selected_pipe = None;
    state.selected_zone_input = None;
    state.selected_note = None;
    state.selected_decal = None;
    state.selected_holo = None;
    state.selected_portal = None;
    state.selected_zone = None;
    state.selected_line = None;
    state.selected_inset = None;
    state.selected_part = None;
    state.pipe_pending = None;
    // Reset the focused field to the new kind's first field so `;`/`'`/`c`/`+`/`-`
    // target a valid property for this kind (no stale focus across kinds).
    if let Some(f) = entity_field_default(kind) {
        state.entity_field = f;
    }
    state.door_target_cursor = 0;
    state.emit_part_cursor = 0;
    state.zone_inset_cursor = 0;
    state.status = format!("place: {}", kind.label());
}

/// Place (or select) the current entity kind at the cursor. No-op (returns the
/// state unchanged, no `dirty`) when the selector is off.
pub fn place_at_cursor(state: &mut EditorState) {
    let Some(kind) = state.place else { return };
    // Entity edits change what build_map stamps (nodes/door/decals/
    // holos/regions) — schedule the debounced preview rebuild so the 3D view
    // catches up ~150 ms after the edit burst (asset-bearing levels skip the
    // auto path and hint Shift+R; see vox::tick_preview_rebuild).
    state.schedule_preview_rebuild();
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    match kind {
        EntityKind::Door => {
            state.level.door.x = x;
            state.level.door.y = y;
            state.level.door.z = z;
            state.mark_dirty();
            state.status = format!("door @ {x},{y},{z}");
        }
        EntityKind::Spawn => {
            state.level.player.x = x;
            state.level.player.y = y;
            state.level.player.z = z;
            state.mark_dirty();
            state.status = format!(
                "spawn @ {x},{y},{z} facing {}",
                crate::state::facing_label(state.level.player.facing_yaw_col)
            );
        }
        EntityKind::Pedestal => {
            let p = &mut state.level.pedestal;
            p.cx = x;
            p.cy = y;
            p.cz = z;
            // A freshly-placed pedestal gets a sane default volume if it has none
            // yet (an unset pedestal is all-zero, which serializes to volume 1).
            if p.w <= 0 {
                p.w = PEDESTAL_DEFAULT_VOL;
            }
            if p.h <= 0 {
                p.h = PEDESTAL_DEFAULT_VOL;
            }
            if p.d <= 0 {
                p.d = PEDESTAL_DEFAULT_VOL;
            }
            state.mark_dirty();
            state.status = format!("pedestal @ {x},{y},{z}");
        }
        EntityKind::Part => {
            // Re-placing on an existing part's anchor SELECTS it for editing rather
            // than pushing a duplicate (which would also collide on name).
            if let Some(idx) = part_at(state, x, y, z) {
                state.selected_part = Some(idx);
                state.emit_part_cursor = idx;
                let name = state.level.parts[idx].name.clone();
                state.status = format!("selected part '{name}'");
            } else {
                // A fresh 2×2×2 all-solid part anchored at the cursor cell, white.
                let name = unique_part_name(state);
                let part = crate::part::new_part(name.clone(), 2, 2, 2, (x, y, z), 0xFF_FFFF);
                state.level.parts.push(part);
                let idx = state.level.parts.len() - 1;
                state.selected_part = Some(idx);
                state.emit_part_cursor = idx;
                state.mark_dirty();
                state.status = format!("part '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Node => {
            // Re-placing on an existing node SELECTS it for editing rather than
            // pushing a duplicate (which would also collide on name).
            if let Some(idx) = state.level.node_at(x, y, z) {
                state.selected_node = Some(idx);
                let name = state.level.nodes[idx].name.clone();
                state.status = format!("selected node '{name}'");
            } else {
                let idx = state.level.push_node(x, y, z);
                state.selected_node = Some(idx);
                let name = state.level.nodes[idx].name.clone();
                state.mark_dirty();
                state.status = format!("node '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Emitter => {
            // Re-placing on an existing emitter SELECTS it for editing rather
            // than pushing a duplicate.
            if let Some(idx) = emitter_at(state, x, y, z) {
                state.selected_emitter = Some(idx);
                state.status = format!("selected emitter #{idx}");
            } else {
                // A fresh emitter anchored at the cell CENTRE with sane defaults
                // (the validated gentle-haze default), then validated so it is
                // immediately render-safe and visible.
                let mut spec = EmitterSpec::default().with_origin(EmitterOrigin::World(
                    glam::Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
                ));
                spec.validate();
                state.level.emitters.push(spec);
                let idx = state.level.emitters.len() - 1;
                state.selected_emitter = Some(idx);
                state.mark_dirty();
                state.status = format!("emitter #{idx} @ {x},{y},{z}");
            }
        }
        EntityKind::Trigger => {
            // Re-placing on a cell already covered by a trigger SELECTS it rather
            // than stacking a duplicate region.
            if let Some(idx) = trigger_at(state, x, y, z) {
                state.selected_trigger = Some(idx);
                state.status = format!("selected trigger #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor. Default action: drive the FIRST
                // node when one exists (the most useful event), else a generic
                // `event` so a trigger placed before any node still round-trips.
                let action = if let Some(n) = state.level.nodes.first() {
                    re_core::TriggerAction::activate_node(n.name.clone())
                } else {
                    re_core::TriggerAction::event("event")
                };
                let mut t = re_core::Trigger {
                    region: re_core::Region::Box {
                        min: (x, y, z),
                        max: (x, y, z),
                    },
                    action,
                    once: true,
                };
                // Resolve the ActivateNode index to node 0 so the engine + the
                // panel show a live target immediately (re_level::finalize would
                // otherwise do this on the next load).
                if let re_core::TriggerAction::ActivateNode { node_index, .. } = &mut t.action {
                    *node_index = 0;
                }
                state.level.triggers.push(t);
                let idx = state.level.triggers.len() - 1;
                state.selected_trigger = Some(idx);
                state.mark_dirty();
                state.status = format!("trigger #{idx} @ {x},{y},{z}");
            }
        }
        EntityKind::Pipe => {
            // PIPE is wired by a two-node connect flow, not point-placed. `space`
            // on a NODE picks the source (then the target on the next press); on a
            // path cell of an existing pipe it SELECTS that pipe for editing.
            if let Some(node) = state.level.node_at(x, y, z) {
                if state.pipe_pending.is_none() {
                    crate::pipe::start_pipe_create(state);
                } else {
                    crate::pipe::pick_node(state, node);
                }
            } else if let Some(pi) = crate::pipe::pipe_at(state, x, y, z) {
                state.selected_pipe = Some(pi);
                state.pipe_pending = None;
                state.status = format!("selected pipe #{pi}");
            } else {
                state.status =
                    "pipe: put the cursor on a node (source/target) or a pipe path".into();
            }
        }
        EntityKind::ZoneInput => {
            // Re-placing on a covered cell SELECTS the zone rather than stacking a
            // duplicate region.
            if let Some(idx) = zone_input_at(state, x, y, z) {
                state.selected_zone_input = Some(idx);
                state.status = format!("selected zone input #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor, linked to the FIRST input node
                // when one exists (the most useful default), else an unresolved name
                // (re_level finalize rejects an unresolved zone, so the author must
                // link an input node before it loads — surfaced in the panel).
                let inputs = input_node_indices(state);
                let (node, node_index) = match inputs.first() {
                    Some(&i) => (state.level.nodes[i].name.clone(), i),
                    None => (String::new(), 0),
                };
                state.zone_node_cursor = node_index;
                let mut zi = re_level::ZoneInput {
                    name: format!("zone{}", state.level.zone_inputs.len()),
                    node,
                    node_index,
                    ..Default::default()
                };
                set_zone_box(&mut zi, (x, y, z), (x, y, z));
                state.level.zone_inputs.push(zi);
                let idx = state.level.zone_inputs.len() - 1;
                state.selected_zone_input = Some(idx);
                state.mark_dirty();
                state.status = format!("zone input #{idx} @ {x},{y},{z}");
            }
        }
        EntityKind::Note => {
            // Re-placing on a note's cell SELECTS it rather than pushing a duplicate.
            if let Some(idx) = note_at(state, x, y, z) {
                state.selected_note = Some(idx);
                state.status = format!("selected note #{idx}");
            } else {
                state.level.notes.push(re_level::LevelNote {
                    kind: re_level::NoteKind::Researcher,
                    x,
                    y,
                    z,
                    facing: 0,
                    text: NOTE_DEFAULT_TEXT.to_string(),
                });
                let idx = state.level.notes.len() - 1;
                state.selected_note = Some(idx);
                state.mark_dirty();
                state.status = format!("note #{idx} @ {x},{y},{z}");
            }
        }
        EntityKind::WallDecal => {
            // Re-placing on a covered cell SELECTS the decal rather than stacking a
            // duplicate region.
            if let Some(idx) = decal_at(state, x, y, z) {
                state.selected_decal = Some(idx);
                state.status = format!("selected wall decal #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor, a `trace` surface, full density,
                // and the default amber head→tail colours.
                let name = unique_decal_name(state);
                let mut d = re_level::WallDecal {
                    name: name.clone(),
                    surface_kind: re_world::CellKind::WallTrace,
                    density: 1.0,
                    color_head: DECAL_DEFAULT_HEAD,
                    color_tail: DECAL_DEFAULT_TAIL,
                    ..Default::default()
                };
                set_decal_box(&mut d, (x, y, z), (x, y, z));
                state.level.decals.push(d);
                let idx = state.level.decals.len() - 1;
                state.selected_decal = Some(idx);
                state.mark_dirty();
                state.status = format!("wall decal '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Holo => {
            // Re-placing on a covered cell SELECTS the hologram rather than stacking
            // a duplicate region.
            if let Some(idx) = holo_at(state, x, y, z) {
                state.selected_holo = Some(idx);
                state.status = format!("selected hologram #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor, a `box` shape, the default style.
                let name = unique_holo_name(state);
                let mut h = re_level::LevelHolo {
                    name: name.clone(),
                    shape: re_world::HoloShape::Box,
                    style: HOLO_DEFAULT_STYLE,
                    ..Default::default()
                };
                set_holo_box(&mut h, (x, y, z), (x, y, z));
                state.level.holos.push(h);
                let idx = state.level.holos.len() - 1;
                state.selected_holo = Some(idx);
                state.mark_dirty();
                state.status = format!("hologram '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Portal => {
            // Re-placing on a cell covered by EITHER region SELECTS the existing
            // portal rather than stacking a duplicate pair.
            if let Some(idx) = portal_at(state, x, y, z) {
                state.selected_portal = Some(idx);
                state.status = format!("selected portal #{idx}");
            } else {
                // PLACEMENT DEFAULTS: a 1×1×1 region is THREE flat axes, which
                // `re_level::finalize` hard-rejects ("got 3 flat axes") — the saved
                // .lvl could not be reopened. Seed two VALID, reloadable wall faces
                // instead: flat-X openings on opposite X room boundaries, near the
                // cursor, with identical (matching-area) Y/Z spans. rotate = 0.0,
                // skin = "". The author then edits both.
                let name = unique_portal_name(state);
                let size = state.map.size();
                let (face_a, face_b) = default_portal_faces((x, y, z), size);
                let mut p = re_level::LevelPortal {
                    name: name.clone(),
                    rotate: 0.0,
                    skin: String::new(),
                    ..Default::default()
                };
                set_portal_a_box(&mut p, face_a.0, face_a.1);
                set_portal_b_box(&mut p, face_b.0, face_b.1);
                state.level.portals.push(p);
                let idx = state.level.portals.len() - 1;
                state.selected_portal = Some(idx);
                state.mark_dirty();
                let (ax0, ay0, az0) = face_a.0;
                let (bx0, by0, bz0) = face_b.0;
                state.status = format!("portal '{name}' A@{ax0},{ay0},{az0} B@{bx0},{by0},{bz0}");
            }
        }
        EntityKind::Zone => {
            // Re-placing on a covered cell SELECTS the zone rather than stacking a
            // duplicate region.
            if let Some(idx) = zone_overlay_at(state, x, y, z) {
                state.selected_zone = Some(idx);
                state.status = format!("selected zone #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor, unbound material, ScaleRule::None,
                // no insets (all defaults the serializer SKIPS, so it round-trips inert).
                let name = unique_zone_name(state);
                let mut zn = re_level::OverlayZone {
                    name: name.clone(),
                    ..Default::default()
                };
                set_zone_overlay_box(&mut zn, (x, y, z), (x, y, z));
                state.level.overlay.zones.push(zn);
                let idx = state.level.overlay.zones.len() - 1;
                state.selected_zone = Some(idx);
                state.mark_dirty();
                state.status = format!("zone '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Line => {
            // A line is a polyline; `space` APPENDS a waypoint at the cursor. On a
            // cursor that already lies on an existing line's path, SELECT that line.
            crate::overlay::line_place_or_extend(state, x, y, z);
        }
        EntityKind::Inset => {
            // Re-placing on a covered cell SELECTS the inset rather than stacking a
            // duplicate region.
            if let Some(idx) = inset_at(state, x, y, z) {
                state.selected_inset = Some(idx);
                state.status = format!("selected inset #{idx}");
            } else {
                // A fresh 1×1×1 box at the cursor, octave (0,0), empty child ref.
                let name = unique_inset_name(state);
                let mut ins = re_level::OverlayInset {
                    name: name.clone(),
                    ..Default::default()
                };
                set_inset_box(&mut ins, (x, y, z), (x, y, z));
                state.level.overlay.insets.push(ins);
                let idx = state.level.overlay.insets.len() - 1;
                state.selected_inset = Some(idx);
                state.mark_dirty();
                state.status = format!("inset '{name}' @ {x},{y},{z}");
            }
        }
        EntityKind::Field => {
            // Re-placing on an existing field's cell SELECTS it rather than
            // pushing a duplicate (see `crate::field`'s module doc — this
            // editor authors the Sphere/Room shapes; `field_at` only ever
            // matches a standalone Sphere field, so a Room field must be
            // re-selected via the object tree/`;`/`'` paging instead).
            if let Some(idx) = crate::field::field_at(state, x, y, z) {
                state.selected_field = Some(idx);
                let name = state.level.field_defs[idx].name.clone();
                state.status = format!("selected field '{name}'");
            } else {
                let name = crate::field::unique_field_name(state);
                let def = re_level::fields::FieldDef {
                    name: name.clone(),
                    line: 0,
                    host: re_level::fields::FieldHost::Standalone,
                    keys: vec![
                        re_level::fields::FieldKey {
                            key: "kind".to_string(),
                            value: "lethe_wash".to_string(),
                            line: 0,
                        },
                        re_level::fields::FieldKey {
                            key: "at".to_string(),
                            value: format!("{x}, {y}, {z}"),
                            line: 0,
                        },
                    ],
                };
                state.level.field_defs.push(def);
                let idx = state.level.field_defs.len() - 1;
                state.selected_field = Some(idx);
                crate::field::re_resolve(state);
                state.mark_dirty();
                state.status = format!("field '{name}' @ {x},{y},{z} (lethe_wash)");
            }
        }
    }
}

/// Delete the entity at the cursor for the current kind. Door/Spawn/Pedestal are
/// cleared back to the origin/default (one-per-level singletons); a Node at the
/// cursor is removed from `nodes`. A no-match cursor is a defined no-op.
pub fn delete_at_cursor(state: &mut EditorState) {
    let Some(kind) = state.place else { return };
    // Entity edits change what build_map stamps (nodes/door/decals/
    // holos/regions) — schedule the debounced preview rebuild so the 3D view
    // catches up ~150 ms after the edit burst (asset-bearing levels skip the
    // auto path and hint Shift+R; see vox::tick_preview_rebuild).
    state.schedule_preview_rebuild();
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    match kind {
        EntityKind::Door => {
            if (state.level.door.x, state.level.door.y, state.level.door.z) == (x, y, z) {
                state.level.door = re_level::Door::default();
                state
                    .level
                    .hier_fixup_after_remove(re_level::HierKind::Door, 0);
                state.mark_dirty();
                state.status = "door cleared".into();
            } else {
                state.status = "no door here".into();
            }
        }
        EntityKind::Spawn => {
            if (
                state.level.player.x,
                state.level.player.y,
                state.level.player.z,
            ) == (x, y, z)
            {
                state.level.player = re_level::PlayerSpawn::default();
                state
                    .level
                    .hier_fixup_after_remove(re_level::HierKind::Spawn, 0);
                state.mark_dirty();
                state.status = "spawn cleared".into();
            } else {
                state.status = "no spawn here".into();
            }
        }
        EntityKind::Pedestal => {
            if (
                state.level.pedestal.cx,
                state.level.pedestal.cy,
                state.level.pedestal.cz,
            ) == (x, y, z)
            {
                state.level.pedestal = re_level::Pedestal::default();
                state
                    .level
                    .hier_fixup_after_remove(re_level::HierKind::Pedestal, 0);
                state.mark_dirty();
                state.status = "pedestal cleared".into();
            } else {
                state.status = "no pedestal here".into();
            }
        }
        EntityKind::Part => {
            if let Some(idx) = part_at(state, x, y, z) {
                let name = state.level.parts[idx].name.clone();
                state.level.parts.remove(idx);
                // REFERENTIAL INTEGRITY: `re_level` has no part-index fixup, so the
                // editor maintains the `[emit]` clauses — drop clauses lighting the
                // removed part and shift down references to later parts.
                state.level.emits.retain(|e| e.part != idx);
                for e in &mut state.level.emits {
                    if e.part > idx {
                        e.part -= 1;
                    }
                }
                // Part is not in the object-nesting hierarchy (no `HierKind::Part`),
                // so no hierarchy fixup — just keep the selection valid.
                state.selected_part = match state.selected_part {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.emit_part_cursor = state
                    .emit_part_cursor
                    .min(state.level.parts.len().saturating_sub(1));
                state.mark_dirty();
                state.status = format!("removed part '{name}'");
            } else {
                state.status = "no part here".into();
            }
        }
        EntityKind::Node => {
            if let Some(idx) = state.level.node_at(x, y, z) {
                let name = state.level.nodes[idx].name.clone();
                // FULL index fixup: `remove_node` removes the node AND shifts/drops
                // EVERY index-based node reference — `door.target_parts`/`target_off`
                // (win targets), `zone_inputs[].node_index`, `pipes[].from_node`/
                // `.to_node`, and the `[hierarchy]` node links. A raw `Vec::remove`
                // + hierarchy-only fixup (the old code) silently re-pointed a door's
                // win condition or a pipe endpoint at the wrong node (B2).
                state.level.remove_node(idx);
                // B1: `[inputs]` is keyed by node NAME (not index), so `remove_node`
                // can't touch it. A leftover `[inputs] <name> = 1` for the deleted
                // node makes `re_level` finalize reject the reload with "[inputs]
                // references unknown node". Drop the deleted node's entry here, the
                // same name-keyed cleanup style as the `emits` index fixup below.
                state.level.inputs.retain(|(n, _)| n != &name);
                // `remove_node` fixes door/zone/pipe/hierarchy node refs but NOT the
                // `[emit]` clauses (re_level has no emit fixup), so the editor does it
                // here: drop clauses sourced by the removed node and shift down later
                // source-node refs, keeping node->part integrity.
                state.level.emits.retain(|e| e.source_node != idx);
                for e in &mut state.level.emits {
                    if e.source_node > idx {
                        e.source_node -= 1;
                    }
                }
                // Drop the selection if it pointed at (or past) the removed node
                // so the io/threshold keys never edit a shifted-in neighbour.
                state.selected_node = match state.selected_node {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed node '{name}'");
            } else {
                state.status = "no node here".into();
            }
        }
        EntityKind::Emitter => {
            if let Some(idx) = emitter_at(state, x, y, z) {
                state.level.emitters.remove(idx);
                state
                    .level
                    .hier_fixup_after_remove(re_level::HierKind::Emitter, idx);
                // Keep the selection pointing at the same logical emitter (or
                // drop it) after the Vec shifts — never edit a shifted neighbour.
                state.selected_emitter = match state.selected_emitter {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed emitter #{idx}");
            } else {
                state.status = "no emitter here".into();
            }
        }
        EntityKind::Trigger => {
            if let Some(idx) = trigger_at(state, x, y, z) {
                state.level.triggers.remove(idx);
                state
                    .level
                    .hier_fixup_after_remove(re_level::HierKind::Trigger, idx);
                // Keep the selection pointing at the same logical trigger (or drop
                // it) after the Vec shifts — never edit a shifted neighbour.
                state.selected_trigger = match state.selected_trigger {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed trigger #{idx}");
            } else {
                state.status = "no trigger here".into();
            }
        }
        EntityKind::Pipe => {
            // Delete the selected pipe, or the pipe whose path covers the cursor.
            // The caller (`x` key / tree delete) already snapshotted undo, so remove
            // directly rather than via `pipe::delete_selected` (which re-snapshots).
            let target = state
                .selected_pipe
                .filter(|&i| i < state.level.pipes.len())
                .or_else(|| crate::pipe::pipe_at(state, x, y, z));
            if let Some(idx) = target {
                let removed = state.level.pipes.remove(idx);
                state.selected_pipe = match state.selected_pipe {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed pipe '{}'->'{}'", removed.from, removed.to);
            } else {
                state.status = "no pipe here".into();
            }
        }
        EntityKind::ZoneInput => {
            if let Some(idx) = zone_input_at(state, x, y, z) {
                state.level.zone_inputs.remove(idx);
                // ZoneInput is not part of the object-nesting hierarchy (no
                // `HierKind::ZoneInput`), so no hierarchy fixup — just keep the
                // selection pointing at the same logical zone (or drop it).
                state.selected_zone_input = match state.selected_zone_input {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed zone input #{idx}");
            } else {
                state.status = "no zone input here".into();
            }
        }
        EntityKind::Note => {
            if let Some(idx) = note_at(state, x, y, z) {
                state.level.notes.remove(idx);
                // Note is not part of the hierarchy either — no fixup needed.
                state.selected_note = match state.selected_note {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed note #{idx}");
            } else {
                state.status = "no note here".into();
            }
        }
        EntityKind::WallDecal => {
            if let Some(idx) = decal_at(state, x, y, z) {
                state.level.decals.remove(idx);
                // WallDecal is not part of the object-nesting hierarchy (no
                // `HierKind::WallDecal`), so no hierarchy fixup — just keep the
                // selection pointing at the same logical decal (or drop it).
                state.selected_decal = match state.selected_decal {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed wall decal #{idx}");
            } else {
                state.status = "no wall decal here".into();
            }
        }
        EntityKind::Holo => {
            if let Some(idx) = holo_at(state, x, y, z) {
                state.level.holos.remove(idx);
                // Holo is not part of the object-nesting hierarchy (no
                // `HierKind::Holo`), so no hierarchy fixup — just keep the selection
                // pointing at the same logical hologram (or drop it).
                state.selected_holo = match state.selected_holo {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed hologram #{idx}");
            } else {
                state.status = "no hologram here".into();
            }
        }
        EntityKind::Portal => {
            if let Some(idx) = portal_at(state, x, y, z) {
                state.level.portals.remove(idx);
                // Portal is not part of the object-nesting hierarchy (no
                // `HierKind::Portal`), so no hierarchy fixup — just keep the selection
                // pointing at the same logical portal (or drop it).
                state.selected_portal = match state.selected_portal {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed portal #{idx}");
            } else {
                state.status = "no portal here".into();
            }
        }
        EntityKind::Zone => {
            if let Some(idx) = zone_overlay_at(state, x, y, z) {
                state.level.overlay.zones.remove(idx);
                // Overlay zones are not part of the object-nesting hierarchy — just
                // keep the selection pointing at the same logical zone (or drop it).
                state.selected_zone = match state.selected_zone {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed zone #{idx}");
            } else {
                state.status = "no zone here".into();
            }
        }
        EntityKind::Line => {
            if let Some(idx) = line_at(state, x, y, z) {
                state.level.overlay.lines.remove(idx);
                state.selected_line = match state.selected_line {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed line #{idx}");
            } else {
                state.status = "no line here".into();
            }
        }
        EntityKind::Inset => {
            if let Some(idx) = inset_at(state, x, y, z) {
                let removed = state.level.overlay.insets[idx].name.clone();
                state.level.overlay.insets.remove(idx);
                // REFERENTIAL INTEGRITY: a zone may name this inset; drop the now-dangling
                // reference so a reload doesn't carry an inset name with no inset.
                for zn in &mut state.level.overlay.zones {
                    zn.insets.retain(|i| i.0 != removed);
                }
                state.selected_inset = match state.selected_inset {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                state.mark_dirty();
                state.status = format!("removed inset '{removed}'");
            } else {
                state.status = "no inset here".into();
            }
        }
        EntityKind::Field => {
            if let Some(idx) = crate::field::field_at(state, x, y, z) {
                let name = state.level.field_defs[idx].name.clone();
                state.level.field_defs.remove(idx);
                // Field is not part of the object-nesting hierarchy (no
                // `HierKind::Field`), so no hierarchy fixup — just keep the
                // selection pointing at the same logical field (or drop it).
                state.selected_field = match state.selected_field {
                    Some(s) if s == idx => None,
                    Some(s) if s > idx => Some(s - 1),
                    other => other,
                };
                crate::field::re_resolve(state);
                state.mark_dirty();
                state.status = format!("removed field '{name}'");
            } else {
                state.status = "no field here".into();
            }
        }
    }
}

/// Snap a stale focused field to the active kind's FIRST field. The interactive
/// `o`-selector path resets focus in [`cycle_selector`], but a caller (or test)
/// that sets `state.place` directly may leave `entity_field` pointing at another
/// kind's field; this keeps `cycle_property` / `adjust_property` well-defined in
/// that case (Emitter has its own focus list and is left alone).
fn ensure_entity_focus(state: &mut EditorState) {
    let Some(kind) = state.place else { return };
    if kind == EntityKind::Emitter {
        return;
    }
    let list = crate::state::entity_fields_for(kind);
    if !list.contains(&state.entity_field) {
        if let Some(&first) = list.first() {
            state.entity_field = first;
        }
    }
}

/// Cycle the FOCUSED property of the active entity kind (`c`). Routed by the
/// focused [`EntityField`] so it is exhaustive:
/// * Door → toggle the cursor node in `target_parts` (on) / `target_off` (off),
/// * Node → glyph candidate, io role (neither → input → output), pool/pool_alt
///   word add (a `c` on a count field appends a word; `+`/`-` add/remove),
/// * Spawn → the cardinal facing,
/// * Pedestal → no enum fields (numeric only), reported as such,
/// * Emitter → the focused emitter enum field (existing behaviour).
pub fn cycle_property(state: &mut EditorState) {
    ensure_entity_focus(state);
    state.schedule_preview_rebuild(); // debounced preview catch-up
    match state.place {
        Some(EntityKind::Emitter) => cycle_emitter_enum(state),
        Some(EntityKind::Door) => cycle_door_enum(state),
        Some(EntityKind::Node) => cycle_node_enum(state),
        Some(EntityKind::Trigger) => cycle_trigger_enum(state),
        Some(EntityKind::Pipe) => crate::pipe::cycle_enum(state),
        Some(EntityKind::ZoneInput) => cycle_zone_enum(state),
        Some(EntityKind::Note) => cycle_note_enum(state),
        Some(EntityKind::WallDecal) => cycle_decal_enum(state),
        Some(EntityKind::Holo) => cycle_holo_enum(state),
        Some(EntityKind::Portal) => cycle_portal_enum(state),
        Some(EntityKind::Zone) => cycle_zone_overlay_enum(state),
        Some(EntityKind::Line) => cycle_line_enum(state),
        Some(EntityKind::Inset) => {
            state.status = "inset: use +/- (child is typed)".into();
        }
        Some(EntityKind::Spawn) => {
            let next = cycle_facing(state, state.level.player.facing_yaw_col);
            state.level.player.facing_yaw_col = next;
            state.mark_dirty();
            state.status = format!("spawn facing {}", crate::state::facing_label(next));
        }
        Some(EntityKind::Pedestal) => {
            state.status = "pedestal: use +/- (no enum field)".into();
        }
        Some(EntityKind::Part) => {
            state.status = "part: use +/- (q edits the mask)".into();
        }
        Some(EntityKind::Field) => crate::field::cycle_field_enum(state),
        None => {
            state.status = "no cyclable property".into();
        }
    }
}

/// `c` on a Door focused field: TOGGLE the target-cursor node's membership in
/// `target_parts` (when focused on `DoorTargetOn`) or `target_off` (when focused
/// on `DoorTargetOff`). A position field is numeric — hint to use `+`/`-`. With no
/// nodes in the level there is nothing to target (defined no-op).
fn cycle_door_enum(state: &mut EditorState) {
    let field = state.entity_field;
    if !matches!(
        field,
        EntityField::DoorTargetOn | EntityField::DoorTargetOff
    ) {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    let n = state.level.nodes.len();
    if n == 0 {
        state.status = "no nodes to target (place a Node first)".into();
        return;
    }
    let idx = state.door_target_cursor.min(n - 1);
    let name = state.level.nodes[idx].name.clone();
    let on_focus = field == EntityField::DoorTargetOn;
    let label = if on_focus { "on" } else { "off" };
    let door = &mut state.level.door;
    // Toggle membership in the focused list. The two lists are disjoint by intent
    // (a node is either a must-FIRE or a must-be-OFF target), so ADDING to one
    // drops it from the other to keep the win condition coherent.
    let (list, other): (&mut Vec<usize>, &mut Vec<usize>) = if on_focus {
        (&mut door.target_parts, &mut door.target_off)
    } else {
        (&mut door.target_off, &mut door.target_parts)
    };
    let was_present = if let Some(pos) = list.iter().position(|&i| i == idx) {
        list.remove(pos);
        true
    } else {
        list.push(idx);
        list.sort_unstable();
        list.dedup();
        // Remove from the other list so it is not a contradictory target.
        if let Some(pos) = other.iter().position(|&i| i == idx) {
            other.remove(pos);
        }
        false
    };
    let verb = if was_present { "removed" } else { "added" };
    state.mark_dirty();
    state.status = format!("door target({label}): {verb} '{name}'");
}

/// `c` on a Node focused field: cycle the glyph candidate or the io role. Other
/// node fields are numeric/selection (`+`/`-`). A no-op (with a hint) when none is
/// selected.
fn cycle_node_enum(state: &mut EditorState) {
    let field = state.entity_field;
    // NodeEmits toggles a `[emit]` clause; handle it before borrowing the node
    // mutably (it touches both `nodes` and `parts`/`emits`).
    if field == EntityField::NodeEmits {
        toggle_node_emit(state);
        return;
    }
    let Some(idx) = state.selected_node.filter(|&i| i < state.level.nodes.len()) else {
        state.status = "no node selected".into();
        return;
    };
    // The glyph cycle REMEMBERS the value it steps off (`state.vocab`), so it
    // needs `&mut state` — resolved before the node itself is borrowed mutably.
    if field == EntityField::NodeGlyph {
        let name = state.level.nodes[idx].name.clone();
        let cur = state.level.nodes[idx].glyph;
        let g = step_node_glyph(state, cur, 1);
        state.level.nodes[idx].glyph = g;
        state.mark_dirty();
        state.status = format!("node '{name}' glyph {g}");
        return;
    }
    let n = &mut state.level.nodes[idx];
    let name = n.name.clone();
    match field {
        EntityField::NodeIo => {
            // neither → input → output → neither
            let (is_input, is_output) = match (n.is_input, n.is_output) {
                (false, false) => (true, false),
                (true, false) => (false, true),
                _ => (false, false),
            };
            // B2: a `[zone_input]` links an input node BY NAME and `re_level` finalize
            // rejects the reload if that node is no longer `is_input` ("[zone_input X]
            // node 'Y' is not an input node"). Cycling THIS node off `is_input` while a
            // zone depends on it would silently author an unreloadable level. REFUSE the
            // cycle (least destructive — keep the node an input and tell the user) so no
            // path can leave a zone_input pointing at a non-input node. Read the role
            // flag out before borrowing `level.zone_inputs` (drops the `n` borrow).
            let leaving_input = n.is_input && !is_input;
            if leaving_input && state.level.zone_inputs.iter().any(|z| z.node == name) {
                state.status =
                    format!("can't change IO: a zone_input depends on input node '{name}'");
                return;
            }
            let n = &mut state.level.nodes[idx];
            n.is_input = is_input;
            n.is_output = is_output;
            let role = io_role_label(is_input, is_output);
            state.mark_dirty();
            state.status = format!("node '{name}' io: {role}");
        }
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
        }
    }
}

/// `c` on the Node `NodeEmits` field: TOGGLE a `[emit]` clause linking the
/// SELECTED node to the `emit_part_cursor`'s part — add a [`re_level::EmitClause`]
/// when absent, remove it when present (the node lights that part when it fires).
/// A defined no-op (with a hint) when no node is selected or the level has no
/// parts. Mirrors [`cycle_door_enum`]'s toggle.
fn toggle_node_emit(state: &mut EditorState) {
    let Some(node) = state.selected_node.filter(|&i| i < state.level.nodes.len()) else {
        state.status = "no node selected".into();
        return;
    };
    let n_parts = state.level.parts.len();
    if n_parts == 0 {
        state.status = "no parts to light (place a Part first)".into();
        return;
    }
    let part = state.emit_part_cursor.min(n_parts - 1);
    state.emit_part_cursor = part;
    let part_name = state.level.parts[part].name.clone();
    let node_name = state.level.nodes[node].name.clone();
    let pos = state
        .level
        .emits
        .iter()
        .position(|e| e.source_node == node && e.part == part);
    let verb = match pos {
        Some(p) => {
            state.level.emits.remove(p);
            "removed"
        }
        None => {
            state.level.emits.push(re_level::EmitClause {
                source_node: node,
                part,
            });
            "added"
        }
    };
    state.mark_dirty();
    state.status = format!("emit: {verb} '{node_name}' -> '{part_name}'");
}

/// Page the FOCUSED non-emitter entity field by `delta` (±1) through the active
/// kind's field list (the `;`/`'` keys), wrapping. Reports the new focus + value.
/// A no-op (with a hint) when the object selector is off or on Emitter (which has
/// its own `cycle_emitter_field`).
pub fn cycle_entity_field(state: &mut EditorState, delta: i32) {
    let Some(kind) = state.place else {
        state.status = "select an object (o) first".into();
        return;
    };
    if kind == EntityKind::Emitter {
        cycle_emitter_field(state, delta);
        return;
    }
    state.entity_field = step_entity_field(kind, state.entity_field, delta);
    let f = state.entity_field;
    // Hint which key edits this field: a pure-enum field wants `c`, the door
    // target toggles want both, everything else wants `+`/`-`.
    let key = if f.is_enum_cyclable() {
        if f.is_numeric() {
            "c/+-"
        } else {
            "c"
        }
    } else {
        "+/-"
    };
    state.status = format!(
        "focus: {} = {} ({key})",
        f.label(),
        entity_field_value(state)
    );
}

/// Render the focused non-emitter field's current value as a compact string for
/// the panel / status line. Mirrors `emitter_field_value`.
pub fn entity_field_value(state: &EditorState) -> String {
    entity_field_value_of(state, state.entity_field)
}

/// Render an ARBITRARY non-emitter entity field's current value (the
/// inspector row model lists EVERY field of the selected object, not just the
/// focused one, so the value renderer must be parameterized by field).
pub fn entity_field_value_of(state: &EditorState, field: EntityField) -> String {
    // The FOCUSED text field shows its live edit buffer with the caret
    // marker (in-progress whitespace included), not the committed model.
    if let Some(readout) = text_field_readout(state, field) {
        return readout;
    }
    let lvl = &state.level;
    match field {
        EntityField::DoorPosX => lvl.door.x.to_string(),
        EntityField::DoorPosY => lvl.door.y.to_string(),
        EntityField::DoorPosZ => lvl.door.z.to_string(),
        EntityField::DoorTargetOn | EntityField::DoorTargetOff => door_target_value(state),
        EntityField::NodePosX => node_field_str(state, |n| n.x.to_string()),
        EntityField::NodePosY => node_field_str(state, |n| n.y.to_string()),
        EntityField::NodePosZ => node_field_str(state, |n| n.z.to_string()),
        EntityField::NodeGlyph => node_field_str(state, |n| n.glyph.to_string()),
        EntityField::NodeColor => {
            node_field_str(state, |n| format!("#{:06X}", n.color_rgb & 0xFF_FFFF))
        }
        EntityField::NodeIo => node_field_str(state, |n| {
            io_role_label(n.is_input, n.is_output).to_string()
        }),
        EntityField::NodeThreshold => node_field_str(state, |n| n.threshold.to_string()),
        EntityField::NodeAnchor => node_field_str(state, |n| n.anchor.to_string()),
        EntityField::NodePool => node_field_str(state, |n| {
            if n.pool.is_empty() {
                "(empty)".to_string()
            } else {
                format!("{} [{}]", n.pool.len(), n.pool.join(" "))
            }
        }),
        EntityField::NodePoolAlt => node_field_str(state, |n| {
            if n.pool_alt.is_empty() {
                "(empty)".to_string()
            } else {
                format!("{} [{}]", n.pool_alt.len(), n.pool_alt.join(" "))
            }
        }),
        EntityField::NodeEmits => node_emits_value(state),
        EntityField::PedCenterX => lvl.pedestal.cx.to_string(),
        EntityField::PedCenterY => lvl.pedestal.cy.to_string(),
        EntityField::PedCenterZ => lvl.pedestal.cz.to_string(),
        EntityField::PedVolW => lvl.pedestal.w.max(1).to_string(),
        EntityField::PedVolH => lvl.pedestal.h.max(1).to_string(),
        EntityField::PedVolD => lvl.pedestal.d.max(1).to_string(),
        EntityField::PartSizeW => part_field_str(state, |p| p.w.to_string()),
        EntityField::PartSizeH => part_field_str(state, |p| p.h.to_string()),
        EntityField::PartSizeD => part_field_str(state, |p| p.d.to_string()),
        EntityField::PartOffX => part_field_str(state, |p| p.off_x.to_string()),
        EntityField::PartOffY => part_field_str(state, |p| p.off_y.to_string()),
        EntityField::PartOffZ => part_field_str(state, |p| p.off_z.to_string()),
        EntityField::PartColor => {
            part_field_str(state, |p| format!("#{:06X}", p.color_rgb & 0xFF_FFFF))
        }
        EntityField::SpawnPosX => lvl.player.x.to_string(),
        EntityField::SpawnPosY => lvl.player.y.to_string(),
        EntityField::SpawnPosZ => lvl.player.z.to_string(),
        EntityField::SpawnFacing => crate::state::facing_label(lvl.player.facing_yaw_col),
        EntityField::TrigMinX => trigger_field_str(state, |b| b.0 .0.to_string()),
        EntityField::TrigMinY => trigger_field_str(state, |b| b.0 .1.to_string()),
        EntityField::TrigMinZ => trigger_field_str(state, |b| b.0 .2.to_string()),
        EntityField::TrigMaxX => trigger_field_str(state, |b| b.1 .0.to_string()),
        EntityField::TrigMaxY => trigger_field_str(state, |b| b.1 .1.to_string()),
        EntityField::TrigMaxZ => trigger_field_str(state, |b| b.1 .2.to_string()),
        EntityField::TrigActionType => trigger_value(state, |_state, t| match t.action {
            re_core::TriggerAction::ActivateNode { .. } => "activate".to_string(),
            re_core::TriggerAction::Event { .. } => "event".to_string(),
            _ => "?".to_string(),
        }),
        EntityField::TrigTarget => trigger_value(state, |state, t| match &t.action {
            re_core::TriggerAction::ActivateNode { name, node_index } => state
                .level
                .nodes
                .get(*node_index)
                .map(|n| n.name.clone())
                .unwrap_or_else(|| name.clone()),
            re_core::TriggerAction::Event { label } => label.clone(),
            _ => "?".to_string(),
        }),
        EntityField::TrigOnce => trigger_value(state, |_state, t| t.once.to_string()),
        EntityField::PipeMode
        | EntityField::PipeState
        | EntityField::PipeSwitch
        | EntityField::PipeDensity
        | EntityField::PipeColor => crate::pipe::field_value_of(state, field),
        EntityField::ZoneMinX => zone_field_str(state, |b| b.0 .0.to_string()),
        EntityField::ZoneMinY => zone_field_str(state, |b| b.0 .1.to_string()),
        EntityField::ZoneMinZ => zone_field_str(state, |b| b.0 .2.to_string()),
        EntityField::ZoneMaxX => zone_field_str(state, |b| b.1 .0.to_string()),
        EntityField::ZoneMaxY => zone_field_str(state, |b| b.1 .1.to_string()),
        EntityField::ZoneMaxZ => zone_field_str(state, |b| b.1 .2.to_string()),
        EntityField::ZoneNode => zone_node_value(state),
        EntityField::NoteKindField => {
            note_field_str(state, |n| crate::state::note_kind_label(n.kind).to_string())
        }
        EntityField::NoteFacing => note_field_str(state, |n| crate::state::facing_label(n.facing)),
        EntityField::NotePosX => note_field_str(state, |n| n.x.to_string()),
        EntityField::NotePosY => note_field_str(state, |n| n.y.to_string()),
        EntityField::NotePosZ => note_field_str(state, |n| n.z.to_string()),
        EntityField::NoteText => note_field_str(state, |n| {
            if n.text.is_empty() {
                "(empty)".to_string()
            } else {
                // Multi-line bodies flatten to one row with a visible ⏎ marker
                // (the focused readout above shows the caret's line instead).
                n.text.replace('\n', "\u{23CE}")
            }
        }),
        EntityField::DecalSurface => {
            decal_value(state, |d| crate::state::decal_surface_label(d.surface_kind))
        }
        EntityField::DecalMinX => decal_field_str(state, |b| b.0 .0.to_string()),
        EntityField::DecalMinY => decal_field_str(state, |b| b.0 .1.to_string()),
        EntityField::DecalMinZ => decal_field_str(state, |b| b.0 .2.to_string()),
        EntityField::DecalMaxX => decal_field_str(state, |b| b.1 .0.to_string()),
        EntityField::DecalMaxY => decal_field_str(state, |b| b.1 .1.to_string()),
        EntityField::DecalMaxZ => decal_field_str(state, |b| b.1 .2.to_string()),
        EntityField::DecalDensity => decal_value(state, |d| format!("{:.2}", d.density)),
        EntityField::DecalColorHead => {
            decal_value(state, |d| format!("#{:06X}", d.color_head & 0xFF_FFFF))
        }
        EntityField::DecalColorTail => {
            decal_value(state, |d| format!("#{:06X}", d.color_tail & 0xFF_FFFF))
        }
        EntityField::HoloShapeField => {
            holo_value(state, |h| crate::state::holo_shape_label(h.shape))
        }
        EntityField::HoloMinX => holo_field_str(state, |b| b.0 .0.to_string()),
        EntityField::HoloMinY => holo_field_str(state, |b| b.0 .1.to_string()),
        EntityField::HoloMinZ => holo_field_str(state, |b| b.0 .2.to_string()),
        EntityField::HoloMaxX => holo_field_str(state, |b| b.1 .0.to_string()),
        EntityField::HoloMaxY => holo_field_str(state, |b| b.1 .1.to_string()),
        EntityField::HoloMaxZ => holo_field_str(state, |b| b.1 .2.to_string()),
        EntityField::HoloStyle => holo_value(state, |h| h.style.to_string()),
        EntityField::PortalAx0 => portal_field_str(state, |p| p.ax0.to_string()),
        EntityField::PortalAy0 => portal_field_str(state, |p| p.ay0.to_string()),
        EntityField::PortalAz0 => portal_field_str(state, |p| p.az0.to_string()),
        EntityField::PortalAx1 => portal_field_str(state, |p| p.ax1.to_string()),
        EntityField::PortalAy1 => portal_field_str(state, |p| p.ay1.to_string()),
        EntityField::PortalAz1 => portal_field_str(state, |p| p.az1.to_string()),
        EntityField::PortalBx0 => portal_field_str(state, |p| p.bx0.to_string()),
        EntityField::PortalBy0 => portal_field_str(state, |p| p.by0.to_string()),
        EntityField::PortalBz0 => portal_field_str(state, |p| p.bz0.to_string()),
        EntityField::PortalBx1 => portal_field_str(state, |p| p.bx1.to_string()),
        EntityField::PortalBy1 => portal_field_str(state, |p| p.by1.to_string()),
        EntityField::PortalBz1 => portal_field_str(state, |p| p.bz1.to_string()),
        EntityField::PortalRotate => portal_field_str(state, |p| format!("{}", p.rotate)),
        EntityField::PortalSkin => portal_field_str(state, |p| {
            if p.skin.is_empty() {
                "(none)".to_string()
            } else {
                p.skin.clone()
            }
        }),
        EntityField::ZoneRegMinX => zone_overlay_field_str(state, |b| b.0 .0.to_string()),
        EntityField::ZoneRegMinY => zone_overlay_field_str(state, |b| b.0 .1.to_string()),
        EntityField::ZoneRegMinZ => zone_overlay_field_str(state, |b| b.0 .2.to_string()),
        EntityField::ZoneRegMaxX => zone_overlay_field_str(state, |b| b.1 .0.to_string()),
        EntityField::ZoneRegMaxY => zone_overlay_field_str(state, |b| b.1 .1.to_string()),
        EntityField::ZoneRegMaxZ => zone_overlay_field_str(state, |b| b.1 .2.to_string()),
        EntityField::ZoneMaterial => zone_overlay_value(state, |z| {
            if z.material.0.is_empty() {
                "(none)".to_string()
            } else {
                z.material.0.clone()
            }
        }),
        EntityField::ZoneScale => zone_overlay_value(state, |z| {
            crate::state::scale_rule_label(&z.scale).to_string()
        }),
        EntityField::ZoneArms => zone_overlay_value(state, |z| match &z.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => p.arms.to_string(),
            _ => "-".to_string(),
        }),
        EntityField::ZoneWarp => zone_overlay_value(state, |z| match &z.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => format!("{:.2}", p.octave_warp),
            _ => "-".to_string(),
        }),
        EntityField::ZoneTwist => zone_overlay_value(state, |z| match &z.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => format!("{:.2}", p.twist),
            _ => "-".to_string(),
        }),
        EntityField::ZoneInsets => zone_insets_value(state),
        EntityField::LineMaterial => line_value(state, |l| {
            if l.material.0.is_empty() {
                "(none)".to_string()
            } else {
                l.material.0.clone()
            }
        }),
        EntityField::LineScale => line_value(state, |l| {
            crate::state::scale_rule_label(&l.scale).to_string()
        }),
        EntityField::LineArms => line_value(state, |l| match &l.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => p.arms.to_string(),
            _ => "-".to_string(),
        }),
        EntityField::LineWarp => line_value(state, |l| match &l.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => format!("{:.2}", p.octave_warp),
            _ => "-".to_string(),
        }),
        EntityField::LineTwist => line_value(state, |l| match &l.scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => format!("{:.2}", p.twist),
            _ => "-".to_string(),
        }),
        EntityField::LinePoints => line_value(state, |l| l.path.points.len().to_string()),
        EntityField::InsetOctaveLo => inset_value(state, |i| i.at_octave.0.to_string()),
        EntityField::InsetOctaveHi => inset_value(state, |i| i.at_octave.1.to_string()),
        EntityField::InsetRegMinX => inset_field_str(state, |b| b.0 .0.to_string()),
        EntityField::InsetRegMinY => inset_field_str(state, |b| b.0 .1.to_string()),
        EntityField::InsetRegMinZ => inset_field_str(state, |b| b.0 .2.to_string()),
        EntityField::InsetRegMaxX => inset_field_str(state, |b| b.1 .0.to_string()),
        EntityField::InsetRegMaxY => inset_field_str(state, |b| b.1 .1.to_string()),
        EntityField::InsetRegMaxZ => inset_field_str(state, |b| b.1 .2.to_string()),
        EntityField::InsetChild => inset_value(state, |i| {
            let re_level::OverlayInsetContent::ChildScene(s) = &i.content;
            if s.is_empty() {
                "(empty)".to_string()
            } else {
                s.clone()
            }
        }),
        EntityField::FieldArchetype
        | EntityField::FieldShapeKind
        | EntityField::FieldPosX
        | EntityField::FieldPosY
        | EntityField::FieldPosZ
        | EntityField::FieldRadius
        | EntityField::FieldInner
        | EntityField::FieldFalloff
        | EntityField::FieldOcclude
        | EntityField::FieldFacing
        | EntityField::FieldCone
        | EntityField::FieldBind
        | EntityField::FieldInvert
        | EntityField::FieldDrive
        | EntityField::FieldBase
        | EntityField::FieldDepth
        | EntityField::FieldPriority => crate::field::value_of(state, field),
    }
}

/// Helper: render the SELECTED zone input's box corners via `f`, or `"-"` when
/// none is selected.
fn zone_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state
        .selected_zone_input
        .filter(|&i| i < state.level.zone_inputs.len())
    {
        Some(i) => f(zone_box(&state.level.zone_inputs[i])),
        None => "-".to_string(),
    }
}

/// The selected zone's linked-input readout: the cursor input-node's name and its
/// position within the input-node list. `"-"` when none selected; a hint when the
/// level has no input nodes.
fn zone_node_value(state: &EditorState) -> String {
    if state
        .selected_zone_input
        .is_none_or(|i| i >= state.level.zone_inputs.len())
    {
        return "-".to_string();
    }
    let inputs = input_node_indices(state);
    if inputs.is_empty() {
        return "no input nodes".to_string();
    }
    let pos = inputs
        .iter()
        .position(|&i| i == state.zone_node_cursor)
        .unwrap_or(0);
    let idx = inputs[pos];
    format!(
        "[{}/{}] '{}'",
        pos + 1,
        inputs.len(),
        state.level.nodes[idx].name
    )
}

/// Helper: render a field of the SELECTED note via `f`, or `"-"` when none.
fn note_field_str(state: &EditorState, f: impl Fn(&re_level::LevelNote) -> String) -> String {
    match state.selected_note.filter(|&i| i < state.level.notes.len()) {
        Some(i) => f(&state.level.notes[i]),
        None => "-".to_string(),
    }
}

/// `c` on a ZoneInput focused field: cycle the linked INPUT node (the `ZoneNode`
/// field). Region corners are numeric (`+`/`-`) and hinted as such. A no-op (with a
/// hint) when no zone is selected or the level has no input nodes.
fn cycle_zone_enum(state: &mut EditorState) {
    let field = state.entity_field;
    if field != EntityField::ZoneNode {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    let Some(idx) = state
        .selected_zone_input
        .filter(|&i| i < state.level.zone_inputs.len())
    else {
        state.status = "no zone input selected".into();
        return;
    };
    cycle_zone_node(state, idx, 1);
}

/// Step the selected zone's linked input node by `delta` through the INPUT nodes
/// (wrapping), keeping `node` (name) + `node_index` + the cursor in sync. A defined
/// no-op when the level has no input nodes.
fn cycle_zone_node(state: &mut EditorState, idx: usize, delta: i32) {
    let inputs = input_node_indices(state);
    if inputs.is_empty() {
        state.status = "no input nodes (place an input Node first)".into();
        return;
    }
    let cur = inputs
        .iter()
        .position(|&i| i == state.zone_node_cursor)
        .unwrap_or(0) as i32;
    let next = (cur + delta).rem_euclid(inputs.len() as i32) as usize;
    let node_index = inputs[next];
    state.zone_node_cursor = node_index;
    let name = state.level.nodes[node_index].name.clone();
    let zi = &mut state.level.zone_inputs[idx];
    zi.node = name.clone();
    zi.node_index = node_index;
    state.mark_dirty();
    state.status = format!("zone #{idx} -> input '{name}'");
}

/// `c` on a WallDecal focused field: cycle the `surface` texture through every kind
/// the `.lvl` format accepts (trace / matrix / solid / hologram / portal / object —
/// see [`crate::state::DECAL_SURFACES`]), so an imported decal keeps its surface
/// instead of being rewritten to `trace`. Region corners + density + colours are
/// numeric (`+`/`-`) and hinted as such. A no-op (with a hint) when no decal is
/// selected.
fn cycle_decal_enum(state: &mut EditorState) {
    let field = state.entity_field;
    if field != EntityField::DecalSurface {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    let Some(idx) = state
        .selected_decal
        .filter(|&i| i < state.level.decals.len())
    else {
        state.status = "no wall decal selected".into();
        return;
    };
    let d = &mut state.level.decals[idx];
    d.surface_kind = crate::state::cycle_decal_surface(d.surface_kind, 1);
    let label = crate::state::decal_surface_label(d.surface_kind);
    state.mark_dirty();
    state.status = format!("wall decal #{idx} surface: {label}");
}

/// `c` on a Holo focused field: cycle the `shape` interior fill (box / pyramid /
/// sphere). Region corners + style are numeric (`+`/`-`) and hinted as such. A
/// no-op (with a hint) when no hologram is selected.
fn cycle_holo_enum(state: &mut EditorState) {
    let field = state.entity_field;
    if field != EntityField::HoloShapeField {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    let Some(idx) = state.selected_holo.filter(|&i| i < state.level.holos.len()) else {
        state.status = "no hologram selected".into();
        return;
    };
    let h = &mut state.level.holos[idx];
    h.shape = crate::state::cycle_holo_shape(h.shape, 1);
    let label = crate::state::holo_shape_label(h.shape);
    state.mark_dirty();
    state.status = format!("hologram #{idx} shape: {label}");
}

/// `c` on a Portal focused field: cycle the `skin` material name (`(none)` + the
/// level's material names, wrapping). The 12 region corners + `rotate` are numeric
/// (`+`/`-`) and hinted as such. A no-op (with a hint) when no portal is selected.
fn cycle_portal_enum(state: &mut EditorState) {
    let field = state.entity_field;
    if field != EntityField::PortalSkin {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    let Some(idx) = state
        .selected_portal
        .filter(|&i| i < state.level.portals.len())
    else {
        state.status = "no portal selected".into();
        return;
    };
    // Step the skin vocabulary (`(none)` + material names + off-list names this
    // session has seen). An empty skin reads as `(none)`.
    let current = state.level.portals[idx].skin.clone();
    let next = step_material_name(state, &current, 1);
    // `(none)` ⇒ the empty string (no skin); any material name is stored verbatim.
    let new_skin = if next == "(none)" {
        String::new()
    } else {
        next.clone()
    };
    let shown = next;
    state.level.portals[idx].skin = new_skin;
    state.mark_dirty();
    state.status = format!("portal #{idx} skin: {shown}");
}

/// `c` on a Zone focused field: cycle the bound MATERIAL name (`(none)` + the
/// level's material names), the SCALE rule (None → Crisp → SelfSimilar), or TOGGLE
/// the inset-cursor's name in the zone's `insets` list. Region corners + the
/// SelfSimilar sub-fields are numeric (`+`/`-`). A no-op (with a hint) when no zone
/// is selected.
fn cycle_zone_overlay_enum(state: &mut EditorState) {
    let field = state.entity_field;
    let Some(idx) = state
        .selected_zone
        .filter(|&i| i < state.level.overlay.zones.len())
    else {
        state.status = "no zone selected".into();
        return;
    };
    match field {
        EntityField::ZoneMaterial => {
            let current = state.level.overlay.zones[idx].material.0.clone();
            let next = step_material_name(state, &current, 1);
            let new_mat = if next == "(none)" {
                String::new()
            } else {
                next.clone()
            };
            let shown = next;
            state.level.overlay.zones[idx].material = MaterialId(new_mat);
            state.mark_dirty();
            state.status = format!("zone #{idx} material: {shown}");
        }
        EntityField::ZoneScale => {
            let z = &mut state.level.overlay.zones[idx];
            let params = scale_params(&z.scale);
            z.scale = crate::state::cycle_scale_rule(&z.scale, params);
            let label = crate::state::scale_rule_label(&z.scale);
            state.mark_dirty();
            state.status = format!("zone #{idx} scale: {label}");
        }
        EntityField::ZoneInsets => toggle_zone_inset(state, idx),
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
        }
    }
}

/// `c` on the Zone `ZoneInsets` field: TOGGLE the `zone_inset_cursor`'s inset name
/// in the selected zone's `insets` Vec — add an [`InsetId`] when absent, remove it
/// when present. A defined no-op (with a hint) when the level has no insets.
/// Mirrors [`cycle_door_enum`]'s membership toggle.
fn toggle_zone_inset(state: &mut EditorState, zone: usize) {
    let n_insets = state.level.overlay.insets.len();
    if n_insets == 0 {
        state.status = "no insets to reference (place an Inset first)".into();
        return;
    }
    let cur = state.zone_inset_cursor.min(n_insets - 1);
    state.zone_inset_cursor = cur;
    let inset_name = state.level.overlay.insets[cur].name.clone();
    let zn = &mut state.level.overlay.zones[zone];
    let verb = if let Some(pos) = zn.insets.iter().position(|i| i.0 == inset_name) {
        zn.insets.remove(pos);
        "removed"
    } else {
        zn.insets.push(InsetId(inset_name.clone()));
        "added"
    };
    state.mark_dirty();
    state.status = format!("zone #{zone} insets: {verb} '{inset_name}'");
}

/// `c` on a Line focused field: cycle the bound line-MATERIAL name (`(none)` + the
/// level's material names) or the SCALE rule. The SelfSimilar sub-fields + waypoint
/// count are numeric (`+`/`-`). A no-op (with a hint) when no line is selected.
fn cycle_line_enum(state: &mut EditorState) {
    let field = state.entity_field;
    let Some(idx) = state
        .selected_line
        .filter(|&i| i < state.level.overlay.lines.len())
    else {
        state.status = "no line selected".into();
        return;
    };
    match field {
        EntityField::LineMaterial => {
            let current = state.level.overlay.lines[idx].material.0.clone();
            let next = step_material_name(state, &current, 1);
            let new_mat = if next == "(none)" {
                String::new()
            } else {
                next.clone()
            };
            let shown = next;
            state.level.overlay.lines[idx].material = LineMaterialId(new_mat);
            state.mark_dirty();
            state.status = format!("line #{idx} material: {shown}");
        }
        EntityField::LineScale => {
            let l = &mut state.level.overlay.lines[idx];
            let params = scale_params(&l.scale);
            l.scale = crate::state::cycle_scale_rule(&l.scale, params);
            let label = crate::state::scale_rule_label(&l.scale);
            state.mark_dirty();
            state.status = format!("line #{idx} scale: {label}");
        }
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
        }
    }
}

/// `+`/`-` on a Zone focused field: nudge a region corner (clamped to the room AND
/// kept `min <= max` per axis), step a SelfSimilar sub-field (arms / octave_warp /
/// twist — only meaningful under SelfSimilar, else a no-op hint), or move the
/// inset-membership cursor (the `c` key then toggles). `material`/`scale` are enum
/// fields (`c`). A no-op (with a hint) when no zone is selected.
fn adjust_zone_overlay_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if matches!(field, EntityField::ZoneMaterial | EntityField::ZoneScale) {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_zone
        .filter(|&i| i < state.level.overlay.zones.len())
    else {
        state.status = "no zone selected".into();
        return;
    };
    // Inset-membership cursor: `+`/`-` move it through the inset list (`c` toggles).
    if field == EntityField::ZoneInsets {
        let n = state.level.overlay.insets.len();
        if n == 0 {
            state.status = "no insets to reference".into();
            return;
        }
        let cur = state.zone_inset_cursor.min(n - 1) as i32;
        state.zone_inset_cursor = (cur + delta).rem_euclid(n as i32) as usize;
        state.status = format!("zone insets {}", zone_insets_value(state));
        return;
    }
    // SelfSimilar sub-fields edit the params in place (only under SelfSimilar).
    if matches!(
        field,
        EntityField::ZoneArms | EntityField::ZoneWarp | EntityField::ZoneTwist
    ) {
        adjust_scale_sub(&mut state.level.overlay.zones[idx].scale, field, delta);
        match &state.level.overlay.zones[idx].scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => {
                let (arms, warp, twist) = (p.arms, p.octave_warp, p.twist);
                state.mark_dirty();
                state.status =
                    format!("zone #{idx} self_similar arms:{arms} warp:{warp:.2} twist:{twist:.2}");
            }
            _ => {
                state.status = format!("'{}' needs scale=self_similar (c on scale)", field.label())
            }
        }
        return;
    }
    // Region corners: clamp to room, keep min <= max per axis.
    let (mut min, mut max) = overlay_zone_box(&state.level.overlay.zones[idx]);
    match field {
        EntityField::ZoneRegMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::ZoneRegMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::ZoneRegMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        EntityField::ZoneRegMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::ZoneRegMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::ZoneRegMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => {
            state.status = "zone: no such field".into();
            return;
        }
    }
    set_zone_overlay_box(&mut state.level.overlay.zones[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "zone #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// Step a SelfSimilar sub-field (arms / octave_warp / twist) of a scale rule by
/// `delta`. A non-SelfSimilar rule is left UNCHANGED (the caller reports the hint).
/// arms: min 0; octave_warp: 0.05 step, clamped ≥ 0; twist: 0.1 step (signed).
fn adjust_scale_sub(rule: &mut re_level::OverlayScaleRule, field: EntityField, delta: i32) {
    let re_level::OverlayScaleRule::SelfSimilar(p) = rule else {
        return;
    };
    match field {
        EntityField::ZoneArms | EntityField::LineArms => {
            p.arms = if delta >= 0 {
                p.arms.saturating_add(delta as u8)
            } else {
                p.arms.saturating_sub((-delta) as u8)
            };
        }
        EntityField::ZoneWarp | EntityField::LineWarp => {
            p.octave_warp = (p.octave_warp + delta as f32 * 0.05).max(0.0);
            p.octave_warp = (p.octave_warp * 100.0).round() / 100.0;
        }
        EntityField::ZoneTwist | EntityField::LineTwist => {
            p.twist += delta as f32 * 0.1;
            p.twist = (p.twist * 100.0).round() / 100.0;
        }
        _ => {}
    }
}

/// `+`/`-` on a Line focused field: step a SelfSimilar sub-field (arms / octave_warp
/// / twist, SelfSimilar only) or REMOVE the last waypoint (`-` on the `points`
/// field; `+` is a hint to append with `space`). `material`/`scale` are enum (`c`).
/// A no-op (with a hint) when no line is selected.
fn adjust_line_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if matches!(field, EntityField::LineMaterial | EntityField::LineScale) {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let Some(idx) = state
        .selected_line
        .filter(|&i| i < state.level.overlay.lines.len())
    else {
        state.status = "no line selected".into();
        return;
    };
    if field == EntityField::LinePoints {
        crate::overlay::line_adjust_points(state, idx, delta);
        return;
    }
    if matches!(
        field,
        EntityField::LineArms | EntityField::LineWarp | EntityField::LineTwist
    ) {
        adjust_scale_sub(&mut state.level.overlay.lines[idx].scale, field, delta);
        match &state.level.overlay.lines[idx].scale {
            re_level::OverlayScaleRule::SelfSimilar(p) => {
                let (arms, warp, twist) = (p.arms, p.octave_warp, p.twist);
                state.mark_dirty();
                state.status =
                    format!("line #{idx} self_similar arms:{arms} warp:{warp:.2} twist:{twist:.2}");
            }
            _ => {
                state.status = format!("'{}' needs scale=self_similar (c on scale)", field.label())
            }
        }
        return;
    }
    state.status = "line: no such field".into();
}

/// `+`/`-` on an Inset focused field: step the octave LO/HI (kept lo ≤ hi), or nudge
/// a region corner (clamped to the room AND kept `min <= max` per axis). `child` is
/// a typed text buffer. A no-op (with a hint) when no inset is selected.
fn adjust_inset_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if field == EntityField::InsetChild {
        state.status = "'child' is a typed field".into();
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_inset
        .filter(|&i| i < state.level.overlay.insets.len())
    else {
        state.status = "no inset selected".into();
        return;
    };
    // Octave range: lo kept ≤ hi, hi kept ≥ lo.
    if matches!(
        field,
        EntityField::InsetOctaveLo | EntityField::InsetOctaveHi
    ) {
        let (mut lo, mut hi) = state.level.overlay.insets[idx].at_octave;
        match field {
            EntityField::InsetOctaveLo => lo = (lo + delta).min(hi),
            EntityField::InsetOctaveHi => hi = (hi + delta).max(lo),
            _ => unreachable!(),
        }
        state.level.overlay.insets[idx].at_octave = (lo, hi);
        state.mark_dirty();
        state.status = format!("inset #{idx} octave {lo}-{hi}");
        return;
    }
    // Region corners: clamp to room, keep min <= max per axis.
    let (mut min, mut max) = inset_box(&state.level.overlay.insets[idx]);
    match field {
        EntityField::InsetRegMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::InsetRegMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::InsetRegMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        EntityField::InsetRegMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::InsetRegMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::InsetRegMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => {
            state.status = "inset: no such field".into();
            return;
        }
    }
    set_inset_box(&mut state.level.overlay.insets[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "inset #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// `c` on a Note focused field: cycle the KIND or the cardinal FACING. Pos is
/// numeric (`+`/`-`), text is typed. A no-op (with a hint) when no note is selected.
fn cycle_note_enum(state: &mut EditorState) {
    let field = state.entity_field;
    let Some(idx) = state.selected_note.filter(|&i| i < state.level.notes.len()) else {
        state.status = "no note selected".into();
        return;
    };
    match field {
        EntityField::NoteKindField => {
            let n = &mut state.level.notes[idx];
            n.kind = crate::state::cycle_note_kind(n.kind, 1);
            let label = crate::state::note_kind_label(n.kind);
            state.mark_dirty();
            state.status = format!("note #{idx} kind: {label}");
        }
        EntityField::NoteFacing => {
            // Cycling REMEMBERS a non-cardinal yaw (`state.vocab`), so resolve the
            // next value before borrowing the note mutably.
            let next = crate::state::cycle_facing(state, state.level.notes[idx].facing);
            state.level.notes[idx].facing = next;
            let label = crate::state::facing_label(next);
            state.mark_dirty();
            state.status = format!("note #{idx} facing: {label}");
        }
        _ => {
            state.status = format!("'{}' is numeric/typed: use +/- or type", field.label());
        }
    }
}

// ─── TEXT AUTHORING — the caret-based text-entry router ─────────────────────
//
// The reusable widget ops (`crate::widget`) give every text surface a REAL
// caret: mid-string insert/delete, Enter = newline (multi-line bodies), and
// arrow-key navigation. Four surfaces are wired:
//   * NOTE BODY        — multi-line, edited in place (story-critical).
//   * INSET CHILD path — single-line, edited in place.
//   * NODE pool/pool_alt — free-text WORD LIST via an edit buffer; every
//     keystroke commits `split_whitespace` back to the node (saved verbatim as
//     `pool = w1 w2 …`), replacing the auto `w0/w1…` placeholder model.
//   * TRIGGER EVENT label — free text via the same buffer (committed trimmed),
//     replacing the 6-preset cycle; these labels feed the game's
//     trigger→stinger/HUD map, so arbitrary text matters.

/// Which text surface the keyboard currently targets. `None` when no
/// text field is focused — the key dispatcher then treats keys normally.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextTarget {
    /// The selected note's multi-line body (`notes[i].text`).
    NoteBody(usize),
    /// The selected inset's child-scene path (single-line).
    InsetChild(usize),
    /// The selected node's word list: `pool` (`alt == false`) or `pool_alt`.
    NodePool(usize, bool),
    /// The selected trigger's EVENT label (only while its action IS `Event`).
    TriggerLabel(usize),
}

/// Resolve the focused text surface, if any: the object selector must be on
/// the matching kind, the matching field focused, and a valid object selected
/// (for the trigger, the action must currently be an `Event`).
pub fn text_target(state: &EditorState) -> Option<TextTarget> {
    let kind = state.place?;
    match (kind, state.entity_field) {
        (EntityKind::Note, EntityField::NoteText) => state
            .selected_note
            .filter(|&i| i < state.level.notes.len())
            .map(TextTarget::NoteBody),
        (EntityKind::Inset, EntityField::InsetChild) => state
            .selected_inset
            .filter(|&i| i < state.level.overlay.insets.len())
            .map(TextTarget::InsetChild),
        (EntityKind::Node, EntityField::NodePool) => state
            .selected_node
            .filter(|&i| i < state.level.nodes.len())
            .map(|i| TextTarget::NodePool(i, false)),
        (EntityKind::Node, EntityField::NodePoolAlt) => state
            .selected_node
            .filter(|&i| i < state.level.nodes.len())
            .map(|i| TextTarget::NodePool(i, true)),
        (EntityKind::Trigger, EntityField::TrigTarget) => state
            .selected_trigger
            .filter(|&i| i < state.level.triggers.len())
            .filter(|&i| {
                matches!(
                    state.level.triggers[i].action,
                    re_core::TriggerAction::Event { .. }
                )
            })
            .map(TextTarget::TriggerLabel),
        _ => None,
    }
}

/// The `(kind, index, field)` identity key of a text target — what
/// `EditorState::text_owner` stores so a focus/selection change re-seeds the
/// caret and buffer.
fn text_owner_key(target: TextTarget) -> (EntityKind, usize, EntityField) {
    match target {
        TextTarget::NoteBody(i) => (EntityKind::Note, i, EntityField::NoteText),
        TextTarget::InsetChild(i) => (EntityKind::Inset, i, EntityField::InsetChild),
        TextTarget::NodePool(i, false) => (EntityKind::Node, i, EntityField::NodePool),
        TextTarget::NodePool(i, true) => (EntityKind::Node, i, EntityField::NodePoolAlt),
        TextTarget::TriggerLabel(i) => (EntityKind::Trigger, i, EntityField::TrigTarget),
    }
}

/// The current SOURCE text of a target (what the buffer is seeded from).
fn text_source(state: &EditorState, target: TextTarget) -> String {
    match target {
        TextTarget::NoteBody(i) => state.level.notes[i].text.clone(),
        TextTarget::InsetChild(i) => {
            let re_level::OverlayInsetContent::ChildScene(s) =
                &state.level.overlay.insets[i].content;
            s.clone()
        }
        TextTarget::NodePool(i, false) => state.level.nodes[i].pool.join(" "),
        TextTarget::NodePool(i, true) => state.level.nodes[i].pool_alt.join(" "),
        TextTarget::TriggerLabel(i) => match &state.level.triggers[i].action {
            re_core::TriggerAction::Event { label } => label.clone(),
            _ => String::new(),
        },
    }
}

/// Seed/refresh the caret (+ buffer for buffered targets) when the focused
/// text target CHANGES — the caret lands after the last char, matching the old
/// append-only feel. A no-op while the owner is unchanged.
fn sync_text_state(state: &mut EditorState, target: TextTarget) {
    let owner = text_owner_key(target);
    if state.text_owner == Some(owner) {
        return;
    }
    let src = text_source(state, target);
    state.text_caret = crate::widget::caret_at_end(&src);
    state.text_buf = match target {
        TextTarget::NodePool(..) | TextTarget::TriggerLabel(_) => Some(src),
        _ => None,
    };
    state.text_owner = Some(owner);
}

/// The focused text field's LIVE readout (buffer + caret marker) for the
/// inspector row, or `None` when `field` is not the focused text surface.
/// Buffered fields show the BUFFER (with in-progress whitespace); direct
/// fields show the caret's line. Pure.
pub fn text_field_readout(state: &EditorState, field: EntityField) -> Option<String> {
    let target = text_target(state)?;
    let owner = text_owner_key(target);
    if owner.2 != field || state.text_owner != Some(owner) {
        return None;
    }
    let text = match (&state.text_buf, target) {
        (Some(buf), _) => buf.clone(),
        (None, _) => text_source(state, target),
    };
    Some(crate::widget::caret_line_readout(&text, state.text_caret))
}

/// `true` while a text surface has the caret — the keymap's TextEntry hint
/// context (`crate::keymap::active_context`). Pure.
pub fn text_entry_active(state: &EditorState) -> bool {
    text_target(state).is_some()
}

/// Handle one key for the focused text surface. Returns `true` when the key
/// was CONSUMED (typed/edited/navigated); `false` when no text surface is
/// focused or the key belongs to the normal 2D-edit dispatch (it falls
/// through untouched — e.g. Shift+arrows keep panning, Tab keeps cycling).
pub fn text_entry_key(state: &mut EditorState, key: re_present::EditorKey) -> bool {
    use re_present::EditorKeyCode as E;
    let Some(target) = text_target(state) else {
        return false;
    };
    sync_text_state(state, target);
    // Shift+arrows stay the pan keys even while a text field is focused.
    let plain_arrow = !key.shift && matches!(key.code, E::Left | E::Right | E::Up | E::Down);
    let multi_line = matches!(target, TextTarget::NoteBody(_));
    match key.code {
        E::Char(c) => {
            // `#` after whitespace is a comment-opener to the `.lvl` parser for
            // the single-line key-value surfaces — a label/pool word containing
            // it would be SILENTLY truncated on the next load. Refuse it there
            // (the multi-line note body is verbatim and keeps it).
            if c == '#' && !multi_line {
                state.status = "'#' not allowed here (starts a .lvl comment)".into();
                return true;
            }
            state.push_undo_checkpoint();
            edit_target_text(state, target, |text, caret| {
                crate::widget::insert_char(text, caret, c);
                true
            });
            state.drop_undo_if_unchanged();
            true
        }
        E::Backspace => {
            state.push_undo_checkpoint();
            edit_target_text(state, target, crate::widget::backspace);
            state.drop_undo_if_unchanged();
            true
        }
        // Forward-DELETE removes the char AT the caret.
        E::Delete => {
            state.push_undo_checkpoint();
            edit_target_text(state, target, crate::widget::delete);
            state.drop_undo_if_unchanged();
            true
        }
        // Home/End jump the caret to the line start/end (pure navigation).
        E::Home => {
            nav_target_text(state, target, crate::widget::home);
            true
        }
        E::End => {
            nav_target_text(state, target, crate::widget::end);
            true
        }
        E::Enter if multi_line => {
            state.push_undo_checkpoint();
            edit_target_text(state, target, |text, caret| {
                crate::widget::insert_newline(text, caret);
                true
            });
            state.drop_undo_if_unchanged();
            true
        }
        E::Left if plain_arrow => {
            nav_target_text(state, target, crate::widget::move_left);
            true
        }
        E::Right if plain_arrow => {
            nav_target_text(state, target, crate::widget::move_right);
            true
        }
        E::Up if plain_arrow && multi_line => {
            nav_target_text(state, target, crate::widget::move_up);
            true
        }
        E::Down if plain_arrow && multi_line => {
            nav_target_text(state, target, crate::widget::move_down);
            true
        }
        // Everything else (Tab, PageUp/Dn, Shift+arrows, single-line Up/Down,
        // Enter outside a note body, F-keys handled earlier) falls through.
        _ => false,
    }
}

/// Run a MUTATING widget op against the target's text (direct targets edit the
/// document string in place; buffered targets edit `text_buf` and COMMIT the
/// derived value back to the document), marking dirty on change.
fn edit_target_text(
    state: &mut EditorState,
    target: TextTarget,
    op: impl FnOnce(&mut String, &mut crate::widget::Caret) -> bool,
) {
    let mut caret = state.text_caret;
    match target {
        TextTarget::NoteBody(i) => {
            if op(&mut state.level.notes[i].text, &mut caret) {
                state.mark_dirty();
                state.schedule_preview_rebuild(); // the body renders on the note wall
            }
        }
        TextTarget::InsetChild(i) => {
            let re_level::OverlayInsetContent::ChildScene(s) =
                &mut state.level.overlay.insets[i].content;
            if op(s, &mut caret) {
                state.mark_dirty();
                state.schedule_preview_rebuild(); // child path affects build_map
            }
        }
        TextTarget::NodePool(i, alt) => {
            let mut buf = state.text_buf.take().unwrap_or_default();
            if op(&mut buf, &mut caret) {
                let words: Vec<String> = buf.split_whitespace().map(str::to_string).collect();
                let n = &mut state.level.nodes[i];
                if alt {
                    n.pool_alt = words;
                } else {
                    n.pool = words;
                }
                state.mark_dirty();
            }
            state.text_buf = Some(buf);
        }
        TextTarget::TriggerLabel(i) => {
            let mut buf = state.text_buf.take().unwrap_or_default();
            if op(&mut buf, &mut caret) {
                let label = buf.trim().to_string();
                state.level.triggers[i].action = re_core::TriggerAction::event(label);
                state.mark_dirty();
            }
            state.text_buf = Some(buf);
        }
    }
    state.text_caret = caret;
}

/// Run a NON-mutating caret-navigation widget op against the target's text.
fn nav_target_text(
    state: &mut EditorState,
    target: TextTarget,
    op: impl FnOnce(&str, &mut crate::widget::Caret),
) {
    let text = match &state.text_buf {
        Some(buf) => buf.clone(),
        None => text_source(state, target),
    };
    let mut caret = state.text_caret;
    op(&text, &mut caret);
    state.text_caret = caret;
}

/// Helper: render the SELECTED trigger's box corners via `f`, or `"-"` when none
/// is selected. `f` receives `((minx,miny,minz),(maxx,maxy,maxz))`.
fn trigger_field_str(
    state: &EditorState,
    f: impl Fn(((i32, i32, i32), (i32, i32, i32))) -> String,
) -> String {
    match state
        .selected_trigger
        .filter(|&i| i < state.level.triggers.len())
    {
        Some(i) => f(trigger_box(&state.level.triggers[i])),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED trigger via `f` (which gets the whole
/// state + the trigger, so it can resolve node names), or `"-"` when none.
fn trigger_value(
    state: &EditorState,
    f: impl Fn(&EditorState, &re_core::Trigger) -> String,
) -> String {
    match state
        .selected_trigger
        .filter(|&i| i < state.level.triggers.len())
    {
        Some(i) => f(state, &state.level.triggers[i]),
        None => "-".to_string(),
    }
}

/// The door-target readout: the cursor node's name, whether it is currently an
/// on/off target, and the cursor position within the node list.
fn door_target_value(state: &EditorState) -> String {
    let nodes = &state.level.nodes;
    if nodes.is_empty() {
        return "no nodes".to_string();
    }
    let idx = state.door_target_cursor.min(nodes.len() - 1);
    let name = &nodes[idx].name;
    let on = state.level.door.target_parts.contains(&idx);
    let off = state.level.door.target_off.contains(&idx);
    let mark = if on {
        "ON"
    } else if off {
        "OFF"
    } else {
        "-"
    };
    format!("[{}/{}] '{name}' {mark}", idx + 1, nodes.len())
}

/// Helper: render a field of the SELECTED node via `f`, or `"-"` when none is
/// selected. Keeps `entity_field_value` flat.
fn node_field_str(state: &EditorState, f: impl Fn(&re_level::Node) -> String) -> String {
    match state.selected_node.filter(|&i| i < state.level.nodes.len()) {
        Some(i) => f(&state.level.nodes[i]),
        None => "-".to_string(),
    }
}

/// Helper: render a field of the SELECTED part via `f`, or `"-"` when none is
/// selected. Mirrors [`node_field_str`].
fn part_field_str(state: &EditorState, f: impl Fn(&re_level::Part) -> String) -> String {
    match state.selected_part.filter(|&i| i < state.level.parts.len()) {
        Some(i) => f(&state.level.parts[i]),
        None => "-".to_string(),
    }
}

/// `c` on a Trigger focused field: cycle the ACTION TYPE (ActivateNode ↔ Event),
/// the TARGET (node for ActivateNode, preset label for Event — `delta=+1`), or
/// toggle `once`. Region corners are numeric (`+`/`-`) and are hinted as such.
/// A no-op (with a hint) when no trigger is selected.
fn cycle_trigger_enum(state: &mut EditorState) {
    let field = state.entity_field;
    let n_nodes = state.level.nodes.len();
    let Some(idx) = state
        .selected_trigger
        .filter(|&i| i < state.level.triggers.len())
    else {
        state.status = "no trigger selected".into();
        return;
    };
    match field {
        EntityField::TrigActionType => {
            // ActivateNode → Event, Event → ActivateNode. Switching to ActivateNode
            // resolves to the first node (or stays unresolved when none exist).
            let t = &mut state.level.triggers[idx];
            t.action = match &t.action {
                re_core::TriggerAction::ActivateNode { .. } => {
                    re_core::TriggerAction::event(crate::state::TRIGGER_EVENT_LABELS[0])
                }
                _ => {
                    let mut a = re_core::TriggerAction::activate_node(
                        state
                            .level
                            .nodes
                            .first()
                            .map(|n| n.name.clone())
                            .unwrap_or_default(),
                    );
                    if let re_core::TriggerAction::ActivateNode { node_index, .. } = &mut a {
                        if n_nodes > 0 {
                            *node_index = 0;
                        }
                    }
                    a
                }
            };
            state.mark_dirty();
            let label = trigger_action_label(state, &state.level.triggers[idx]);
            state.status = format!("trigger #{idx} action: {label}");
        }
        EntityField::TrigTarget => {
            cycle_trigger_target(state, idx, 1);
        }
        EntityField::TrigOnce => {
            let t = &mut state.level.triggers[idx];
            t.once = !t.once;
            let once = t.once;
            state.mark_dirty();
            state.status = format!("trigger #{idx} once: {once}");
        }
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
        }
    }
}

/// Step the selected trigger's TARGET by `delta`: for `ActivateNode`, move the
/// target through `level.nodes` (wrapping, keeping `name` + `node_index` in sync);
/// for `Event`, step the preset label list. A defined no-op when ActivateNode has
/// no nodes to point at.
fn cycle_trigger_target(state: &mut EditorState, idx: usize, delta: i32) {
    let n_nodes = state.level.nodes.len();
    let action = state.level.triggers[idx].action.clone();
    match action {
        re_core::TriggerAction::ActivateNode { node_index, .. } => {
            if n_nodes == 0 {
                state.status = "no nodes to target (place a Node first)".into();
                return;
            }
            let cur = node_index.min(n_nodes - 1) as i32;
            let next = (cur + delta).rem_euclid(n_nodes as i32) as usize;
            let name = state.level.nodes[next].name.clone();
            state.level.triggers[idx].action = re_core::TriggerAction::ActivateNode {
                name: name.clone(),
                node_index: next,
            };
            state.mark_dirty();
            state.status = format!("trigger #{idx} -> node '{name}'");
        }
        re_core::TriggerAction::Event { label } => {
            let next = crate::state::cycle_event_label(state, &label, delta);
            state.level.triggers[idx].action = re_core::TriggerAction::event(&next);
            state.mark_dirty();
            state.status = format!("trigger #{idx} event '{next}'");
        }
        _ => {
            state.status = "unknown action".into();
        }
    }
}

/// Cycle the FOCUSED emitter ENUM field (`c`): DirectionAxis preset, ColorMode,
/// Family name, or Bold policy. A no-op with a clear status when the focused
/// field is numeric (use `+`/`-`), or no emitter is selected.
fn cycle_emitter_enum(state: &mut EditorState) {
    let field = state.emitter_field;
    let Some(idx) = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
    else {
        state.status = "no emitter selected".into();
        return;
    };
    // The three OPEN-domain fields (an arbitrary unit vector, an arbitrary glyph
    // sequence, an arbitrary family name) cycle through `state.vocab`, which they
    // also WRITE — so they are resolved before the emitter is borrowed mutably.
    // Each used to snap an authored off-list value to index 0: `direction` to
    // `+Z` (rotating `21-emitter-showcase.lvl`'s `1.0 0.0 0.25` emitter by ~76°),
    // `glyph_set` to `none` (erasing that level's authored `· : * ✦ ✧`), and
    // `family` to `haze_o`.
    if matches!(
        field,
        EmitterField::DirectionAxis | EmitterField::GlyphSet | EmitterField::Family
    ) {
        let msg = cycle_emitter_open_domain(state, idx, field);
        state.level.emitters[idx].validate();
        state.mark_dirty();
        state.status = msg;
        return;
    }
    let e = &mut state.level.emitters[idx];
    let msg = match field {
        EmitterField::Kind => {
            e.kind = match e.kind {
                EmitterKind::Standard => EmitterKind::GodRays,
                EmitterKind::GodRays => EmitterKind::Standard,
            };
            format!("kind {}", kind_label(e.kind))
        }
        EmitterField::ColorMode => {
            e.color_mode = match e.color_mode {
                ColorMode::RampOverAge => ColorMode::RandomPick,
                ColorMode::RandomPick => ColorMode::Fixed,
                ColorMode::Fixed => ColorMode::RampOverAge,
            };
            format!("color_mode {}", color_mode_label(e.color_mode))
        }
        EmitterField::Bold => {
            e.bold = match e.bold {
                BoldMode::Off => BoldMode::On,
                BoldMode::On => BoldMode::ByProximity,
                BoldMode::ByProximity => BoldMode::ByAge,
                BoldMode::ByAge => BoldMode::Off,
            };
            format!("bold {}", bold_label(e.bold))
        }
        EmitterField::Blend => {
            e.blend = match e.blend {
                EmitterBlend::KeepBg => EmitterBlend::Opaque,
                EmitterBlend::Opaque => EmitterBlend::KeepBg,
            };
            format!("blend {}", blend_label(e.blend))
        }
        // Numeric fields are adjusted with +/-, not c.
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
            return;
        }
    };
    e.validate();
    state.mark_dirty();
    state.status = msg;
}

/// The list a `direction` cycle walks, as labels — the six cardinal presets plus
/// every non-cardinal direction this session has seen, plus the current one.
pub fn direction_options(state: &EditorState, current: &[f32; 3]) -> Vec<[f32; 3]> {
    crate::vocab::options_at_by(
        &direction_canon(),
        &state.vocab.direction,
        current,
        crate::vocab::dir_eq,
    )
    .0
}

/// The list a `glyph_set` cycle walks — the [`GLYPH_SETS`] presets plus every
/// bespoke sequence this session has seen, plus the current one.
pub fn glyph_set_options(
    state: &EditorState,
    current: Option<&Vec<char>>,
) -> Vec<Option<Vec<char>>> {
    crate::vocab::options_at(
        &glyph_set_canon(),
        &state.vocab.glyph_set,
        &current.cloned(),
    )
    .0
}

/// The list a `family` cycle walks — `re_render`'s glyph-family library plus
/// every unknown family name this session has seen, plus the current one.
pub fn family_options(state: &EditorState, current: &str) -> Vec<String> {
    crate::vocab::options_at(
        &crate::vocab::emitter_families(),
        &state.vocab.family,
        &current.to_string(),
    )
    .0
}

/// The three OPEN-domain emitter cycles (`direction`, `glyph_set`, `family`),
/// hoisted out of [`cycle_emitter_enum`] because each REMEMBERS the value it
/// steps off in `state.vocab` and so needs `&mut EditorState` rather than a
/// mutable borrow of the emitter. Returns the status line.
fn cycle_emitter_open_domain(state: &mut EditorState, idx: usize, field: EmitterField) -> String {
    match field {
        EmitterField::DirectionAxis => {
            let d = state.level.emitters[idx].direction;
            let cur = [d.x, d.y, d.z];
            crate::vocab::remember_by(
                &direction_canon(),
                &mut state.vocab.direction,
                &cur,
                crate::vocab::dir_eq,
            );
            let next = crate::vocab::step_by(
                &direction_canon(),
                &state.vocab.direction,
                &cur,
                1,
                crate::vocab::dir_eq,
            );
            state.level.emitters[idx].direction = glam::Vec3::new(next[0], next[1], next[2]);
            format!("dir {}", direction_axis_label(&next))
        }
        EmitterField::GlyphSet => {
            let cur = state.level.emitters[idx].glyph_set.clone();
            let canon = glyph_set_canon();
            crate::vocab::remember(&canon, &mut state.vocab.glyph_set, &cur);
            let next = crate::vocab::step(&canon, &state.vocab.glyph_set, &cur, 1);
            state.level.emitters[idx].glyph_set = next.clone();
            format!("glyph_set {}", glyph_set_option_label(next.as_ref()))
        }
        // `Family` — the only remaining variant the caller routes here.
        _ => {
            let cur = state.level.emitters[idx].family.clone();
            let canon = crate::vocab::emitter_families();
            crate::vocab::remember(&canon, &mut state.vocab.family, &cur);
            let next = crate::vocab::step(&canon, &state.vocab.family, &cur, 1);
            state.level.emitters[idx].family.clone_from(&next);
            format!("family {next}")
        }
    }
}

/// Step the FOCUSED emitter field by `delta` (±1) through `EMITTER_FIELDS`
/// (the `;`/`'` keys), wrapping. Reports the new focus + its current value. A
/// no-op (with a hint) when the object selector isn't on Emitter.
pub fn cycle_emitter_field(state: &mut EditorState, delta: i32) {
    if state.place != Some(EntityKind::Emitter) {
        state.status = "select Emitter (o) first".into();
        return;
    }
    state.emitter_field = state.emitter_field.step(delta);
    state.status = format!(
        "focus: {} = {}",
        state.emitter_field.label(),
        emitter_field_value(state)
    );
}

/// Open the 3×3×3 DIRECTION PICKER (`D`) for the currently-selected emitter. The
/// cube cursor starts on the cell matching the emitter's current `direction` so
/// the overlay opens "where the aim already is". A no-op with a clear status when
/// the object selector isn't on Emitter or no emitter is selected.
pub fn open_dir_picker(state: &mut EditorState) {
    if state.place != Some(EntityKind::Emitter) {
        state.status = "select Emitter (o) first".into();
        return;
    }
    let Some(idx) = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
    else {
        state.status = "no emitter selected (space on one first)".into();
        return;
    };
    let dir = state.level.emitters[idx].direction;
    state.dir_picker = Some(crate::picker::DirPickerState::for_direction(idx, dir));
    state.status = "aim picker: arrows move, [/] layer, Enter pick, Esc cancel".into();
}

/// PICK the cube cell under the direction-picker cursor: set the selected
/// emitter's `direction` to `normalize((dx, dy, dz))` (the centre cell keeps the
/// current aim) and close the overlay. Snapshots undo so the aim change is one
/// undoable step. A defined no-op if the picker or emitter index is stale.
pub fn pick_dir_picker_cell(state: &mut EditorState) {
    let Some(p) = state.dir_picker else {
        return;
    };
    state.dir_picker = None;
    let idx = p.emitter;
    if idx >= state.level.emitters.len() {
        state.status = "emitter gone — aim unchanged".into();
        return;
    }
    match p.cursor_cell().direction() {
        Some(dir) => {
            state.push_undo();
            state.schedule_preview_rebuild(); // emitter aim shows in preview
            let e = &mut state.level.emitters[idx];
            e.direction = dir;
            e.validate();
            state.mark_dirty();
            state.status = format!(
                "aim {} {}",
                crate::picker::direction_to_arrow_str(dir),
                direction_label(&state.level.emitters[idx])
            );
        }
        None => {
            // Centre cell: no direction — keep the current aim, just close.
            state.status = "aim unchanged (centre)".into();
        }
    }
}

/// Render the focused field's current value as a compact string for the panel /
/// status line. `"-"` when no emitter is selected.
pub fn emitter_field_value(state: &EditorState) -> String {
    emitter_field_value_of(state, state.emitter_field)
}

/// Render an ARBITRARY emitter field's current value (the inspector row model
/// gives every emitter field its own row). `"-"` when no emitter is selected.
pub fn emitter_field_value_of(state: &EditorState, field: EmitterField) -> String {
    let Some(e) = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
        .map(|i| &state.level.emitters[i])
    else {
        return "-".into();
    };
    let (ox, oy, oz) = origin_xyz(e);
    match field {
        EmitterField::Kind => kind_label(e.kind).to_string(),
        EmitterField::Rate => format!("{:.1}", e.rate),
        EmitterField::Burst => match e.burst {
            Some(n) => n.to_string(),
            None => "none".to_string(),
        },
        EmitterField::MaxParticles => e.max_particles.to_string(),
        EmitterField::Speed => format!("{:.2}", e.speed),
        EmitterField::SpeedSpread => format!("{:.2}", e.speed_spread),
        EmitterField::SpreadDeg => format!("{:.0}", e.spread_deg),
        EmitterField::Lifetime => format!("{:.2}", e.lifetime),
        EmitterField::LifetimeSpread => format!("{:.2}", e.lifetime_spread),
        EmitterField::Size => format!("{:.2}", e.size),
        EmitterField::Growth => format!("{:.2}", e.growth),
        EmitterField::AccelX => format!("{:.2}", e.accel.x),
        EmitterField::AccelY => format!("{:.2}", e.accel.y),
        EmitterField::AccelZ => format!("{:.2}", e.accel.z),
        EmitterField::Swirl => format!("{:.2}", e.swirl),
        EmitterField::PathJitter => format!("{:.2}", e.path_jitter),
        EmitterField::Color0 => format!("#{:06X}", e.colors[0] & 0xFF_FFFF),
        EmitterField::Color1 => format!("#{:06X}", e.colors[1] & 0xFF_FFFF),
        EmitterField::Color2 => format!("#{:06X}", e.colors[2] & 0xFF_FFFF),
        EmitterField::DirectionAxis => direction_label(e),
        EmitterField::ColorMode => color_mode_label(e.color_mode).to_string(),
        EmitterField::Family => e.family.clone(),
        EmitterField::GlyphSet => glyph_set_label(e),
        EmitterField::Bold => bold_label(e.bold).to_string(),
        EmitterField::Blend => blend_label(e.blend).to_string(),
        EmitterField::Seed => e.seed.to_string(),
        EmitterField::OriginX => format!("{ox:.1}"),
        EmitterField::OriginY => format!("{oy:.1}"),
        EmitterField::OriginZ => format!("{oz:.1}"),
    }
}

/// Nudge the FOCUSED numeric property by `delta` (±1) for the active kind, routed
/// by the focused [`EntityField`] so every numeric/selection field is reachable:
/// * Door → move the target-cursor (on the two target fields) or step pos x/y/z,
/// * Node → pos x/y/z (clamped to the room), color (palette step), threshold,
///   anchor, pool/pool_alt word add/remove,
/// * Pedestal → center x/y/z (clamped) or per-axis volume w/h/d (1..=max),
/// * Spawn → pos x/y/z (clamped),
/// * Emitter → the focused emitter numeric field (existing behaviour).
pub fn adjust_property(state: &mut EditorState, delta: i32) {
    ensure_entity_focus(state);
    state.schedule_preview_rebuild(); // debounced preview catch-up
                                      // GROUP-MOVE: snapshot every nested object's position before the edit, then
                                      // (after it) push the SAME delta onto every descendant of anything that moved.
                                      // A hierarchy-less level snapshots EMPTY, so this is byte-for-byte transparent.
    let before = crate::hierarchy::snapshot_positions(state);
    match state.place {
        Some(EntityKind::Emitter) => adjust_emitter_numeric(state, delta),
        Some(EntityKind::Door) => adjust_door_numeric(state, delta),
        Some(EntityKind::Node) => adjust_node_numeric(state, delta),
        Some(EntityKind::Pedestal) => adjust_pedestal_numeric(state, delta),
        Some(EntityKind::Part) => adjust_part_numeric(state, delta),
        Some(EntityKind::Spawn) => adjust_spawn_numeric(state, delta),
        Some(EntityKind::Trigger) => adjust_trigger_numeric(state, delta),
        Some(EntityKind::Pipe) => crate::pipe::adjust_numeric(state, delta),
        Some(EntityKind::ZoneInput) => adjust_zone_numeric(state, delta),
        Some(EntityKind::Note) => adjust_note_numeric(state, delta),
        Some(EntityKind::WallDecal) => adjust_decal_numeric(state, delta),
        Some(EntityKind::Holo) => adjust_holo_numeric(state, delta),
        Some(EntityKind::Portal) => adjust_portal_numeric(state, delta),
        Some(EntityKind::Zone) => adjust_zone_overlay_numeric(state, delta),
        Some(EntityKind::Line) => adjust_line_numeric(state, delta),
        Some(EntityKind::Inset) => adjust_inset_numeric(state, delta),
        Some(EntityKind::Field) => crate::field::adjust_field_numeric(state, delta),
        None => {
            state.status = "no adjustable property".into();
        }
    }
    crate::hierarchy::apply_group_moves(state, &before);
}

/// Clamp a coordinate to `0..extent` (the room dimension), saturating. A negative
/// or over-large result is pulled back into the grid so a placed entity never
/// leaves the room (the parser rejects out-of-bounds node `pos`).
fn clamp_axis(v: i32, extent: usize) -> i32 {
    v.clamp(0, extent.saturating_sub(1) as i32)
}

/// `+`/`-` on a Door focused field: move the target-cursor through the node list
/// (the two target fields) or step the door anchor pos.
fn adjust_door_numeric(state: &mut EditorState, delta: i32) {
    let (sx, sy, sz) = state.map.size();
    match state.entity_field {
        EntityField::DoorTargetOn | EntityField::DoorTargetOff => {
            let n = state.level.nodes.len();
            if n == 0 {
                state.status = "no nodes to target".into();
                return;
            }
            let cur = state.door_target_cursor.min(n - 1) as i32;
            state.door_target_cursor = (cur + delta).rem_euclid(n as i32) as usize;
            state.status = format!("door target {}", door_target_value(state));
        }
        EntityField::DoorPosX => {
            state.level.door.x = clamp_axis(state.level.door.x + delta, sx);
            mark_pos(state, "door");
        }
        EntityField::DoorPosY => {
            state.level.door.y = clamp_axis(state.level.door.y + delta, sy);
            mark_pos(state, "door");
        }
        EntityField::DoorPosZ => {
            state.level.door.z = clamp_axis(state.level.door.z + delta, sz);
            mark_pos(state, "door");
        }
        _ => state.status = "use c to toggle this field".into(),
    }
}

/// `+`/`-` on a Node focused field. Pos is clamped to the room; pool/pool_alt
/// add a uniquely-named word on `+` and pop the last on `-`.
fn adjust_node_numeric(state: &mut EditorState, delta: i32) {
    let (sx, sy, sz) = state.map.size();
    let field = state.entity_field;
    if matches!(field, EntityField::NodeGlyph | EntityField::NodeIo) {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    // NodeEmits: `+`/`-` move the emit-part cursor through `level.parts` (the `c`
    // key then toggles the clause), mirroring the door target cursor.
    if field == EntityField::NodeEmits {
        if state
            .selected_node
            .filter(|&i| i < state.level.nodes.len())
            .is_none()
        {
            state.status = "no node selected".into();
            return;
        }
        let n = state.level.parts.len();
        if n == 0 {
            state.status = "no parts to light".into();
            return;
        }
        let cur = state.emit_part_cursor.min(n - 1) as i32;
        state.emit_part_cursor = (cur + delta).rem_euclid(n as i32) as usize;
        state.status = format!("node emit {}", node_emits_value(state));
        return;
    }
    let Some(idx) = state.selected_node.filter(|&i| i < state.level.nodes.len()) else {
        state.status = "no node selected".into();
        return;
    };
    let n = &mut state.level.nodes[idx];
    let name = n.name.clone();
    let msg = match field {
        EntityField::NodePosX => {
            n.x = clamp_axis(n.x + delta, sx);
            format!("node '{name}' x {}", n.x)
        }
        EntityField::NodePosY => {
            n.y = clamp_axis(n.y + delta, sy);
            format!("node '{name}' y {}", n.y)
        }
        EntityField::NodePosZ => {
            n.z = clamp_axis(n.z + delta, sz);
            format!("node '{name}' z {}", n.z)
        }
        EntityField::NodeColor => {
            n.color_rgb = step_node_color(n.color_rgb, delta);
            format!("node '{name}' color #{:06X}", n.color_rgb & 0xFF_FFFF)
        }
        EntityField::NodeThreshold => {
            n.threshold += delta;
            format!("node '{name}' threshold {}", n.threshold)
        }
        EntityField::NodeAnchor => {
            let next = n.anchor as i64 + delta as i64;
            n.anchor = next.max(0) as usize;
            format!("node '{name}' anchor {}", n.anchor)
        }
        EntityField::NodePool => {
            adjust_word_list(&mut n.pool, "w", delta);
            format!("node '{name}' pool: {} words", n.pool.len())
        }
        EntityField::NodePoolAlt => {
            adjust_word_list(&mut n.pool_alt, "a", delta);
            format!("node '{name}' pool_alt: {} words", n.pool_alt.len())
        }
        _ => unreachable!("enum fields guarded above"),
    };
    state.mark_dirty();
    state.status = msg;
}

/// Grow (`+`) or shrink (`-`) a word list: append an auto-named placeholder word
/// (`<prefix><k>`) on a positive step, pop the last on a negative step. The
/// placeholder is a valid single token so it round-trips through `pool =`.
fn adjust_word_list(list: &mut Vec<String>, prefix: &str, delta: i32) {
    if delta > 0 {
        let word = format!("{prefix}{}", list.len());
        list.push(word);
    } else if delta < 0 {
        list.pop();
    }
}

/// `+`/`-` on a Pedestal focused field: center x/y/z (clamped to the room) or
/// per-axis volume w/h/d (each clamped `1..=PEDESTAL_MAX_VOL`).
fn adjust_pedestal_numeric(state: &mut EditorState, delta: i32) {
    let (sx, sy, sz) = state.map.size();
    let p = &mut state.level.pedestal;
    let msg = match state.entity_field {
        EntityField::PedCenterX => {
            p.cx = clamp_axis(p.cx + delta, sx);
            format!("pedestal cx {}", p.cx)
        }
        EntityField::PedCenterY => {
            p.cy = clamp_axis(p.cy + delta, sy);
            format!("pedestal cy {}", p.cy)
        }
        EntityField::PedCenterZ => {
            p.cz = clamp_axis(p.cz + delta, sz);
            format!("pedestal cz {}", p.cz)
        }
        EntityField::PedVolW => {
            p.w = (p.w.max(1) + delta).clamp(1, PEDESTAL_MAX_VOL);
            format!("pedestal w {}", p.w)
        }
        EntityField::PedVolH => {
            p.h = (p.h.max(1) + delta).clamp(1, PEDESTAL_MAX_VOL);
            format!("pedestal h {}", p.h)
        }
        EntityField::PedVolD => {
            p.d = (p.d.max(1) + delta).clamp(1, PEDESTAL_MAX_VOL);
            format!("pedestal d {}", p.d)
        }
        _ => "pedestal: no such field".to_string(),
    };
    state.mark_dirty();
    state.status = msg;
}

/// `+`/`-` on a Part focused field: per-axis size w/h/d (clamped `>= 1`, RESIZING
/// the mask via [`crate::part::resize_mask`] so the mask length stays `w*h*d`),
/// world anchor off x/y/z (clamped to the room), or `color` (palette step). A
/// no-op (with a hint) when no part is selected.
fn adjust_part_numeric(state: &mut EditorState, delta: i32) {
    let (sx, sy, sz) = state.map.size();
    let field = state.entity_field;
    let Some(idx) = state.selected_part.filter(|&i| i < state.level.parts.len()) else {
        state.status = "no part selected".into();
        return;
    };
    let msg = match field {
        // Size edits RESIZE the mask (SSOT through `part::resize_mask`), clamped >=1.
        EntityField::PartSizeW => {
            let p = &state.level.parts[idx];
            let (w, h, d) = ((p.w + delta).max(1), p.h, p.d);
            crate::part::resize_mask(&mut state.level.parts[idx], w, h, d);
            format!("part w {}", state.level.parts[idx].w)
        }
        EntityField::PartSizeH => {
            let p = &state.level.parts[idx];
            let (w, h, d) = (p.w, (p.h + delta).max(1), p.d);
            crate::part::resize_mask(&mut state.level.parts[idx], w, h, d);
            // Resizing h may shrink the active mask layer out of range.
            state.mask_layer = state.mask_layer.min((h as usize).saturating_sub(1));
            format!("part h {}", state.level.parts[idx].h)
        }
        EntityField::PartSizeD => {
            let p = &state.level.parts[idx];
            let (w, h, d) = (p.w, p.h, (p.d + delta).max(1));
            crate::part::resize_mask(&mut state.level.parts[idx], w, h, d);
            format!("part d {}", state.level.parts[idx].d)
        }
        EntityField::PartOffX => {
            let p = &mut state.level.parts[idx];
            p.off_x = clamp_axis(p.off_x + delta, sx);
            format!("part offX {}", p.off_x)
        }
        EntityField::PartOffY => {
            let p = &mut state.level.parts[idx];
            p.off_y = clamp_axis(p.off_y + delta, sy);
            format!("part offY {}", p.off_y)
        }
        EntityField::PartOffZ => {
            let p = &mut state.level.parts[idx];
            p.off_z = clamp_axis(p.off_z + delta, sz);
            format!("part offZ {}", p.off_z)
        }
        EntityField::PartColor => {
            let p = &mut state.level.parts[idx];
            p.color_rgb = step_stop(p.color_rgb, delta);
            format!("part color #{:06X}", p.color_rgb & 0xFF_FFFF)
        }
        _ => "part: no such field".to_string(),
    };
    state.mark_dirty();
    state.status = msg;
}

/// `+`/`-` on a Spawn focused field: pos x/y/z (clamped to the room). Facing is
/// the enum field (`c`).
fn adjust_spawn_numeric(state: &mut EditorState, delta: i32) {
    let (sx, sy, sz) = state.map.size();
    let pl = &mut state.level.player;
    let msg = match state.entity_field {
        EntityField::SpawnPosX => {
            pl.x = clamp_axis(pl.x + delta, sx);
            format!("spawn x {}", pl.x)
        }
        EntityField::SpawnPosY => {
            pl.y = clamp_axis(pl.y + delta, sy);
            format!("spawn y {}", pl.y)
        }
        EntityField::SpawnPosZ => {
            pl.z = clamp_axis(pl.z + delta, sz);
            format!("spawn z {}", pl.z)
        }
        EntityField::SpawnFacing => "facing is an enum: use c".to_string(),
        _ => "spawn: no such field".to_string(),
    };
    state.mark_dirty();
    state.status = msg;
}

/// `+`/`-` on a Trigger focused field: nudge a region corner (clamped to the room
/// AND kept `min <= max` per axis), or step the action TARGET (node / event label).
/// Action type + `once` are enum fields (use `c`); a no-op (with a hint) when no
/// trigger is selected.
fn adjust_trigger_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if matches!(field, EntityField::TrigActionType | EntityField::TrigOnce) {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_trigger
        .filter(|&i| i < state.level.triggers.len())
    else {
        state.status = "no trigger selected".into();
        return;
    };
    // The target field steps the node/label through the same path as `c`.
    if field == EntityField::TrigTarget {
        cycle_trigger_target(state, idx, delta);
        return;
    }
    let (mut min, mut max) = trigger_box(&state.level.triggers[idx]);
    match field {
        // min corners: clamp to room, then keep <= the matching max.
        EntityField::TrigMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::TrigMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::TrigMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        // max corners: clamp to room, then keep >= the matching min.
        EntityField::TrigMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::TrigMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::TrigMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => unreachable!("enum/target fields handled above"),
    }
    set_trigger_box(&mut state.level.triggers[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "trigger #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// `+`/`-` on a ZoneInput focused field: nudge a region corner (clamped to the
/// room AND kept `min <= max` per axis), or step the linked input node (the
/// `ZoneNode` field uses the same path as `c`). A no-op (with a hint) when no zone
/// is selected.
fn adjust_zone_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_zone_input
        .filter(|&i| i < state.level.zone_inputs.len())
    else {
        state.status = "no zone input selected".into();
        return;
    };
    // The input-node field steps the node through the same path as `c`.
    if field == EntityField::ZoneNode {
        cycle_zone_node(state, idx, delta);
        return;
    }
    let (mut min, mut max) = zone_box(&state.level.zone_inputs[idx]);
    match field {
        EntityField::ZoneMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::ZoneMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::ZoneMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        EntityField::ZoneMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::ZoneMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::ZoneMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => {
            state.status = "zone: no such field".into();
            return;
        }
    }
    set_zone_box(&mut state.level.zone_inputs[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "zone #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// `+`/`-` on a WallDecal focused field: nudge a region corner (clamped to the
/// room AND kept `min <= max` per axis), step `density` (clamped `0..=1`, mirroring
/// `PipeDensity`), or step a colour (`color head` / `color tail`) through the shared
/// palette. `surface` is the enum field (`c`). A no-op (with a hint) when no decal
/// is selected.
fn adjust_decal_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if field == EntityField::DecalSurface {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_decal
        .filter(|&i| i < state.level.decals.len())
    else {
        state.status = "no wall decal selected".into();
        return;
    };
    // Non-region scalar/colour fields edit the decal directly.
    match field {
        EntityField::DecalDensity => {
            let d = &mut state.level.decals[idx];
            let v = (d.density + delta as f32 * 0.1).clamp(0.0, 1.0);
            // Snap to one decimal so repeated steps stay clean (mirrors PipeDensity).
            d.density = (v * 10.0).round() / 10.0;
            let den = d.density;
            state.mark_dirty();
            state.status = format!("wall decal #{idx} density {den:.1}");
            return;
        }
        EntityField::DecalColorHead => {
            let d = &mut state.level.decals[idx];
            d.color_head = step_stop(d.color_head, delta);
            let head = d.color_head & 0xFF_FFFF;
            state.mark_dirty();
            state.status = format!("wall decal #{idx} color head #{head:06X}");
            return;
        }
        EntityField::DecalColorTail => {
            let d = &mut state.level.decals[idx];
            d.color_tail = step_stop(d.color_tail, delta);
            let tail = d.color_tail & 0xFF_FFFF;
            state.mark_dirty();
            state.status = format!("wall decal #{idx} color tail #{tail:06X}");
            return;
        }
        _ => {}
    }
    // Region corners: clamp to room, keep min <= max per axis.
    let (mut min, mut max) = decal_box(&state.level.decals[idx]);
    match field {
        EntityField::DecalMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::DecalMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::DecalMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        EntityField::DecalMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::DecalMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::DecalMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => {
            state.status = "wall decal: no such field".into();
            return;
        }
    }
    set_decal_box(&mut state.level.decals[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "wall decal #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// `+`/`-` on a Holo focused field: nudge a region corner (clamped to the room AND
/// kept `min <= max` per axis) or step the `style` tag (a `u16`, saturating). The
/// `shape` is the enum field (`c`). A no-op (with a hint) when no hologram is
/// selected. Mirrors [`adjust_decal_numeric`].
fn adjust_holo_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if field == EntityField::HoloShapeField {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state.selected_holo.filter(|&i| i < state.level.holos.len()) else {
        state.status = "no hologram selected".into();
        return;
    };
    // The `style` scalar edits the hologram directly (saturating `u16`, min 0).
    if field == EntityField::HoloStyle {
        let h = &mut state.level.holos[idx];
        h.style = if delta >= 0 {
            h.style.saturating_add(delta as u16)
        } else {
            h.style.saturating_sub((-delta) as u16)
        };
        let style = h.style;
        state.mark_dirty();
        state.status = format!("hologram #{idx} style {style}");
        return;
    }
    // Region corners: clamp to room, keep min <= max per axis.
    let (mut min, mut max) = holo_box(&state.level.holos[idx]);
    match field {
        EntityField::HoloMinX => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
        EntityField::HoloMinY => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
        EntityField::HoloMinZ => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
        EntityField::HoloMaxX => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
        EntityField::HoloMaxY => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
        EntityField::HoloMaxZ => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
        _ => {
            state.status = "hologram: no such field".into();
            return;
        }
    }
    set_holo_box(&mut state.level.holos[idx], min, max);
    state.mark_dirty();
    state.status = format!(
        "hologram #{idx} region ({},{},{})-({},{},{})",
        min.0, min.1, min.2, max.0, max.1, max.2
    );
}

/// `+`/`-` on a Portal focused field: nudge a corner of Region A or Region B
/// (clamped to the room AND kept `min <= max` per axis), or step `rotate` by 15°.
/// The `skin` is the enum field (`c`). A no-op (with a hint) when no portal is
/// selected. Mirrors [`adjust_holo_numeric`] but spans both regions + rotate.
fn adjust_portal_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if field == EntityField::PortalSkin {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state
        .selected_portal
        .filter(|&i| i < state.level.portals.len())
    else {
        state.status = "no portal selected".into();
        return;
    };
    // `rotate` is a free yaw scalar (degrees), stepped by 15° per press.
    if field == EntityField::PortalRotate {
        let p = &mut state.level.portals[idx];
        p.rotate += delta as f32 * 15.0;
        let r = p.rotate;
        state.mark_dirty();
        state.status = format!("portal #{idx} rotate {r}");
        return;
    }
    // Region A corners.
    if matches!(
        field,
        EntityField::PortalAx0
            | EntityField::PortalAy0
            | EntityField::PortalAz0
            | EntityField::PortalAx1
            | EntityField::PortalAy1
            | EntityField::PortalAz1
    ) {
        let (mut min, mut max) = portal_a_box(&state.level.portals[idx]);
        match field {
            EntityField::PortalAx0 => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
            EntityField::PortalAy0 => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
            EntityField::PortalAz0 => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
            EntityField::PortalAx1 => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
            EntityField::PortalAy1 => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
            EntityField::PortalAz1 => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
            _ => unreachable!(),
        }
        set_portal_a_box(&mut state.level.portals[idx], min, max);
        state.mark_dirty();
        state.status = format!(
            "portal #{idx} region A ({},{},{})-({},{},{})",
            min.0, min.1, min.2, max.0, max.1, max.2
        );
        warn_if_portal_invalid(state, idx);
        return;
    }
    // Region B corners.
    if matches!(
        field,
        EntityField::PortalBx0
            | EntityField::PortalBy0
            | EntityField::PortalBz0
            | EntityField::PortalBx1
            | EntityField::PortalBy1
            | EntityField::PortalBz1
    ) {
        let (mut min, mut max) = portal_b_box(&state.level.portals[idx]);
        match field {
            EntityField::PortalBx0 => min.0 = clamp_axis(min.0 + delta, sx).min(max.0),
            EntityField::PortalBy0 => min.1 = clamp_axis(min.1 + delta, sy).min(max.1),
            EntityField::PortalBz0 => min.2 = clamp_axis(min.2 + delta, sz).min(max.2),
            EntityField::PortalBx1 => max.0 = clamp_axis(max.0 + delta, sx).max(min.0),
            EntityField::PortalBy1 => max.1 = clamp_axis(max.1 + delta, sy).max(min.1),
            EntityField::PortalBz1 => max.2 = clamp_axis(max.2 + delta, sz).max(min.2),
            _ => unreachable!(),
        }
        set_portal_b_box(&mut state.level.portals[idx], min, max);
        state.mark_dirty();
        state.status = format!(
            "portal #{idx} region B ({},{},{})-({},{},{})",
            min.0, min.1, min.2, max.0, max.1, max.2
        );
        warn_if_portal_invalid(state, idx);
        return;
    }
    state.status = "portal: no such field".into();
}

/// AUTHORING GUARD: after a portal region edit, append a warning to the
/// status line when the portal would NOT survive `re_level::finalize` — i.e. a
/// region is not a flat wall face on a boundary, or the two openings' areas no
/// longer match. This surfaces the problem AS the author makes it, so a user can
/// never unknowingly save an unreloadable portal. (The default placement is valid;
/// this catches edits that break it.)
fn warn_if_portal_invalid(state: &mut EditorState, idx: usize) {
    let size = state.map.size();
    let Some(p) = state.level.portals.get(idx) else {
        return;
    };
    if !portal_is_reloadable(p, size) {
        state.status = format!(
            "{} — WARNING: portal region must be a flat wall face on a boundary, \
             both openings same area (won't reload until fixed)",
            state.status
        );
    }
}

/// `+`/`-` on a Note focused field: pos x/y/z (clamped to the room). Kind/facing
/// are enum (`c`); text is typed. A no-op (with a hint) when no note is selected.
fn adjust_note_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if matches!(
        field,
        EntityField::NoteKindField | EntityField::NoteFacing | EntityField::NoteText
    ) {
        state.status = format!("'{}' is an enum/typed field", field.label());
        return;
    }
    let (sx, sy, sz) = state.map.size();
    let Some(idx) = state.selected_note.filter(|&i| i < state.level.notes.len()) else {
        state.status = "no note selected".into();
        return;
    };
    let n = &mut state.level.notes[idx];
    let msg = match field {
        EntityField::NotePosX => {
            n.x = clamp_axis(n.x + delta, sx);
            format!("note #{idx} x {}", n.x)
        }
        EntityField::NotePosY => {
            n.y = clamp_axis(n.y + delta, sy);
            format!("note #{idx} y {}", n.y)
        }
        EntityField::NotePosZ => {
            n.z = clamp_axis(n.z + delta, sz);
            format!("note #{idx} z {}", n.z)
        }
        _ => "note: no such field".to_string(),
    };
    state.mark_dirty();
    state.status = msg;
}

/// Mark dirty + set a `<who> @ x,y,z` status after a door/singleton pos nudge.
fn mark_pos(state: &mut EditorState, who: &str) {
    state.mark_dirty();
    let (x, y, z) = (state.level.door.x, state.level.door.y, state.level.door.z);
    state.status = format!("{who} @ {x},{y},{z}");
}

/// Nudge the FOCUSED emitter NUMERIC field by `delta` (±1) with a per-field step
/// and clamp, then re-`validate` so the spec stays render-safe. A no-op (with a
/// hint) when the focused field is an enum (use `c`) or no emitter is selected.
fn adjust_emitter_numeric(state: &mut EditorState, delta: i32) {
    let field = state.emitter_field;
    let Some(idx) = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
    else {
        state.status = "no emitter selected".into();
        return;
    };
    if !field.is_numeric() {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    let d = delta as f32;
    let e = &mut state.level.emitters[idx];
    let msg = match field {
        // rate: 1/sec step, never negative.
        EmitterField::Rate => {
            e.rate = (e.rate + d).max(0.0);
            format!("rate {:.1}", e.rate)
        }
        // burst: integer step. Stepping at/below 0 clears to None (continuous).
        EmitterField::Burst => {
            let cur = e.burst.unwrap_or(0) as i64;
            let next = cur + delta as i64;
            e.burst = if next <= 0 { None } else { Some(next as u32) };
            match e.burst {
                Some(n) => format!("burst {n}"),
                None => "burst none".to_string(),
            }
        }
        // max_particles: 16-step, clamped to >= 1.
        EmitterField::MaxParticles => {
            let next = e.max_particles as i64 + delta as i64 * 16;
            e.max_particles = next.max(1) as u32;
            format!("max_particles {}", e.max_particles)
        }
        // speed: 0.1 cell/sec step, never negative.
        EmitterField::Speed => {
            e.speed = (e.speed + d * 0.1).max(0.0);
            format!("speed {:.2}", e.speed)
        }
        // speed_spread: 0.05 step, never negative.
        EmitterField::SpeedSpread => {
            e.speed_spread = (e.speed_spread + d * 0.05).max(0.0);
            format!("speed_spread {:.2}", e.speed_spread)
        }
        // spread: 5° step, clamped to the spec's legal [0,180].
        EmitterField::SpreadDeg => {
            e.spread_deg = (e.spread_deg + d * 5.0).clamp(0.0, 180.0);
            format!("spread {:.0}", e.spread_deg)
        }
        // lifetime: 0.25 s step, kept >= 0.01 (validate enforces, clamp pre-empts).
        EmitterField::Lifetime => {
            e.lifetime = (e.lifetime + d * 0.25).max(0.01);
            format!("life {:.2}", e.lifetime)
        }
        // lifetime_spread: 0.1 s step, never negative (validate caps the upper end).
        EmitterField::LifetimeSpread => {
            e.lifetime_spread = (e.lifetime_spread + d * 0.1).max(0.0);
            format!("life_spread {:.2}", e.lifetime_spread)
        }
        // size: 0.1 step, kept positive.
        EmitterField::Size => {
            e.size = (e.size + d * 0.1).max(f32::EPSILON);
            format!("size {:.2}", e.size)
        }
        // growth: 0.1 step, kept positive.
        EmitterField::Growth => {
            e.growth = (e.growth + d * 0.1).max(f32::EPSILON);
            format!("growth {:.2}", e.growth)
        }
        // accel x/y/z: 0.05 cells/s² step (signed — gravity/buoyancy/drift).
        EmitterField::AccelX => {
            e.accel.x += d * 0.05;
            format!("accelX {:.2}", e.accel.x)
        }
        EmitterField::AccelY => {
            e.accel.y += d * 0.05;
            format!("accelY {:.2}", e.accel.y)
        }
        EmitterField::AccelZ => {
            e.accel.z += d * 0.05;
            format!("accelZ {:.2}", e.accel.z)
        }
        // swirl: 0.05 step (signed — CCW positive, CW negative).
        EmitterField::Swirl => {
            e.swirl += d * 0.05;
            format!("swirl {:.2}", e.swirl)
        }
        // path_jitter: 0.05 step, never negative.
        EmitterField::PathJitter => {
            e.path_jitter = (e.path_jitter + d * 0.05).max(0.0);
            format!("path_jitter {:.2}", e.path_jitter)
        }
        // colour stops: step the shared palette.
        EmitterField::Color0 => {
            e.colors[0] = step_stop(e.colors[0], delta);
            format!("color0 #{:06X}", e.colors[0] & 0xFF_FFFF)
        }
        EmitterField::Color1 => {
            e.colors[1] = step_stop(e.colors[1], delta);
            format!("color1 #{:06X}", e.colors[1] & 0xFF_FFFF)
        }
        EmitterField::Color2 => {
            e.colors[2] = step_stop(e.colors[2], delta);
            format!("color2 #{:06X}", e.colors[2] & 0xFF_FFFF)
        }
        // seed: ±1 step (deterministic RNG seed). Wraps within u64.
        EmitterField::Seed => {
            e.seed = e.seed.wrapping_add_signed(delta as i64);
            format!("seed {}", e.seed)
        }
        // origin fine-tune: 0.1-cell nudge per axis (World origins only).
        EmitterField::OriginX => {
            nudge_origin(e, 0, d * 0.1);
            let (x, _, _) = origin_xyz(e);
            format!("originX {x:.1}")
        }
        EmitterField::OriginY => {
            nudge_origin(e, 1, d * 0.1);
            let (_, y, _) = origin_xyz(e);
            format!("originY {y:.1}")
        }
        EmitterField::OriginZ => {
            nudge_origin(e, 2, d * 0.1);
            let (_, _, z) = origin_xyz(e);
            format!("originZ {z:.1}")
        }
        _ => unreachable!("non-numeric field guarded above"),
    };
    e.validate();
    state.mark_dirty();
    state.status = msg;
}

/// Short label for a node's io role.
pub fn io_role_label(is_input: bool, is_output: bool) -> &'static str {
    match (is_input, is_output) {
        (true, _) => "input",
        (false, true) => "output",
        _ => "neither",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The swatch list the colour steppers share, for the palette assertions.
    use crate::material::PALETTE as STOP_PALETTE;
    use crate::state::EntityKind;

    fn at(state: &mut EditorState, x: usize, y: usize, z: usize) {
        state.cursor = (x, y);
        state.z = z;
    }

    #[test]
    fn selector_cycles_through_all_kinds() {
        let mut s = EditorState::new_empty(8, 8, 4);
        assert!(s.place.is_none());
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Door));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Spawn));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Pedestal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Part));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Node));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Emitter));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Trigger));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Pipe));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::ZoneInput));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Note));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::WallDecal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Holo));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Portal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Zone));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Line));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Inset));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Field));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Door), "wraps back to Door");
    }

    #[test]
    fn place_door_sets_position_and_dirty() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Door);
        at(&mut s, 3, 5, 2);
        place_at_cursor(&mut s);
        assert_eq!((s.level.door.x, s.level.door.y, s.level.door.z), (3, 5, 2));
        assert!(s.dirty);
    }

    #[test]
    fn place_spawn_sets_position_and_facing_cycles() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Spawn);
        at(&mut s, 2, 2, 1);
        place_at_cursor(&mut s);
        assert_eq!(
            (s.level.player.x, s.level.player.y, s.level.player.z),
            (2, 2, 1)
        );
        let f0 = s.level.player.facing_yaw_col;
        cycle_property(&mut s);
        assert_ne!(s.level.player.facing_yaw_col, f0, "facing changed");
        // Four steps return to the start.
        cycle_property(&mut s);
        cycle_property(&mut s);
        cycle_property(&mut s);
        assert_eq!(s.level.player.facing_yaw_col, f0, "facing wraps after 4");
    }

    #[test]
    fn place_pedestal_gets_default_volume_then_grows() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Pedestal);
        at(&mut s, 4, 4, 0);
        place_at_cursor(&mut s);
        assert_eq!(
            (
                s.level.pedestal.cx,
                s.level.pedestal.cy,
                s.level.pedestal.cz
            ),
            (4, 4, 0)
        );
        assert_eq!(s.level.pedestal.w, 1, "default volume");
        // Per-axis: focus W and grow it.
        s.entity_field = EntityField::PedVolW;
        adjust_property(&mut s, 1);
        assert_eq!(s.level.pedestal.w, 2, "grow w");
        adjust_property(&mut s, -1);
        adjust_property(&mut s, -1); // clamps at 1
        assert_eq!(s.level.pedestal.w, 1);
    }

    #[test]
    fn place_node_pushes_unique_then_reselects_not_duplicates() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 1);
        assert_eq!(s.selected_node, Some(0));
        let name0 = s.level.nodes[0].name.clone();

        // A second cell → a second, uniquely-named node.
        at(&mut s, 2, 1, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 2);
        assert_ne!(s.level.nodes[1].name, name0, "unique auto-id");

        // Re-placing on the FIRST node selects it, no duplicate.
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 2, "no duplicate on re-place");
        assert_eq!(s.selected_node, Some(0));
    }

    #[test]
    fn node_io_cycle_and_threshold() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        // io role is the focused-field NodeIo: neither → input → output → neither
        s.entity_field = EntityField::NodeIo;
        assert!(!s.level.nodes[0].is_input && !s.level.nodes[0].is_output);
        cycle_property(&mut s);
        assert!(s.level.nodes[0].is_input && !s.level.nodes[0].is_output);
        cycle_property(&mut s);
        assert!(!s.level.nodes[0].is_input && s.level.nodes[0].is_output);
        cycle_property(&mut s);
        assert!(!s.level.nodes[0].is_input && !s.level.nodes[0].is_output);
        // Threshold up/down (focus the threshold field).
        s.entity_field = EntityField::NodeThreshold;
        adjust_property(&mut s, 1);
        assert_eq!(s.level.nodes[0].threshold, 1);
        adjust_property(&mut s, -2);
        assert_eq!(s.level.nodes[0].threshold, -1);
    }

    #[test]
    fn delete_door_pedestal_and_node() {
        let mut s = EditorState::new_empty(8, 8, 4);
        // Door.
        s.place = Some(EntityKind::Door);
        at(&mut s, 3, 3, 1);
        place_at_cursor(&mut s);
        delete_at_cursor(&mut s);
        assert_eq!((s.level.door.x, s.level.door.y, s.level.door.z), (0, 0, 0));

        // Pedestal.
        s.place = Some(EntityKind::Pedestal);
        at(&mut s, 4, 4, 0);
        place_at_cursor(&mut s);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.pedestal.w, 0, "pedestal cleared to default");

        // Node.
        s.place = Some(EntityKind::Node);
        at(&mut s, 2, 2, 2);
        place_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 1);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 0);
        assert!(s.selected_node.is_none(), "selection dropped on delete");
    }

    #[test]
    fn delete_with_no_match_is_a_defined_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Node);
        at(&mut s, 5, 5, 0);
        // Nothing here — must not panic, must not mark dirty.
        delete_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 0);
        assert!(!s.dirty);
    }

    /// B2: deleting a node BELOW a door win-target via the editor `x` delete must
    /// keep the door pointed at the SAME logical node (its index shifts down), not
    /// silently re-point the win condition at the wrong node. Exercises the
    /// `entity::delete_at_cursor` → `Level::remove_node` wiring.
    #[test]
    fn delete_node_below_door_target_keeps_win_target() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(EntityKind::Node);
        // Place n0,n1,n2 at distinct cells.
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 2, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 3, 1, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.nodes.len(), 3);
        let n2_name = s.level.nodes[2].name.clone();
        // Door wins on n2 (index 2).
        s.level.door.target_parts = vec![2];

        // Delete n1 (index 1, below the target) by pointing the cursor at it.
        at(&mut s, 2, 1, 0);
        delete_at_cursor(&mut s);

        assert_eq!(s.level.nodes.len(), 2);
        let new_n2 = s.level.node_index(&n2_name).expect("n2 survives");
        assert_eq!(new_n2, 1, "n2 shifted to index 1");
        assert_eq!(
            s.level.door.target_parts,
            vec![new_n2],
            "door win-target still points at n2 after the delete"
        );
    }

    #[test]
    fn place_is_noop_when_selector_off() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = None;
        place_at_cursor(&mut s);
        delete_at_cursor(&mut s);
        assert!(!s.dirty);
        assert_eq!(s.level.nodes.len(), 0);
    }

    // ── EMITTER place / select / delete ──────────────────────────────────────

    #[test]
    fn selector_now_includes_emitter() {
        // Door→Spawn→Pedestal→Part→Node→Emitter→Door.
        let mut s = EditorState::new_empty(8, 8, 4);
        for _ in 0..5 {
            cycle_selector(&mut s);
        }
        assert_eq!(s.place, Some(EntityKind::Node));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Emitter));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Trigger));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Pipe));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::ZoneInput));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Note));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::WallDecal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Holo));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Portal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Zone));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Line));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Inset));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Field));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Door), "wraps back to Door");
    }

    #[test]
    fn place_emitter_anchors_at_cell_centre_and_selects() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 3, 5, 2);
        place_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 1);
        assert_eq!(s.selected_emitter, Some(0));
        assert!(s.dirty);
        // Origin sits at the cell CENTRE (x+0.5,y+0.5,z+0.5).
        match s.level.emitters[0].origin {
            EmitterOrigin::World(p) => {
                assert!((p.x - 3.5).abs() < 1e-6);
                assert!((p.y - 5.5).abs() < 1e-6);
                assert!((p.z - 2.5).abs() < 1e-6);
            }
            _ => panic!("emitter origin must be World"),
        }
        // The default spec is render-valid.
        assert!(s.level.emitters[0].is_valid());
    }

    #[test]
    fn replace_on_existing_emitter_selects_not_duplicates() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        // A different cell → a second emitter.
        at(&mut s, 4, 4, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 2);
        s.selected_emitter = None;
        // Re-placing on the FIRST cell selects index 0, no duplicate.
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 2, "no duplicate on re-place");
        assert_eq!(s.selected_emitter, Some(0));
    }

    #[test]
    fn delete_emitter_removes_and_fixes_selection() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 2, 1, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 2);
        s.selected_emitter = Some(1);
        // Delete index 0 → the survivor shifts to 0; the selection follows.
        at(&mut s, 1, 1, 0);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 1);
        assert_eq!(s.selected_emitter, Some(0));
    }

    #[test]
    fn delete_emitter_with_no_match_is_a_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 5, 5, 0);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.emitters.len(), 0);
        assert!(!s.dirty);
    }

    // ── EMITTER focused-field editing ────────────────────────────────────────

    #[test]
    fn focus_cycle_wraps_and_only_when_emitter_selected() {
        use crate::state::EMITTER_FIELDS;
        let mut s = EditorState::new_empty(8, 8, 4);
        // No Emitter kind active → no-op (focus unchanged).
        let f0 = s.emitter_field;
        cycle_emitter_field(&mut s, 1);
        assert_eq!(s.emitter_field, f0, "no-op when not in Emitter mode");

        s.place = Some(EntityKind::Emitter);
        assert_eq!(s.emitter_field, EMITTER_FIELDS[0]);
        cycle_emitter_field(&mut s, 1);
        assert_eq!(s.emitter_field, EMITTER_FIELDS[1]);
        // Step backward past 0 wraps to the last field.
        cycle_emitter_field(&mut s, -1);
        cycle_emitter_field(&mut s, -1);
        assert_eq!(s.emitter_field, EMITTER_FIELDS[EMITTER_FIELDS.len() - 1]);
    }

    #[test]
    fn adjust_numeric_fields_step_and_clamp() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);

        // rate: +1 step, clamps at 0.
        s.emitter_field = EmitterField::Rate;
        let r0 = s.level.emitters[0].rate;
        adjust_property(&mut s, 1);
        assert!((s.level.emitters[0].rate - (r0 + 1.0)).abs() < 1e-4);
        for _ in 0..1000 {
            adjust_property(&mut s, -1);
        }
        assert_eq!(s.level.emitters[0].rate, 0.0, "rate clamps at 0");

        // spread: 5° step, clamps to [0,180].
        s.emitter_field = EmitterField::SpreadDeg;
        for _ in 0..100 {
            adjust_property(&mut s, 1);
        }
        assert_eq!(
            s.level.emitters[0].spread_deg, 180.0,
            "spread clamps at 180"
        );
        for _ in 0..100 {
            adjust_property(&mut s, -1);
        }
        assert_eq!(s.level.emitters[0].spread_deg, 0.0, "spread clamps at 0");

        // lifetime stays render-valid (>= 0.01) under heavy decrement.
        s.emitter_field = EmitterField::Lifetime;
        for _ in 0..1000 {
            adjust_property(&mut s, -1);
        }
        assert!(s.level.emitters[0].lifetime >= 0.01);
        assert!(s.level.emitters[0].is_valid());
    }

    #[test]
    fn adjust_on_enum_field_is_a_hinted_noop() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.emitter_field = EmitterField::ColorMode;
        let before = s.level.emitters[0].clone();
        s.dirty = false;
        adjust_property(&mut s, 1);
        assert_eq!(s.level.emitters[0], before, "enum field unchanged by +/-");
        assert!(!s.dirty);
    }

    #[test]
    fn cycle_enum_fields_advance() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);

        // color_mode: RampOverAge → RandomPick.
        s.emitter_field = EmitterField::ColorMode;
        assert_eq!(s.level.emitters[0].color_mode, ColorMode::RampOverAge);
        cycle_property(&mut s);
        assert_eq!(s.level.emitters[0].color_mode, ColorMode::RandomPick);

        // family: advances to the next canonical name.
        s.emitter_field = EmitterField::Family;
        let f0 = s.level.emitters[0].family.clone();
        cycle_property(&mut s);
        assert_ne!(s.level.emitters[0].family, f0, "family cycles");
        assert!(crate::vocab::emitter_families().contains(&s.level.emitters[0].family));

        // bold: Off → On.
        s.emitter_field = EmitterField::Bold;
        assert_eq!(s.level.emitters[0].bold, BoldMode::Off);
        cycle_property(&mut s);
        assert_eq!(s.level.emitters[0].bold, BoldMode::On);

        // direction axis: +Z preset → -Z.
        s.emitter_field = EmitterField::DirectionAxis;
        cycle_property(&mut s);
        assert!(
            (s.level.emitters[0].direction.z + 1.0).abs() < 1e-3,
            "dir flips to -Z"
        );
    }

    #[test]
    fn cycle_enum_on_numeric_field_is_a_hinted_noop() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.emitter_field = EmitterField::Rate;
        let before = s.level.emitters[0].clone();
        s.dirty = false;
        cycle_property(&mut s);
        assert_eq!(s.level.emitters[0], before, "numeric field unchanged by c");
        assert!(!s.dirty);
    }

    // ── EXHAUSTIVE emitter field coverage ────────────────────────────────────

    /// Every field in `EMITTER_FIELDS` is reachable by `;` stepping from the first,
    /// and each produces a non-empty focus value readout. Guards against a field
    /// being declared but unreachable / unrendered.
    #[test]
    fn every_emitter_field_is_reachable_and_has_a_value() {
        use crate::state::EMITTER_FIELDS;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        // Step through the whole cycle, visiting each field exactly once.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..EMITTER_FIELDS.len() {
            seen.insert(s.emitter_field);
            let v = emitter_field_value(&s);
            assert!(
                !v.is_empty() && v != "-",
                "field {:?} has a value",
                s.emitter_field
            );
            cycle_emitter_field(&mut s, 1);
        }
        assert_eq!(seen.len(), EMITTER_FIELDS.len(), "all fields visited once");
        // One more step wraps back to the first field.
        assert_eq!(s.emitter_field, EMITTER_FIELDS[0], "cycle wraps");
    }

    /// The newly-exposed scalar fields step and clamp, and the spec stays valid.
    #[test]
    fn new_numeric_fields_step_and_stay_valid() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);

        // speed_spread: 0.05 step, clamps at 0.
        s.emitter_field = EmitterField::SpeedSpread;
        for _ in 0..1000 {
            adjust_property(&mut s, -1);
        }
        assert_eq!(s.level.emitters[0].speed_spread, 0.0);

        // path_jitter: positive step survives.
        s.emitter_field = EmitterField::PathJitter;
        adjust_property(&mut s, 1);
        assert!(s.level.emitters[0].path_jitter > 0.0);

        // max_particles: 16-step, never below 1.
        s.emitter_field = EmitterField::MaxParticles;
        for _ in 0..1000 {
            adjust_property(&mut s, -1);
        }
        assert_eq!(s.level.emitters[0].max_particles, 1);
        adjust_property(&mut s, 1);
        assert_eq!(s.level.emitters[0].max_particles, 17);

        // accel z and swirl are signed.
        s.emitter_field = EmitterField::AccelZ;
        let z0 = s.level.emitters[0].accel.z;
        adjust_property(&mut s, -1);
        assert!(s.level.emitters[0].accel.z < z0);
        s.emitter_field = EmitterField::Swirl;
        adjust_property(&mut s, 1);
        assert!(s.level.emitters[0].swirl > 0.0);

        // seed steps by 1.
        s.emitter_field = EmitterField::Seed;
        let seed0 = s.level.emitters[0].seed;
        adjust_property(&mut s, 1);
        assert_eq!(s.level.emitters[0].seed, seed0.wrapping_add(1));

        assert!(
            s.level.emitters[0].is_valid(),
            "spec stays valid after edits"
        );
    }

    /// Burst toggles None ⇄ Some via `+`/`-`.
    #[test]
    fn burst_field_toggles_none_and_count() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.emitter_field = EmitterField::Burst;
        assert_eq!(s.level.emitters[0].burst, None);
        adjust_property(&mut s, 1);
        assert_eq!(s.level.emitters[0].burst, Some(1));
        adjust_property(&mut s, 5);
        assert_eq!(s.level.emitters[0].burst, Some(6));
        // Stepping back to/below 0 clears it.
        adjust_property(&mut s, -10);
        assert_eq!(s.level.emitters[0].burst, None);
    }

    /// The three colour stops step through the palette independently.
    #[test]
    fn colour_stops_step_palette_independently() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.emitter_field = EmitterField::Color1;
        let c0 = s.level.emitters[0].colors;
        adjust_property(&mut s, 1);
        let c1 = s.level.emitters[0].colors;
        assert_eq!(c1[0], c0[0], "stop 0 untouched");
        assert_eq!(c1[2], c0[2], "stop 2 untouched");
        assert!(
            STOP_PALETTE.contains(&(c1[1] & 0xFF_FFFF)),
            "stop 1 is a palette swatch"
        );
    }

    /// glyph_set and blend are enum-cycled by `c`.
    #[test]
    fn glyph_set_and_blend_cycle_via_c() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);

        s.emitter_field = EmitterField::GlyphSet;
        assert!(s.level.emitters[0].glyph_set.is_none());
        cycle_property(&mut s);
        assert!(
            s.level.emitters[0].glyph_set.is_some(),
            "glyph_set set by c"
        );

        s.emitter_field = EmitterField::Blend;
        assert_eq!(s.level.emitters[0].blend, EmitterBlend::KeepBg);
        cycle_property(&mut s);
        assert_eq!(s.level.emitters[0].blend, EmitterBlend::Opaque);
    }

    // ── EXHAUSTIVE non-emitter entity focused-field editing ──────────────────

    use crate::state::{entity_fields_for, EntityField};

    /// Every focused field of every non-emitter kind is reachable by `;` stepping
    /// from the kind's first field, and each renders a non-empty value (with the
    /// entity placed/selected). Guards against a declared-but-unreachable field.
    #[test]
    fn every_entity_field_is_reachable_and_has_a_value() {
        for &kind in &[
            EntityKind::Door,
            EntityKind::Node,
            EntityKind::Pedestal,
            EntityKind::Spawn,
        ] {
            let mut s = EditorState::new_empty(10, 10, 4);
            // Place the entity so a value exists (Node needs a node + a target node).
            s.place = Some(EntityKind::Node);
            at(&mut s, 1, 1, 0);
            place_at_cursor(&mut s); // n0 (door targets need a node to point at)
            s.place = Some(kind);
            at(&mut s, 2, 2, 1);
            place_at_cursor(&mut s);
            // Reset focus to the kind's first field.
            s.entity_field = entity_fields_for(kind)[0];

            let list = entity_fields_for(kind);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..list.len() {
                seen.insert(s.entity_field);
                let v = entity_field_value(&s);
                assert!(
                    !v.is_empty(),
                    "{kind:?} field {:?} has a value",
                    s.entity_field
                );
                cycle_entity_field(&mut s, 1);
            }
            assert_eq!(seen.len(), list.len(), "{kind:?}: all fields visited");
            assert_eq!(s.entity_field, list[0], "{kind:?}: focus cycle wraps");
        }
    }

    /// Door target ON toggling adds then removes the cursor node from
    /// `target_parts`, and a node added to OFF is dropped from ON (disjoint).
    #[test]
    fn door_target_on_off_toggle_add_remove_and_disjoint() {
        let mut s = EditorState::new_empty(8, 8, 4);
        // Two nodes.
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 2, 1, 0);
        place_at_cursor(&mut s);

        s.place = Some(EntityKind::Door);
        at(&mut s, 4, 4, 0);
        place_at_cursor(&mut s);

        // Focus the ON target, cursor on node 0 → add.
        s.entity_field = EntityField::DoorTargetOn;
        s.door_target_cursor = 0;
        cycle_property(&mut s);
        assert_eq!(s.level.door.target_parts, vec![0], "node0 added to ON");
        // Toggle again → remove.
        cycle_property(&mut s);
        assert!(
            s.level.door.target_parts.is_empty(),
            "node0 removed from ON"
        );

        // Add node0 to ON, then add it to OFF: it should leave ON (disjoint).
        cycle_property(&mut s); // ON has [0]
        s.entity_field = EntityField::DoorTargetOff;
        cycle_property(&mut s); // add 0 to OFF, drop from ON
        assert!(
            s.level.door.target_parts.is_empty(),
            "node0 left ON when added to OFF"
        );
        assert_eq!(s.level.door.target_off, vec![0], "node0 now an OFF target");
    }

    /// The door target-cursor moves through the node list with `+`/`-`, wrapping.
    #[test]
    fn door_target_cursor_steps_through_nodes() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Node);
        for x in 1..=3 {
            at(&mut s, x, 1, 0);
            place_at_cursor(&mut s);
        }
        s.place = Some(EntityKind::Door);
        at(&mut s, 5, 5, 0);
        place_at_cursor(&mut s);
        s.entity_field = EntityField::DoorTargetOn;
        s.door_target_cursor = 0;
        adjust_property(&mut s, 1);
        assert_eq!(s.door_target_cursor, 1);
        adjust_property(&mut s, 1);
        assert_eq!(s.door_target_cursor, 2);
        adjust_property(&mut s, 1); // wraps (3 nodes)
        assert_eq!(s.door_target_cursor, 0);
        adjust_property(&mut s, -1); // wraps back
        assert_eq!(s.door_target_cursor, 2);
    }

    /// Door target editing with NO nodes is a defined no-op (no panic, no dirty).
    #[test]
    fn door_target_with_no_nodes_is_a_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Door);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        s.dirty = false;
        s.entity_field = EntityField::DoorTargetOn;
        cycle_property(&mut s);
        assert!(s.level.door.target_parts.is_empty());
        assert!(!s.dirty, "no nodes → no toggle → not dirty");
    }

    /// Node glyph + color cycle, io role cycles, threshold/anchor step (anchor
    /// clamps at 0), and pos clamps to the room.
    #[test]
    fn node_exhaustive_field_edits() {
        let mut s = EditorState::new_empty(6, 6, 3);
        s.place = Some(EntityKind::Node);
        at(&mut s, 2, 2, 1);
        place_at_cursor(&mut s);
        s.selected_node = Some(0);

        // glyph cycles off 'N'.
        s.entity_field = EntityField::NodeGlyph;
        let g0 = s.level.nodes[0].glyph;
        cycle_property(&mut s);
        assert_ne!(s.level.nodes[0].glyph, g0, "glyph cycles");

        // color steps the palette.
        s.entity_field = EntityField::NodeColor;
        let c0 = s.level.nodes[0].color_rgb;
        adjust_property(&mut s, 1);
        assert_ne!(s.level.nodes[0].color_rgb, c0, "color steps");

        // anchor clamps at 0 on heavy decrement.
        s.entity_field = EntityField::NodeAnchor;
        adjust_property(&mut s, 3);
        assert_eq!(s.level.nodes[0].anchor, 3);
        adjust_property(&mut s, -100);
        assert_eq!(s.level.nodes[0].anchor, 0, "anchor clamps at 0");

        // pos clamps to the room (x in 0..=5).
        s.entity_field = EntityField::NodePosX;
        for _ in 0..100 {
            adjust_property(&mut s, 1);
        }
        assert_eq!(s.level.nodes[0].x, 5, "node x clamps to room");
        for _ in 0..100 {
            adjust_property(&mut s, -1);
        }
        assert_eq!(s.level.nodes[0].x, 0, "node x clamps at 0");

        // pool grows then shrinks.
        s.entity_field = EntityField::NodePool;
        adjust_property(&mut s, 1);
        adjust_property(&mut s, 1);
        assert_eq!(s.level.nodes[0].pool.len(), 2, "pool grew by 2");
        adjust_property(&mut s, -1);
        assert_eq!(s.level.nodes[0].pool.len(), 1, "pool shrank by 1");
    }

    /// Pedestal per-axis volume is INDEPENDENT (w/h/d differ), each clamps
    /// 1..=max, and the center clamps to the room.
    #[test]
    fn pedestal_per_axis_volume_is_independent() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Pedestal);
        at(&mut s, 4, 4, 1);
        place_at_cursor(&mut s);
        // Grow only W.
        s.entity_field = EntityField::PedVolW;
        adjust_property(&mut s, 3);
        assert_eq!(s.level.pedestal.w, 4);
        assert_eq!(s.level.pedestal.h, 1, "h unchanged");
        assert_eq!(s.level.pedestal.d, 1, "d unchanged");
        // Each axis clamps at the max independently.
        for _ in 0..50 {
            adjust_property(&mut s, 1);
        }
        assert_eq!(s.level.pedestal.w, PEDESTAL_MAX_VOL, "w clamps at max");
        // Center clamps to room.
        s.entity_field = EntityField::PedCenterX;
        for _ in 0..100 {
            adjust_property(&mut s, 1);
        }
        assert_eq!(s.level.pedestal.cx, 7, "center x clamps to room (8 wide)");
    }

    /// Spawn pos steps + clamps; facing cycles via `c`.
    #[test]
    fn spawn_pos_and_facing_edits() {
        let mut s = EditorState::new_empty(6, 6, 3);
        s.place = Some(EntityKind::Spawn);
        at(&mut s, 2, 2, 1);
        place_at_cursor(&mut s);
        s.entity_field = EntityField::SpawnPosY;
        adjust_property(&mut s, 1);
        assert_eq!(s.level.player.y, 3);
        // Facing via c.
        s.entity_field = EntityField::SpawnFacing;
        let f0 = s.level.player.facing_yaw_col;
        cycle_property(&mut s);
        assert_ne!(s.level.player.facing_yaw_col, f0, "facing cycles");
    }

    /// Switching the object-selector kind resets the focused field to that kind's
    /// first field (no stale focus leaks across kinds).
    #[test]
    fn switching_kind_resets_focus() {
        let mut s = EditorState::new_empty(8, 8, 4);
        cycle_selector(&mut s); // Door
        assert_eq!(s.entity_field, entity_fields_for(EntityKind::Door)[0]);
        cycle_selector(&mut s); // Spawn
        assert_eq!(s.entity_field, entity_fields_for(EntityKind::Spawn)[0]);
        cycle_selector(&mut s); // Pedestal
        assert_eq!(s.entity_field, entity_fields_for(EntityKind::Pedestal)[0]);
        cycle_selector(&mut s); // Part
        assert_eq!(s.entity_field, entity_fields_for(EntityKind::Part)[0]);
        cycle_selector(&mut s); // Node
        assert_eq!(s.entity_field, entity_fields_for(EntityKind::Node)[0]);
    }

    // ── TRIGGER place / select / delete / edit ───────────────────────────────

    #[test]
    fn selector_includes_trigger_and_wraps() {
        let mut s = EditorState::new_empty(8, 8, 4);
        for _ in 0..6 {
            cycle_selector(&mut s);
        }
        assert_eq!(s.place, Some(EntityKind::Emitter));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Trigger));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Pipe));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::ZoneInput));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Note));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::WallDecal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Holo));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Portal));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Zone));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Line));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Inset));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Field));
        cycle_selector(&mut s);
        assert_eq!(s.place, Some(EntityKind::Door), "wraps back to Door");
    }

    #[test]
    fn place_trigger_makes_unit_box_and_selects() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 3, 4, 1);
        place_at_cursor(&mut s);
        assert_eq!(s.level.triggers.len(), 1);
        assert_eq!(s.selected_trigger, Some(0));
        assert!(s.dirty);
        // Default 1x1x1 box at the cursor cell.
        let (min, max) = trigger_box(&s.level.triggers[0]);
        assert_eq!(min, (3, 4, 1));
        assert_eq!(max, (3, 4, 1));
        // No nodes ⇒ default action is a generic Event.
        assert!(matches!(
            s.level.triggers[0].action,
            re_core::TriggerAction::Event { .. }
        ));
        assert!(s.level.triggers[0].once, "default once = true");
    }

    #[test]
    fn place_trigger_with_a_node_defaults_to_activate_node() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        let name = s.level.nodes[0].name.clone();
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 5, 5, 1);
        place_at_cursor(&mut s);
        match &s.level.triggers[0].action {
            re_core::TriggerAction::ActivateNode {
                name: n,
                node_index,
            } => {
                assert_eq!(n, &name, "defaults to the first node's name");
                assert_eq!(*node_index, 0, "resolved to node 0");
            }
            other => panic!("expected ActivateNode, got {other:?}"),
        }
    }

    #[test]
    fn replace_on_existing_trigger_region_selects_not_duplicates() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        // Grow the box so it covers (3,3,0) too.
        s.entity_field = EntityField::TrigMaxX;
        adjust_property(&mut s, 2);
        s.entity_field = EntityField::TrigMaxY;
        adjust_property(&mut s, 2);
        s.selected_trigger = None;
        // Placing on a covered cell selects the existing trigger, no duplicate.
        at(&mut s, 3, 3, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.triggers.len(), 1, "no duplicate on re-place");
        assert_eq!(s.selected_trigger, Some(0));
    }

    #[test]
    fn delete_trigger_removes_and_fixes_selection() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 4, 4, 0);
        place_at_cursor(&mut s);
        assert_eq!(s.level.triggers.len(), 2);
        s.selected_trigger = Some(1);
        // Delete index 0 → survivor shifts to 0; the selection follows.
        at(&mut s, 1, 1, 0);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.triggers.len(), 1);
        assert_eq!(s.selected_trigger, Some(0));
    }

    #[test]
    fn delete_trigger_with_no_match_is_a_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 6, 6, 0);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.triggers.len(), 0);
        assert!(!s.dirty);
    }

    #[test]
    fn trigger_region_corners_clamp_to_room_and_keep_min_le_max() {
        let mut s = EditorState::new_empty(6, 6, 3);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 2, 2, 1);
        place_at_cursor(&mut s);

        // Grow max X to the room edge (clamps at sx-1 = 5).
        s.entity_field = EntityField::TrigMaxX;
        for _ in 0..100 {
            adjust_property(&mut s, 1);
        }
        let (_min, max) = trigger_box(&s.level.triggers[0]);
        assert_eq!(max.0, 5, "max X clamps to the room");

        // Pushing min X past max X is held at max (min <= max invariant).
        s.entity_field = EntityField::TrigMinX;
        for _ in 0..100 {
            adjust_property(&mut s, 1);
        }
        let (min, max) = trigger_box(&s.level.triggers[0]);
        assert!(min.0 <= max.0, "min X never exceeds max X");
        assert_eq!(min.0, max.0);

        // Pushing max below min is held at min.
        s.entity_field = EntityField::TrigMaxY;
        for _ in 0..100 {
            adjust_property(&mut s, -1);
        }
        let (min, max) = trigger_box(&s.level.triggers[0]);
        assert!(max.1 >= min.1, "max Y never drops below min Y");
    }

    #[test]
    fn trigger_action_type_toggles_and_target_cycles() {
        let mut s = EditorState::new_empty(8, 8, 4);
        // Two nodes so ActivateNode has targets to cycle.
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        at(&mut s, 2, 1, 0);
        place_at_cursor(&mut s);
        let n0 = s.level.nodes[0].name.clone();
        let n1 = s.level.nodes[1].name.clone();

        s.place = Some(EntityKind::Trigger);
        at(&mut s, 5, 5, 1);
        place_at_cursor(&mut s);
        // Defaults to ActivateNode(n0).
        s.entity_field = EntityField::TrigTarget;
        // Cycle target → n1.
        cycle_property(&mut s);
        match &s.level.triggers[0].action {
            re_core::TriggerAction::ActivateNode { name, node_index } => {
                assert_eq!(name, &n1);
                assert_eq!(*node_index, 1);
            }
            other => panic!("expected ActivateNode, got {other:?}"),
        }
        let _ = n0;

        // Toggle the action type → Event.
        s.entity_field = EntityField::TrigActionType;
        cycle_property(&mut s);
        assert!(matches!(
            s.level.triggers[0].action,
            re_core::TriggerAction::Event { .. }
        ));
        // Target now cycles the preset event labels.
        s.entity_field = EntityField::TrigTarget;
        let before = match &s.level.triggers[0].action {
            re_core::TriggerAction::Event { label } => label.clone(),
            _ => unreachable!(),
        };
        cycle_property(&mut s);
        let after = match &s.level.triggers[0].action {
            re_core::TriggerAction::Event { label } => label.clone(),
            _ => unreachable!(),
        };
        assert_ne!(before, after, "event label cycles");
        assert!(crate::state::TRIGGER_EVENT_LABELS.contains(&after.as_str()));

        // Toggle back to ActivateNode.
        s.entity_field = EntityField::TrigActionType;
        cycle_property(&mut s);
        assert!(matches!(
            s.level.triggers[0].action,
            re_core::TriggerAction::ActivateNode { .. }
        ));
    }

    #[test]
    fn trigger_once_toggles() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 3, 3, 0);
        place_at_cursor(&mut s);
        s.entity_field = EntityField::TrigOnce;
        assert!(s.level.triggers[0].once);
        cycle_property(&mut s);
        assert!(!s.level.triggers[0].once, "once toggled off");
        cycle_property(&mut s);
        assert!(s.level.triggers[0].once, "once toggled back on");
    }

    #[test]
    fn trigger_target_with_no_nodes_is_a_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s); // no nodes ⇒ Event action
                                 // Toggle to ActivateNode with no nodes: target cycling must not panic.
        s.entity_field = EntityField::TrigActionType;
        cycle_property(&mut s); // Event → ActivateNode (unresolved)
        s.entity_field = EntityField::TrigTarget;
        s.dirty = false;
        cycle_property(&mut s); // no nodes → defined no-op
        assert!(!s.dirty, "no nodes → no target change → not dirty");
    }

    /// Every trigger focused field is reachable by `;` stepping and renders a value.
    #[test]
    fn every_trigger_field_is_reachable_and_has_a_value() {
        use crate::state::entity_fields_for;
        let mut s = EditorState::new_empty(8, 8, 4);
        // A node so ActivateNode's target renders a name.
        s.place = Some(EntityKind::Node);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.place = Some(EntityKind::Trigger);
        at(&mut s, 4, 4, 1);
        place_at_cursor(&mut s);
        s.entity_field = entity_fields_for(EntityKind::Trigger)[0];
        let list = entity_fields_for(EntityKind::Trigger);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..list.len() {
            seen.insert(s.entity_field);
            let v = entity_field_value(&s);
            assert!(
                !v.is_empty() && v != "-",
                "field {:?} has a value",
                s.entity_field
            );
            cycle_entity_field(&mut s, 1);
        }
        assert_eq!(seen.len(), list.len(), "all trigger fields visited");
        assert_eq!(s.entity_field, list[0], "focus cycle wraps");
    }

    /// Origin nudges move a World origin by 0.1 cell per axis.
    #[test]
    fn origin_nudges_move_world_origin() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        at(&mut s, 3, 3, 1);
        place_at_cursor(&mut s);
        let (x0, _, _) = origin_xyz(&s.level.emitters[0]);
        s.emitter_field = EmitterField::OriginX;
        adjust_property(&mut s, 1);
        let (x1, _, _) = origin_xyz(&s.level.emitters[0]);
        assert!((x1 - (x0 + 0.1)).abs() < 1e-4, "originX nudged +0.1");
    }

    // ── text authoring (the caret-based text-entry router) ──────────────────

    /// Route one key through the FULL dispatcher (the acceptance path).
    fn type_key(s: &mut EditorState, code: re_present::EditorKeyCode) {
        crate::edit::apply_key(s, re_present::EditorKey { code, shift: false });
    }

    fn type_str(s: &mut EditorState, text: &str) {
        for c in text.chars() {
            type_key(s, re_present::EditorKeyCode::Char(c));
        }
    }

    /// Acceptance: place a note → type 3 lines with Enter → arrow into
    /// line 2 → insert a word mid-line → the body is exact and survives
    /// serialize + reparse identically.
    #[test]
    fn note_body_multiline_caret_editing_roundtrips() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.place = Some(EntityKind::Note);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        let idx = s.selected_note.expect("note placed");
        s.level.notes[idx].text.clear();
        s.entity_field = EntityField::NoteText;
        // Three lines typed with real Enters.
        type_str(&mut s, "one");
        type_key(&mut s, E::Enter);
        type_str(&mut s, "two");
        type_key(&mut s, E::Enter);
        type_str(&mut s, "three");
        assert_eq!(s.level.notes[idx].text, "one\ntwo\nthree");
        // Arrow up into line 2, walk to its start, insert a word.
        type_key(&mut s, E::Up);
        for _ in 0..3 {
            type_key(&mut s, E::Left);
        }
        type_str(&mut s, "B ");
        assert_eq!(
            s.level.notes[idx].text, "one\nB two\nthree",
            "mid-line insert landed"
        );
        // Round-trip: serializer + parser preserve the body. (The parser keeps
        // ONE trailing `\n` on a verbatim note body — pre-existing `re_level`
        // behaviour the writer's `trim_end` normalizes — so the CONTENT is
        // compared trimmed, and full-document stability is proven byte-for-byte
        // by re-serializing the reloaded level.)
        let text = crate::io::capture_and_serialize(&mut s);
        let reloaded = re_level::load_string(&text, "levels/_scratch/w23-note.lvl")
            .expect("serialized level reparses");
        assert_eq!(
            reloaded.notes[idx].text.trim_end(),
            "one\nB two\nthree",
            "body content identical after reload"
        );
        assert_eq!(
            re_voximport::to_lvl_text(&reloaded),
            text,
            "save → load → save is byte-identical (stable fixed point)"
        );
    }

    /// Acceptance: node pool as FREE TEXT — `alpha beta gamma` typed on
    /// the focused pool field lands as three words and saves verbatim as
    /// `pool = alpha beta gamma`.
    #[test]
    fn node_pool_free_text_words_roundtrip() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(1, 1, 0);
        s.place = Some(EntityKind::Node);
        s.selected_node = Some(0);
        s.entity_field = EntityField::NodePool;
        type_str(&mut s, "alpha beta gamma");
        assert_eq!(s.level.nodes[0].pool, vec!["alpha", "beta", "gamma"]);
        let text = crate::io::capture_and_serialize(&mut s);
        assert!(
            text.contains("pool = alpha beta gamma"),
            "pool saved verbatim, space-separated:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "levels/_scratch/w23-pool.lvl")
            .expect("serialized level reparses");
        assert_eq!(reloaded.nodes[0].pool, vec!["alpha", "beta", "gamma"]);
        // pool_alt is independently editable the same way.
        s.entity_field = EntityField::NodePoolAlt;
        type_str(&mut s, "x y");
        assert_eq!(s.level.nodes[0].pool_alt, vec!["x", "y"]);
    }

    /// Acceptance: trigger EVENT labels are free text (the 6-preset cycle
    /// is no longer a limit) and round-trip through `action = event LABEL`.
    #[test]
    fn trigger_event_label_free_text_roundtrips() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.triggers.push(re_core::Trigger {
            region: re_core::Region::Box {
                min: (1, 1, 0),
                max: (1, 1, 0),
            },
            action: re_core::TriggerAction::event("event"),
            once: true,
        });
        s.place = Some(EntityKind::Trigger);
        s.selected_trigger = Some(0);
        s.entity_field = EntityField::TrigTarget;
        // Clear the seeded "event" text, then type an arbitrary two-word label.
        for _ in 0..5 {
            type_key(&mut s, E::Backspace);
        }
        type_str(&mut s, "reactor breach");
        let re_core::TriggerAction::Event { label } = &s.level.triggers[0].action else {
            panic!("action stayed an Event");
        };
        assert_eq!(label, "reactor breach", "arbitrary text label");
        let text = crate::io::capture_and_serialize(&mut s);
        assert!(
            text.contains("action = event reactor breach"),
            "label serialized verbatim:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "levels/_scratch/w23-trig.lvl")
            .expect("serialized level reparses");
        let re_core::TriggerAction::Event { label } = &reloaded.triggers[0].action else {
            panic!("reloaded action is an Event");
        };
        assert_eq!(label, "reactor breach", "label identical after reload");
    }

    /// Guard: `#` is refused in single-line text fields (it would be
    /// silently truncated as a `.lvl` comment on the next load), and a
    /// non-Event trigger target captures NO text.
    #[test]
    fn text_router_refuses_hash_and_non_event_targets() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(1, 1, 0);
        s.place = Some(EntityKind::Node);
        s.selected_node = Some(0);
        s.entity_field = EntityField::NodePool;
        type_str(&mut s, "ok");
        type_key(&mut s, E::Char('#'));
        assert_eq!(
            s.level.nodes[0].pool,
            vec!["ok"],
            "# never entered the pool"
        );
        assert!(s.status.contains("not allowed"), "refusal is explained");
        // An ActivateNode trigger target is a selection cursor, not text.
        let mut t = EditorState::new_empty(8, 8, 2);
        t.level.push_node(1, 1, 0);
        t.level.triggers.push(re_core::Trigger {
            region: re_core::Region::Box {
                min: (1, 1, 0),
                max: (1, 1, 0),
            },
            action: re_core::TriggerAction::activate_node("node0"),
            once: true,
        });
        t.place = Some(EntityKind::Trigger);
        t.selected_trigger = Some(0);
        t.entity_field = EntityField::TrigTarget;
        assert_eq!(
            text_target(&t),
            None,
            "ActivateNode target captures no text"
        );
    }

    /// Switching the focused text target re-seeds the caret at the new
    /// text's end (edits never leak across objects/fields).
    #[test]
    fn text_owner_change_reseeds_caret() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.place = Some(EntityKind::Note);
        at(&mut s, 2, 2, 0);
        place_at_cursor(&mut s);
        let idx = s.selected_note.unwrap();
        s.level.notes[idx].text = "abc".into();
        s.entity_field = EntityField::NoteText;
        type_key(&mut s, E::Left); // seeds owner + caret at end, then moves left
        assert_eq!(s.text_caret, crate::widget::Caret { line: 0, col: 2 });
        // Select a DIFFERENT note: the owner changes, so the caret re-seeds at
        // the NEW text's end instead of splicing at the stale column.
        s.level.notes.push(re_level::LevelNote {
            kind: re_level::NoteKind::Researcher,
            x: 4,
            y: 4,
            z: 0,
            facing: 0,
            text: "wxyz".into(),
        });
        s.selected_note = Some(idx + 1);
        type_key(&mut s, E::Char('!'));
        assert_eq!(
            s.level.notes[idx + 1].text,
            "wxyz!",
            "re-seeded caret appends at the end"
        );
        assert_eq!(
            s.level.notes[idx].text, "abc",
            "the first note is untouched"
        );
    }

    /// The `Delete`/`Home`/`End` keycodes
    /// drive the caret ops in a NOTE BODY — forward-delete at the caret, line
    /// start/end jumps, mid-line insertion after Home.
    #[test]
    fn note_body_delete_home_end_work_at_the_caret() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.place = Some(EntityKind::Note);
        at(&mut s, 1, 1, 0);
        place_at_cursor(&mut s);
        s.level.notes[0].text.clear(); // drop the placement seed text
        s.entity_field = crate::state::EntityField::NoteText;
        type_str(&mut s, "ab");
        type_key(&mut s, E::Enter); // note body: Enter = newline
        type_str(&mut s, "cd");
        assert_eq!(s.level.notes[0].text, "ab\ncd");
        // Home → line start; Delete removes the char AT the caret ('c').
        type_key(&mut s, E::Home);
        type_key(&mut s, E::Delete);
        assert_eq!(
            s.level.notes[0].text, "ab\nd",
            "forward-delete at the line start"
        );
        // End → line end; Delete there is a no-op (nothing after the text).
        type_key(&mut s, E::End);
        type_key(&mut s, E::Delete);
        assert_eq!(
            s.level.notes[0].text, "ab\nd",
            "delete at the very end is a no-op"
        );
        // Home then typing INSERTS at the line start (a true mid-string caret).
        type_key(&mut s, E::Home);
        type_key(&mut s, E::Char('X'));
        assert_eq!(
            s.level.notes[0].text, "ab\nXd",
            "insert lands at the caret, not the end"
        );
    }
}
