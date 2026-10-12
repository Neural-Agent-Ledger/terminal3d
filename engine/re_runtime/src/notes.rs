//! `notes` — the generic `[note]` placard mechanism: resolve every authored
//! note to a real wall face and stamp a synthesized text material there.
//! `re_imgqa` and a host game both call this crate rather than each keeping
//! their own copy of the placement rules in sync by hand.
//!
//! # What is generic here, and what is not
//!
//! `re_render` owns the MECHANISM (a per-level [`TextSurfaceRegistry`] of owned
//! text panels, plus the `NoteWall` recipe that paints one placard on one wall
//! face). This module owns the PLACEMENT — resolving an authored `pos` +
//! `facing` to a real, paintable, visible wall cell — which is a pure function
//! of the level's geometry and needs no host-specific content.
//!
//! What is NOT generic, and stays with the host: **which panels exist** (a
//! host may have its own embedded "code wall" content) and **what a note's
//! body says** (a host may template it, e.g. an operator-substitution pass
//! over the raw text). Those decisions cross the seam as plain data/closures
//! into [`build_level_text`] — this module never imports a host crate.
//!
//! # How a host binds its own text
//!
//! Call [`build_level_text`] once per level load, after `level.build_map()` and
//! BEFORE `re_render::registry_for_map(&map)` (the note materials are appended
//! to the map's material list, so the render registry must be built after they
//! are there):
//!
//! ```ignore
//! let panels: Vec<(&str, &str)> = my_panel_sources().collect();
//! let (registry, report) = re_runtime::notes::build_level_text(
//!     &level,
//!     &mut map,
//!     &panels,
//!     Some(&|body: &str| my_template(body)),
//! );
//! re_render::scene_text::install(registry);
//! ```
//!
//! `panels` are registered FIRST, in order, so a `CodeWall { panel: N }` index
//! keeps whatever meaning the host gives it. Then one panel per `[note]` of the
//! loaded level, in parse order, so `NoteWall { note: N }` means "the N-th
//! placard". `transform` is applied to each note's trimmed body before it is
//! registered — pass `None` for a host with no templating step. This function
//! is PURE: it builds a registry and stamps `map`, but never publishes
//! anything process-wide (`re_render::scene_text::install` is the caller's own
//! choice, and callers that need only a diagnostic dry run can skip it).
//!
//! ## Note placement — the pinned semantics, and the three tolerances
//!
//! `pos` is the wall cell the text occupies; `facing` is the OUTWARD normal of
//! the face it is painted on (the reader stands on that side). When `pos` is a
//! solid cell, that is the wall and we are done.
//!
//! Shipped placards, however, mostly author `pos` as a point in the ROOM — the
//! spot the reader stands at — not the wall cell itself. Rejecting them would
//! leave authored placards unrendered, and rewriting them is a separate content
//! item. So when `pos` is not solid we search AWAY from the reader (opposite
//! the facing normal, i.e. the direction the reader is looking) for the first
//! solid cell within [`NOTE_WALL_SEARCH_MAX`] cells, and paint the placard
//! there on the same face — which is exactly the wall the author was pointing
//! at. A note that resolves to no wall at all is skipped and reported, never
//! silently dropped.
//!
//! ### The READER'S EYE LAYER — why the authored `z` is a storey, not a layer
//!
//! The renderer is a 2.5D caster: it walks exactly ONE horizontal grid slice,
//! the one the camera's eye sits in (`re_render::gameplay_eye_ref_z(eye_z).floor()`
//! — `1` on the ground storey, `1 + storey·H` on an upper one). A wall
//! material stamped on any OTHER z-layer is never sampled during play: it is
//! invisible.
//!
//! The authored `z` therefore keeps the meaning it can actually carry — **which
//! storey the placard belongs to** — and the VISIBLE layer is derived from it:
//! [`reader_eye_layer`] maps the authored `z` to the eye layer of the storey it
//! falls in (`z / storey_height`), clamped down into the map's habitable band
//! (a `z` that would land on/above the ceiling slab falls back to the storey
//! below, ultimately the ground storey).
//!
//! ### Sliding along the wall run
//!
//! Collapsing the z axis can make two placards authored one above the other
//! want the SAME cell, and a wall cell that was masonry at the authored `z` can
//! be a DOOR at the eye layer. Rather than drop those placards, one is hung
//! NEXT TO the other: the placard slides along the wall run (the face's
//! tangent) to the nearest cell within [`NOTE_WALL_SLIDE_MAX`] that is
//!
//! * paintable ([`is_paintable_wall`] — masonry, untagged),
//! * unclaimed by an earlier placard this load, and
//! * actually LOOKABLE-AT ([`face_is_visible`] — the cell in front of the face
//!   is not itself solid).
//!
//! It stays on the same wall, on the same face, in the reader's view — and
//! every authored placard survives.
//!
//! ## Cost / purity
//!
//! Runs ONCE per level load: it builds the note panels, appends one material
//! per placed note, and stamps one `side_data` per placard cell. Nothing here
//! runs per frame, so the render dirty-flag is untouched — a placard is static
//! world data, the same cost model as a code wall.

use re_core::FaceDir;
use re_level::{Level, LevelNote};
use re_render::TextSurfaceRegistry;
use re_world::{CellKind, Map, MaterialAnim, MaterialDef, MaterialRecipe};

/// How far to search for the placard's wall when the authored `pos` is not
/// itself a solid cell (see the module docs). Sized to cross a room of the
/// scale levels are typically authored at (16×14) but not to wander into the
/// next chamber through a doorway.
pub const NOTE_WALL_SEARCH_MAX: i32 = 24;

/// How far a placard may SLIDE along its wall run (the face's tangent) when
/// the cell the resolver picked cannot carry it at the reader's eye layer —
/// because another placard already claimed it, or because that cell is a
/// door/tagged cell there.
///
/// Four cells is "still the same stretch of wall, still in the same view": at
/// the ~2.5-cell reading distance a placard occupies, four cells is within the
/// same glance. Larger would let a placard wander around a corner of the room
/// and read as being about something else.
pub const NOTE_WALL_SLIDE_MAX: i32 = 4;

/// Sentinel prefix on a synthesized note material's name, so it can never
/// collide with an authored `[material NAME]` and is obvious in a material
/// dump. Unit-separator framed, like `re_level`'s preset prefix.
pub const NOTE_MATERIAL_PREFIX: &str = "\u{1f}note\u{1f}";

/// Foreground/background of the synthesized placard material. The `NoteWall`
/// recipe drives its own ink/plate palette per character, so these are only
/// the fallback colours a non-recipe path would use; they are kept close to
/// the recipe's own so a fallback never flashes a wrong-coloured plate.
const NOTE_MAT_FG: u32 = 0x00E0_DCC8;
const NOTE_MAT_BG: u32 = 0x0009_0807;

/// The five surface-kind material slots every level's material list starts
/// with; the per-cell colour palette follows them
/// (`re_level::renderer_material_defs`).
const SURFACE_SLOTS: usize = 5;

/// What [`build_level_text`] did, for the caller's log/diagnostics. A note
/// that could not be placed is REPORTED (it is a level data error per the
/// pinned semantics), not swallowed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoteBindReport {
    /// How many placards were bound to a wall face.
    pub placed: usize,
    /// Indices of the notes whose `pos`/`facing` resolved to no solid wall.
    pub unplaced: Vec<usize>,
}

/// Build this level's text registry and stamp its note placards into `map` —
/// the generic mechanism a host wraps at level-load time (see the module
/// docs).
///
/// `panels` are registered first, in order — a host's own embedded content
/// (e.g. code walls), so `CodeWall { panel: N }` keeps whatever meaning the
/// host gives it. `transform`, when `Some`, is applied to every note's
/// trimmed body before it is registered (a host's own text templating step,
/// e.g. Form A-7 operator interpolation); `None` registers the trimmed body
/// verbatim.
///
/// This is the PURE half: it never publishes the registry process-wide
/// (`re_render::scene_text::install` is the caller's own choice).
pub fn build_level_text(
    level: &Level,
    map: &mut Map,
    panels: &[(&str, &str)],
    transform: Option<&dyn Fn(&str) -> String>,
) -> (TextSurfaceRegistry, NoteBindReport) {
    let mut registry = TextSurfaceRegistry::new();

    // 1. The host's own panels, FIRST, so `CodeWall { panel: N }` keeps its index.
    for (label, src) in panels {
        registry.push_code_source(label, src);
    }

    // 2. One panel per `[note]`, in parse order ⇒ `NoteWall { note: N }` == note N.
    for (i, note) in level.notes.iter().enumerate() {
        let trimmed = note.text.trim_end();
        let body = match transform {
            Some(f) => f(trimmed),
            None => trimmed.to_string(),
        };
        registry.push_note_text(&note_label(i, note), &body);
    }

    // 3. The pipe word pools (same lifecycle, same registry) — and the level's
    // glitch density, which drives the clean↔corrupt (`pool_alt`) register.
    registry.set_pools(level.pool_table());
    registry.set_glitch(level.effects.glitch);

    // 4. Stamp the placards onto real wall faces.
    let report = stamp_notes(level, map);
    (registry, report)
}

/// A human label for a note panel (diagnostics only): its register + index.
fn note_label(i: usize, note: &LevelNote) -> String {
    format!("note{i}:{:?}", note.kind)
}

/// Resolve every note to a wall face and stamp a synthesized `NoteWall`
/// material there. Returns which notes landed and which could not.
fn stamp_notes(level: &Level, map: &mut Map) -> NoteBindReport {
    let mut report = NoteBindReport::default();
    if level.notes.is_empty() {
        return report;
    }
    // The colour-material palette is `materials()[5..]`, in 1-based colour-id
    // order (id 1 ⇒ index 5). Appending keeps every existing id stable.
    let mut mats: Vec<MaterialDef> = map.materials().to_vec();
    // A map with no authored materials has an EMPTY list; the palette is
    // defined relative to the five surface slots, so those must exist before
    // we append.
    if mats.len() < SURFACE_SLOTS {
        mats.resize(SURFACE_SLOTS, MaterialDef::fallback());
    }

    // Cells already claimed by a placard this load, so two notes cannot fight
    // over one cell. The resolver treats a claimed cell as unavailable and
    // slides the second placard along the wall run (see the module docs) —
    // only a note with no free cell in reach reports as unplaced, and nothing
    // is ever overwritten.
    let mut claimed: Vec<(i32, i32, i32)> = Vec::new();
    let storey_height = level.storey_height();

    for (i, note) in level.notes.iter().enumerate() {
        let face = FaceDir::from_yaw_col(note.facing);
        let Some((x, y, z)) = resolve_note_wall(map, note, face, storey_height, &claimed) else {
            report.unplaced.push(i);
            continue;
        };
        claimed.push((x, y, z));

        mats.push(MaterialDef {
            name: format!("{NOTE_MATERIAL_PREFIX}{i}"),
            fg: NOTE_MAT_FG,
            bg: NOTE_MAT_BG,
            glyph: None,
            recipe: Some(MaterialRecipe::NoteWall {
                note: i as u32,
                face,
            }),
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        });
        // Colour id = position within the palette (1-based).
        let color_id = (mats.len() - SURFACE_SLOTS) as u16;
        let kind = map.cell_at(x, y, z);
        map.set_cell(x as usize, y as usize, z as usize, kind, color_id);
        report.placed += 1;
    }

    map.set_materials(mats);
    report
}

/// Find the wall cell a note is painted on, AT THE READER'S EYE LAYER (the
/// only layer the 2.5D caster samples — see the module docs).
///
/// Three stages, in preference order:
///
/// 1. **Which wall**, decided in the AUTHORED z-plane — the plane the level
///    author actually looked at: the authored `pos` when it is itself
///    masonry, otherwise the first masonry cell along the reader's gaze
///    ([`wall_in_plane`]).
/// 2. **Which layer**: that wall column, taken at the eye layer of the storey
///    the authored `z` falls in ([`reader_eye_layer`]) — sliding along the
///    wall run when the exact cell there is claimed, tagged, or has its face
///    blocked ([`slide_to_free_wall`]).
/// 3. **Fallback**: if the authored plane names a wall column with nothing
///    usable at the eye layer at all, re-run the gaze search IN the eye layer
///    (a level whose upper wall geometry differs from the layer the reader
///    sees).
///
/// `claimed` is the cells already taken by earlier placards this load.
/// Returns `None` only when no free wall cell exists in reach — the note is
/// then reported unplaced, never silently dropped.
fn resolve_note_wall(
    map: &Map,
    note: &LevelNote,
    face: FaceDir,
    storey_height: u32,
    claimed: &[(i32, i32, i32)],
) -> Option<(i32, i32, i32)> {
    let eye_z = reader_eye_layer(map, note.z, storey_height);
    if let Some((wx, wy)) = wall_in_plane(map, note.x, note.y, note.z, face) {
        if let Some(cell) = slide_to_free_wall(map, wx, wy, eye_z, face, claimed) {
            return Some(cell);
        }
    }
    if eye_z != note.z {
        if let Some((wx, wy)) = wall_in_plane(map, note.x, note.y, eye_z, face) {
            if let Some(cell) = slide_to_free_wall(map, wx, wy, eye_z, face, claimed) {
                return Some(cell);
            }
        }
    }
    None
}

/// The z-LAYER the reader's eye sits in for the storey the authored `z`
/// belongs to — i.e. the one horizontal slice the gameplay caster walks, and
/// therefore the ONLY layer a placard can be seen on.
///
/// `storey = z / storey_height`; the layer is that storey's grounded eye
/// layer, derived from [`re_render::gameplay_eye_ref_z`] at that storey's
/// standing eye height so there is NO hand-copied "1" here — the render
/// layer's own reference is the single source of truth.
///
/// A storey whose eye layer would land on or above the CEILING slab is not a
/// storey the reader can stand on (`re_world`'s camera clamp keeps the eye
/// layer inside `[1, sz-2]`), so we step down a storey at a time until it
/// fits. That is what makes an authored `z=4` on a 6-cell-tall SINGLE-storey
/// room resolve to the ground layer instead of vanishing into the ceiling.
fn reader_eye_layer(map: &Map, authored_z: i32, storey_height: u32) -> i32 {
    let h = storey_height.max(1) as i32;
    let (_, _, sz) = map.size();
    let ground = storey_eye_layer(0, h);
    // Habitable top: the ceiling slab is `sz-1`, so the highest layer a
    // standing eye can occupy is `sz-2` (the same band `CameraPose::move_z`
    // clamps to).
    let top = (sz as i32 - 2).max(ground);
    let mut storey = authored_z.max(0) / h;
    while storey > 0 && storey_eye_layer(storey, h) > top {
        storey -= 1;
    }
    storey_eye_layer(storey, h).clamp(ground, top)
}

/// The grounded eye LAYER (integer z-cell) of storey `storey` on a level whose
/// storey height is `h` cells: exactly what the caster computes for a player
/// standing there.
fn storey_eye_layer(storey: i32, h: i32) -> i32 {
    let eye = re_world::STANDING_EYE_Z + (storey * h) as f32;
    re_render::gameplay_eye_ref_z(eye).floor() as i32
}

/// WHICH WALL, within one z-plane: the authored `(x,y)` when it is itself
/// masonry, otherwise the first masonry cell along the reader's gaze
/// (opposite the face's outward normal) within [`NOTE_WALL_SEARCH_MAX`].
/// Returns the wall column's `(x, y)`; the layer is decided by the caller.
///
/// This stage asks only "where does the reader's line of sight MEET a wall"
/// — it tests the cell KIND ([`is_wall_kind`]), not paintability. Whether that
/// exact cell can carry the placard (untagged, unclaimed) is
/// [`slide_to_free_wall`]'s decision: a wall that is already tagged must move
/// the placard SIDEWAYS along the run, not send the search hunting for a
/// different wall on the far side of the room.
fn wall_in_plane(map: &Map, x: i32, y: i32, z: i32, face: FaceDir) -> Option<(i32, i32)> {
    if is_wall_kind(map, x, y, z) {
        return Some((x, y));
    }
    // The reader stands on the `face` side and looks along `-face`; the wall
    // is therefore somewhere along `-face` from the authored point.
    let (dx, dy) = face.step();
    for step in 1..=NOTE_WALL_SEARCH_MAX {
        let (cx, cy) = (x - dx * step, y - dy * step);
        if !map.in_bounds(cx, cy, z) {
            break;
        }
        if is_wall_kind(map, cx, cy, z) {
            return Some((cx, cy));
        }
    }
    None
}

/// Take wall column `(wx, wy)` at layer `z`, sliding ALONG THE WALL RUN to the
/// nearest paintable, unclaimed cell **whose face the reader can actually
/// see**, within [`NOTE_WALL_SLIDE_MAX`].
///
/// Offset order is `0, +1, −1, +2, −2, …`: nearest first (so an unobstructed
/// placard lands exactly where the resolver pointed), and deterministic — the
/// same level always produces the same placard layout, which the golden
/// frames depend on.
fn slide_to_free_wall(
    map: &Map,
    wx: i32,
    wy: i32,
    z: i32,
    face: FaceDir,
    claimed: &[(i32, i32, i32)],
) -> Option<(i32, i32, i32)> {
    // The wall RUN is perpendicular to the face normal: an x-facing placard
    // slides along y, a y-facing one along x.
    let (tx, ty) = match face {
        FaceDir::PosX | FaceDir::NegX => (0, 1),
        FaceDir::PosY | FaceDir::NegY => (1, 0),
    };
    for d in 0..=NOTE_WALL_SLIDE_MAX {
        for sign in [1, -1] {
            if d == 0 && sign < 0 {
                continue; // offset 0 is tried once
            }
            let off = d * sign;
            let cell = (wx + tx * off, wy + ty * off, z);
            if is_paintable_wall(map, cell.0, cell.1, cell.2)
                && face_is_visible(map, cell.0, cell.1, cell.2, face)
                && !claimed.contains(&cell)
            {
                return Some(cell);
            }
        }
    }
    None
}

/// Whether the reader can SEE this face at all: the cell immediately in front
/// of it (one step along the face's outward normal) must be non-solid,
/// in-bounds air the reader can stand in and look from.
///
/// A placard is painted on ONE face of ONE cell, and the caster only shades a
/// face it reaches — so a face pressed against a door anchor, a node, or
/// another wall renders nothing no matter how well the text is laid out.
/// Checking it here lets the slide step past such a cell instead of hanging a
/// placard in a place with no viewer.
///
/// One cell of look-ahead only: this is a "is there a viewing spot at all"
/// test, not visibility from the player's actual position (which changes
/// every frame and is the caster's business).
fn face_is_visible(map: &Map, x: i32, y: i32, z: i32, face: FaceDir) -> bool {
    let (dx, dy) = face.step();
    let (fx, fy) = (x + dx, y + dy);
    map.in_bounds(fx, fy, z) && !map.cell_at(fx, fy, z).is_solid()
}

/// Whether a cell is architectural MASONRY — one of the three WALL kinds.
///
/// A level's shell is `WallSolid`, `WallTrace`, or `WallMatrix` depending on
/// its authored `[room] ambient`, so all three are ordinary masonry from a
/// placard's point of view. A node/door (`Object`) cell is NOT: its
/// `side_data` is a puzzle TAG, and stamping a colour id over it would break
/// the puzzle's own binding (and hide a node).
fn is_wall_kind(map: &Map, x: i32, y: i32, z: i32) -> bool {
    map.in_bounds(x, y, z)
        && matches!(
            map.cell_at(x, y, z),
            CellKind::WallSolid | CellKind::WallTrace | CellKind::WallMatrix
        )
}

/// Whether a cell can CARRY a placard: masonry ([`is_wall_kind`]) that is also
/// UNTAGGED.
///
/// The `side_data` must be `0`. A non-zero tag is already meaningful: a pipe
/// tag on a `WallTrace` (the high-bit pipe namespace), an authored per-cell
/// colour id, or an earlier placard's own id. Overwriting either would
/// silently retexture the level.
fn is_paintable_wall(map: &Map, x: i32, y: i32, z: i32) -> bool {
    is_wall_kind(map, x, y, z) && map.side_data_at(x, y, z) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::NoteKind;

    /// The active registry `re_render::scene_text` publishes is PROCESS-GLOBAL
    /// (by design — see `re_render::scene_text`'s module docs), and `cargo
    /// test` runs a crate's tests in parallel threads. Every test here that
    /// installs/reads the active registry takes this lock first so they
    /// cannot race each other or a host game's / `re_imgqa`'s own copy of the
    /// same lock (each crate's tests run in their own process).
    fn registry_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The CROSS-CRATE contract: a host writes the panel label, `re_render`
    /// reads the register back out of it, and the two agree only because
    /// `NoteRegister::label_suffix` is spelled exactly like `NoteKind`'s
    /// `Debug` name.
    ///
    /// Nothing else enforces that. Rename either side and every placard
    /// silently degrades to the `Plain` monochrome plate — no panic, no
    /// failing build, just a level whose voices all look the same. This test
    /// is the enforcement: it walks EVERY `NoteKind` so a newly added variant
    /// fails here until its register exists.
    #[test]
    fn every_note_kind_round_trips_through_the_panel_label() {
        use re_render::note_style::NoteRegister;
        for (kind, want) in [
            (NoteKind::Researcher, NoteRegister::Researcher),
            (NoteKind::SystemLog, NoteRegister::SystemLog),
            (NoteKind::ModelThought, NoteRegister::ModelThought),
            (NoteKind::Stencil, NoteRegister::Stencil),
        ] {
            let n = note(kind, 1, 1, 1, 360, "body");
            let label = note_label(3, &n);
            assert_eq!(
                NoteRegister::from_label(&label),
                want,
                "label {label:?} (from {kind:?}) must resolve to {want:?}, not the \
                 Plain fallback — the register is selected by SPELLING"
            );
        }
    }

    fn note(kind: NoteKind, x: i32, y: i32, z: i32, facing: i32, text: &str) -> LevelNote {
        LevelNote {
            kind,
            x,
            y,
            z,
            facing,
            text: text.to_string(),
        }
    }

    /// The default storey height every `.lvl` without an explicit `[room]
    /// storey_height` gets (`re_level::DEFAULT_STOREY_HEIGHT`), so the
    /// fixtures exercise the same arithmetic the shipped levels do.
    const H: u32 = re_level::DEFAULT_STOREY_HEIGHT;

    /// A 8×8×3 room: solid perimeter, air inside. `sz = 3` ⇒ the habitable
    /// band is the single layer `z=1`, which is also the reader's eye layer.
    fn room() -> Map {
        walled_box(8, 8, 3)
    }

    /// An `sx × sy × sz` room: solid perimeter columns on every layer, a
    /// floor slab at `z=0`, air inside.
    fn walled_box(sx: usize, sy: usize, sz: usize) -> Map {
        let mut m = Map::new(sx, sy, sz);
        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    let perim = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    if perim {
                        m.set(x, y, z, CellKind::WallSolid);
                    } else if z == 0 {
                        m.set(x, y, z, CellKind::Floor);
                    }
                }
            }
        }
        m
    }

    /// `resolve_note_wall` with nothing claimed yet — the shape most tests want.
    fn resolve(map: &Map, n: &LevelNote, face: FaceDir) -> Option<(i32, i32, i32)> {
        resolve_note_wall(map, n, face, H, &[])
    }

    #[test]
    fn a_note_on_a_solid_cell_binds_to_that_cell() {
        let map = room();
        // (7,3,1) is the +x perimeter wall; its inward-facing face normal is −x.
        let n = note(NoteKind::Researcher, 7, 3, 1, 360 /* -x */, "hi");
        assert_eq!(resolve(&map, &n, FaceDir::NegX), Some((7, 3, 1)));
    }

    /// The shipped-placard case: `pos` is the spot the READER stands. The
    /// wall is found by looking along the reader's gaze (opposite the face
    /// normal).
    #[test]
    fn a_note_in_open_air_resolves_to_the_wall_the_reader_faces() {
        let map = room();
        // Reader at (3,3,1) looking +x (the face they read has normal −x).
        let n = note(NoteKind::SystemLog, 3, 3, 1, 360 /* -x */, "hi");
        assert_eq!(
            resolve(&map, &n, FaceDir::NegX),
            Some((7, 3, 1)),
            "searching along the gaze must land on the far +x wall"
        );
        // The same point facing the other way lands on the opposite wall.
        let n2 = note(NoteKind::SystemLog, 3, 3, 1, 120 /* +x */, "hi");
        assert_eq!(resolve(&map, &n2, FaceDir::PosX), Some((0, 3, 1)));
    }

    #[test]
    fn a_note_with_no_wall_in_range_is_unplaced() {
        // An all-air map: nothing to paint on.
        let map = Map::new(4, 4, 2);
        let n = note(NoteKind::Researcher, 1, 1, 1, 0, "hi");
        assert_eq!(resolve(&map, &n, FaceDir::PosY), None);
    }

    // ── THE EYE-LAYER SNAP (without it, most placards were invisible) ──

    /// A note authored ABOVE the reader's eye layer on the GROUND storey is
    /// stamped AT that eye layer — the only horizontal slice the 2.5D caster
    /// samples. Every shipped placard authors `z ≥ 2`, so this is the fix
    /// that makes them exist.
    #[test]
    fn a_ground_storey_note_authored_at_z2_stamps_at_the_eye_layer() {
        // sz = 6 ⇒ the habitable band is z ∈ [1,4]; the ground eye layer is 1.
        let map = walled_box(8, 8, 6);
        assert_eq!(reader_eye_layer(&map, 2, H), 1);
        let n = note(NoteKind::ModelThought, 3, 3, 2, 360 /* -x */, "hi");
        assert_eq!(
            resolve(&map, &n, FaceDir::NegX),
            Some((7, 3, 1)),
            "an authored z=2 placard must be stamped on the z=1 (eye) layer"
        );
        // Same for the other authored heights the shipped levels use (z=3, z=4):
        // on a single-storey room they are all the ground storey.
        for z in [1, 2, 3, 4] {
            assert_eq!(reader_eye_layer(&map, z, H), 1, "authored z={z}");
        }
    }

    /// A note on an UPPER storey keeps its own storey's eye layer — the
    /// authored `z` selects the STOREY, and storey 1 of an `H=4` level is
    /// read at `z=5` (exactly what `re_render::gameplay_eye_ref_z` returns for
    /// a player standing there).
    #[test]
    fn an_upper_storey_note_stamps_at_that_storeys_eye_layer() {
        // A two-storey shell (L1's shape): sz = 13, H = 4 ⇒ eye layers 1, 5, 9.
        let map = walled_box(8, 8, 13);
        assert_eq!(reader_eye_layer(&map, 5, H), 5, "storey 1 is read at z=5");
        assert_eq!(
            reader_eye_layer(&map, 4, H),
            5,
            "z=4 is storey 1's floor slab"
        );
        assert_eq!(reader_eye_layer(&map, 9, H), 9, "storey 2 is read at z=9");
        assert_eq!(
            reader_eye_layer(&map, 3, H),
            1,
            "z=3 is still the ground storey"
        );
        let n = note(NoteKind::ModelThought, 3, 3, 5, 360 /* -x */, "hi");
        assert_eq!(resolve(&map, &n, FaceDir::NegX), Some((7, 3, 5)));
        // …and the render layer agrees this IS the layer it samples for a
        // player standing on storey 1 (no hand-copied constant on either
        // side).
        let storey1_eye = re_world::STANDING_EYE_Z + H as f32;
        assert_eq!(re_render::gameplay_eye_ref_z(storey1_eye).floor() as i32, 5);
    }

    /// A storey whose eye layer would sit in (or above) the CEILING slab is
    /// not a storey anyone stands on, so the note falls back to the storey
    /// below. This is what keeps a `z=4` placard on a single-storey room only
    /// 5–6 cells tall on the ground layer instead of stamping it into the
    /// ceiling.
    #[test]
    fn an_eye_layer_above_the_habitable_band_falls_back_to_the_storey_below() {
        let short = walled_box(8, 8, 5); // habitable band = z ∈ [1,3]
        for z in [2, 3, 4, 7, 99] {
            assert_eq!(
                reader_eye_layer(&short, z, H),
                1,
                "a 5-cell-tall room has ONE reader layer; authored z={z}"
            );
        }
        // Degenerate maps still answer (never a panic, never a negative layer).
        let flat = Map::new(4, 4, 1);
        assert_eq!(reader_eye_layer(&flat, 3, H), 1);
        assert_eq!(reader_eye_layer(&short, -5, H), 1);
        // A degenerate `storey_height = 0` (the `re_level` sentinel, which the
        // `storey_height()` accessor never actually returns) must not divide
        // by zero; it degrades to "one storey per cell" and still answers
        // inside the band.
        let degenerate = reader_eye_layer(&short, 2, 0);
        assert!(
            (1..=3).contains(&degenerate),
            "H=0 must stay inside the habitable band, got {degenerate}"
        );
    }

    /// Collapsing the z axis makes two placards authored one ABOVE the other
    /// want the same cell. The second hangs NEXT TO the first along the wall
    /// run instead of being dropped.
    #[test]
    fn a_second_placard_on_a_claimed_cell_slides_along_the_wall() {
        let map = walled_box(8, 8, 6);
        let a = note(NoteKind::Researcher, 3, 3, 2, 360 /* -x */, "first");
        let first = resolve(&map, &a, FaceDir::NegX).unwrap();
        assert_eq!(first, (7, 3, 1));
        let b = note(NoteKind::SystemLog, 3, 3, 3, 360, "second");
        let second = resolve_note_wall(&map, &b, FaceDir::NegX, H, &[first]).unwrap();
        assert_eq!(
            second,
            (7, 4, 1),
            "the second placard slides one cell along the wall run (+y), same face, same layer"
        );
        // A third slides to the other side of the first (offset order 0,+1,−1,…).
        let c = note(NoteKind::ModelThought, 3, 3, 4, 360, "third");
        let third = resolve_note_wall(&map, &c, FaceDir::NegX, H, &[first, second]).unwrap();
        assert_eq!(third, (7, 2, 1));
    }

    /// The wall cell the author pointed at can be a DOOR at the reader's eye
    /// layer even though it was masonry at the authored `z`. The placard
    /// slides along the wall rather than reporting unplaced.
    #[test]
    fn a_placard_slides_past_a_door_cell_at_the_eye_layer() {
        let mut map = walled_box(8, 8, 6);
        map.set_cell(7, 3, 1, CellKind::Object, 0); // the door anchor, at eye level
        let n = note(NoteKind::ModelThought, 3, 3, 2, 360 /* -x */, "hi");
        assert_eq!(
            resolve(&map, &n, FaceDir::NegX),
            Some((7, 4, 1)),
            "a placard must hang BESIDE the door, not vanish"
        );
        assert_eq!(map.side_data_at(7, 3, 1), 0, "the door cell is untouched");
    }

    /// A face with a SOLID cell standing in front of it can never be seen.
    /// The placard slides to a face the reader can actually look at.
    #[test]
    fn a_placard_skips_a_face_blocked_by_the_cell_in_front_of_it() {
        let mut map = walled_box(8, 8, 6);
        // The door anchor stands in the air cell in FRONT of the +x wall at y=3.
        map.set_cell(6, 3, 1, CellKind::Object, 0);
        assert!(!face_is_visible(&map, 7, 3, 1, FaceDir::NegX));
        assert!(face_is_visible(&map, 7, 4, 1, FaceDir::NegX));
        let n = note(NoteKind::Researcher, 3, 3, 2, 360 /* -x */, "hi");
        assert_eq!(
            resolve(&map, &n, FaceDir::NegX),
            Some((7, 4, 1)),
            "a placard must not hang on a face nobody can stand in front of"
        );
        // An out-of-bounds "front" (a placard on the outer skin, facing out)
        // is not visible either — there is no room out there to read from.
        assert!(!face_is_visible(&map, 0, 3, 1, FaceDir::NegX));
        assert!(face_is_visible(&map, 0, 3, 1, FaceDir::PosX));
    }

    /// The slide is bounded: a wall run fully blocked within
    /// [`NOTE_WALL_SLIDE_MAX`] reports the note unplaced rather than
    /// wandering across the room.
    #[test]
    fn a_fully_blocked_wall_run_reports_unplaced() {
        let mut map = walled_box(16, 16, 6);
        // Tag out the whole reachable stretch of the +x wall at the eye layer.
        for y in 0..16 {
            map.set_cell(15, y, 1, CellKind::WallSolid, 9); // authored colour id ⇒ not paintable
        }
        let n = note(NoteKind::Researcher, 3, 3, 2, 360 /* -x */, "hi");
        assert_eq!(resolve(&map, &n, FaceDir::NegX), None);
    }

    #[test]
    fn a_note_never_binds_to_a_puzzle_object_cell() {
        let mut map = room();
        map.set_cell(3, 3, 1, CellKind::Object, 7); // a node cell with tag 7
        assert!(
            !is_paintable_wall(&map, 3, 3, 1),
            "a placard must not claim a node/door cell (its side_data is a puzzle tag)"
        );
        // Resolution therefore walks PAST it to the masonry beyond.
        let n = note(NoteKind::Researcher, 2, 3, 1, 360, "hi");
        assert_eq!(resolve(&map, &n, FaceDir::NegX), Some((7, 3, 1)));
        // …and the node's tag is untouched.
        assert_eq!(map.side_data_at(3, 3, 1), 7);
    }

    #[test]
    fn stamping_appends_one_palette_material_per_placed_note_and_tags_the_cell() {
        let mut map = room();
        let level = Level {
            notes: vec![
                note(NoteKind::Researcher, 3, 3, 1, 360, "first"),
                note(NoteKind::SystemLog, 3, 5, 1, 120, "second"),
            ],
            ..Default::default()
        };
        let report = stamp_notes(&level, &mut map);
        assert_eq!(report.placed, 2);
        assert!(report.unplaced.is_empty());

        // Two palette entries appended after the five surface slots, in note order.
        let mats = map.materials();
        assert_eq!(mats.len(), SURFACE_SLOTS + 2);
        assert!(matches!(
            mats[SURFACE_SLOTS].recipe,
            Some(MaterialRecipe::NoteWall {
                note: 0,
                face: FaceDir::NegX
            })
        ));
        assert!(matches!(
            mats[SURFACE_SLOTS + 1].recipe,
            Some(MaterialRecipe::NoteWall {
                note: 1,
                face: FaceDir::PosX
            })
        ));

        // The resolved wall cells carry the 1-based colour ids, and stay walls.
        assert_eq!(map.side_data_at(7, 3, 1), 1);
        assert_eq!(map.cell_at(7, 3, 1), CellKind::WallSolid);
        assert_eq!(map.side_data_at(0, 5, 1), 2);
    }

    /// A level whose `[room] ambient` makes the shell a TRACE/MATRIX wall is
    /// still paintable — every shipped `trace-pattern` room would otherwise
    /// lose its notes.
    #[test]
    fn a_trace_or_matrix_shell_is_paintable_but_a_tagged_cell_is_not() {
        let mut map = room();
        map.set(7, 3, 1, CellKind::WallTrace);
        assert!(
            is_paintable_wall(&map, 7, 3, 1),
            "a trace shell is masonry too"
        );
        map.set(7, 4, 1, CellKind::WallMatrix);
        assert!(is_paintable_wall(&map, 7, 4, 1));
        // A pipe-tagged trace cell (high-bit pipe namespace) must NOT be claimed.
        map.set_cell(
            7,
            5,
            1,
            CellKind::WallTrace,
            re_core::encode_pipe_side_data(3),
        );
        assert!(!is_paintable_wall(&map, 7, 5, 1));
        // Nor an already-colour-tagged wall.
        map.set_cell(7, 6, 1, CellKind::WallSolid, 4);
        assert!(!is_paintable_wall(&map, 7, 6, 1));
    }

    /// Two notes that resolve to ONE cell both survive: the second hangs
    /// beside the first (never on top of it), each with its own material and
    /// colour id.
    #[test]
    fn two_notes_aiming_at_one_cell_hang_side_by_side() {
        let mut map = room();
        // Both aim at (7,3,1).
        let level = Level {
            notes: vec![
                note(NoteKind::Researcher, 3, 3, 1, 360, "first"),
                note(NoteKind::SystemLog, 5, 3, 1, 360, "second"),
            ],
            ..Default::default()
        };
        let report = stamp_notes(&level, &mut map);
        assert_eq!(report.placed, 2);
        assert!(report.unplaced.is_empty());
        assert_eq!(
            map.side_data_at(7, 3, 1),
            1,
            "note 0 keeps the cell it resolved to"
        );
        assert_eq!(
            map.side_data_at(7, 4, 1),
            2,
            "note 1 slid one cell along the run"
        );
    }

    #[test]
    fn a_level_with_no_notes_leaves_the_map_untouched() {
        let mut map = room();
        let before = map.materials().len();
        let level = Level::default();
        let report = stamp_notes(&level, &mut map);
        assert_eq!(report, NoteBindReport::default());
        assert_eq!(map.materials().len(), before, "no materials appended");
    }

    /// The registry LIFECYCLE across two sequential builds: each build
    /// produces a FRESH registry carrying only THAT level's note panels
    /// (never the sum of both), and the panels list is registered once per
    /// build, not doubled. Uses a synthetic panel list — this is the generic
    /// contract, independent of any host's real content.
    #[test]
    fn building_a_second_level_produces_a_fresh_registry() {
        let panels: [(&str, &str); 1] = [("test-panel", "fn f() {}")];
        let mut map_a = room();
        let a = Level {
            notes: vec![
                note(NoteKind::Researcher, 3, 3, 1, 360, "first"),
                note(NoteKind::SystemLog, 3, 5, 1, 360, "second"),
            ],
            ..Default::default()
        };
        let (reg_a, _) = build_level_text(&a, &mut map_a, &panels, None);
        assert_eq!(reg_a.note_len(), 2);
        assert_eq!(reg_a.code_len(), 1, "the panel list is registered");

        let mut map_b = room();
        let b = Level {
            notes: vec![note(NoteKind::ModelThought, 3, 3, 1, 360, "only")],
            ..Default::default()
        };
        let (reg_b, _) = build_level_text(&b, &mut map_b, &panels, None);
        assert_eq!(
            reg_b.note_len(),
            1,
            "a second build must produce a FRESH panel set, not accumulate them"
        );
        assert_eq!(reg_b.code_len(), 1, "panels registered once, not doubled");
    }

    /// A `transform` is applied to every note body before it is registered —
    /// the seam a host's own text templating (e.g. Form A-7 interpolation)
    /// crosses through.
    #[test]
    fn a_transform_is_applied_to_every_note_body() {
        let mut map = room();
        let level = Level {
            notes: vec![note(NoteKind::Researcher, 3, 3, 1, 360, "hello")],
            ..Default::default()
        };
        let upper: &dyn Fn(&str) -> String = &|s: &str| s.to_uppercase();
        let (registry, report) = build_level_text(&level, &mut map, &[], Some(upper));
        assert_eq!(report.placed, 1);
        let _serial = registry_test_lock();
        let _scope = re_render::SceneTextScope::new(registry);
        // Scan across the panel width for the first alphabetic glyph — exactly
        // where the source text starts is a layout detail this test does not
        // pin; that the letter it finds is UPPERCASE is what proves the
        // transform ran before the body was registered.
        let mut found = None;
        let mut steps = 0u32;
        while steps <= 100 {
            let u = f32::from(steps as u16) / 100.0;
            let g = re_render::scene_text::sample_note_panel(0, u, 0.5, 0.0, 0.0);
            if let Some(ch) = char::from_u32(g.glyph.glyph) {
                if ch.is_ascii_alphabetic() {
                    found = Some(ch);
                    break;
                }
            }
            steps += 1;
        }
        assert_eq!(
            found.map(|c| c.is_ascii_uppercase()),
            Some(true),
            "the transform must run before the body is registered; found {found:?}"
        );
    }

    /// The pool table reaches the render layer through the registry: a
    /// pipe's tag resolves to its SOURCE node's pool.
    #[test]
    fn building_publishes_the_pool_table() {
        let mut map = room();
        let level = Level {
            nodes: vec![
                re_level::Node {
                    name: "a".into(),
                    pool: vec!["river".into(), "cold".into()],
                    anchor: 1,
                    ..Default::default()
                },
                re_level::Node {
                    name: "b".into(),
                    pool: vec!["seen".into()],
                    ..Default::default()
                },
            ],
            pipes: vec![re_level::Pipe {
                from: "a".into(),
                to: "b".into(),
                from_node: 0,
                to_node: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let (registry, _) = build_level_text(&level, &mut map, &[], None);

        // Pipe tag 0 ⇒ node "a" (the SOURCE) ⇒ its pool, anchor-first.
        let anchor = registry
            .pools()
            .pool_for_pipe(0)
            .and_then(|p| p.ordered_word(0));
        assert_eq!(anchor, Some("cold"), "anchor = pool[1] of node a");
    }

    // ── the SHIPPED content: every placard must actually exist in the render ──

    /// The workspace root, for reading `levels/*.lvl` (same derivation as the
    /// golden-framebuffer test).
    fn workspace_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    /// Every `.lvl` under `levels/`, in a stable order.
    fn shipped_levels() -> Vec<std::path::PathBuf> {
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(workspace_root().join("levels"))
            .expect("levels/ must be readable from the workspace root")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "lvl"))
            .collect();
        paths.sort();
        paths
    }

    /// Mirror of [`stamp_notes`]'s resolution loop, returning the CELL each
    /// note landed on (which `NoteBindReport` deliberately does not carry).
    fn resolved_cells(level: &Level, map: &Map) -> Vec<Option<(i32, i32, i32)>> {
        let mut claimed: Vec<(i32, i32, i32)> = Vec::new();
        let h = level.storey_height();
        level
            .notes
            .iter()
            .map(|n| {
                let face = FaceDir::from_yaw_col(n.facing);
                let cell = resolve_note_wall(map, n, face, h, &claimed);
                if let Some(c) = cell {
                    claimed.push(c);
                }
                cell
            })
            .collect()
    }

    /// THE REGRESSION: every placard of every shipped level lands on a wall
    /// cell **at a layer the reader's eye actually occupies**, each on its own
    /// cell.
    #[test]
    fn every_shipped_placard_lands_on_a_layer_the_reader_can_see() {
        let mut checked = 0usize;
        let mut placards = 0usize;
        for path in shipped_levels() {
            let Ok(level) = re_level::load(&path) else {
                continue; // an unparseable level is another test's failure, not this one's
            };
            if level.notes.is_empty() {
                continue;
            }
            let mut map = level.build_map();
            let cells = resolved_cells(&level, &map);
            let report = stamp_notes(&level, &mut map);
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            assert!(
                report.unplaced.is_empty(),
                "{name}: placard(s) {:?} resolve to no wall — they would render nowhere",
                report.unplaced
            );
            assert_eq!(
                report.placed,
                level.notes.len(),
                "{name}: placed != authored"
            );
            for (i, cell) in cells.iter().enumerate() {
                let (x, y, z) = cell.unwrap_or_else(|| panic!("{name}: note {i} unplaced"));
                let expected = reader_eye_layer(&map, level.notes[i].z, level.storey_height());
                assert_eq!(
                    z, expected,
                    "{name}: note {i} (authored z={}) stamped at z={z}, not the reader's \
                     eye layer {expected} — the caster would never sample it",
                    level.notes[i].z
                );
                assert!(
                    is_wall_kind(&map, x, y, z),
                    "{name}: note {i} landed on a non-wall cell ({x},{y},{z})"
                );
                placards += 1;
            }
            // No two placards share a cell (each is a distinct material binding).
            let mut seen = cells.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(
                seen.len(),
                cells.len(),
                "{name}: two placards share one cell"
            );
            checked += 1;
        }
        // This engine-only corpus's note-carrying levels (`04`, `06`, `07`,
        // `08`, `100-signage-showcase`, `U1-underline`) total 6 — the campaign
        // story levels, which also author notes, live in the game repo's own
        // `levels/` and are not part of this corpus.
        assert!(
            checked >= 6,
            "expected the shipped corpus to carry notes, saw {checked}"
        );
        println!("[notes] {placards} placards across {checked} levels, all on a reader layer");
    }

    /// THE LETTER-COUNT METRIC: composite
    /// `levels/06.lvl` with the camera at reading distance from a resolved
    /// placard face and count the ALPHABETIC glyphs in the frame. A placard
    /// that renders shows dozens of letters; the pre-fix frame (placard
    /// stamped on an unsampled layer) showed exactly zero. Uses `panels: &[]`
    /// and `transform: None` — this test is about placement + the panel
    /// mechanism, not any host's content.
    #[test]
    fn compositing_level_06_at_reading_distance_shows_placard_letters() {
        use re_core::Framebuffer;

        let level = re_level::load(workspace_root().join("levels/06.lvl"))
            .expect("levels/06.lvl must parse");
        assert!(!level.notes.is_empty(), "06.lvl must still author placards");
        let mut map = level.build_map();
        let cells = resolved_cells(&level, &map);
        let (registry, report) = build_level_text(&level, &mut map, &[], None);
        assert_eq!(report.placed, level.notes.len());

        // Stand 2.5 cells out from the resolved placard FACE, looking straight at it.
        let (cx, cy, cz) = cells[0].expect("note 0 must resolve");
        let face = FaceDir::from_yaw_col(level.notes[0].facing);
        let n = face.normal();
        let mut cam = re_world::CameraPose::spawn_at(cx, cy, cz, level.notes[0].facing);
        cam.pos = re_core::Vec3::new(
            cx as f32 + 0.5 + n.x * (0.5 + READING_DISTANCE),
            cy as f32 + 0.5 + n.y * (0.5 + READING_DISTANCE),
            re_world::STANDING_EYE_Z,
        );
        // Look along −normal (into the wall). `CameraPose::dir` is `(sin yaw, cos yaw)`.
        cam.yaw = (-n.x).atan2(-n.y);
        cam.pitch = 0.0;
        assert_eq!(
            cz,
            re_render::gameplay_eye_ref_z(cam.pos.z).floor() as i32,
            "the placard layer must be the layer this camera's cast samples"
        );

        let shade = re_render::registry_for_map(&map);
        let _serial = registry_test_lock();
        // The active text registry is process-global; another test in this
        // binary can replace it mid-flight. Retry a bounded number of times
        // so the assertion below is about the RENDER, not the race.
        let mut best = 0usize;
        let mut registry = Some(registry);
        for _ in 0..3 {
            let reg = match registry.take() {
                Some(r) => r,
                None => build_level_text(&level, &mut level.build_map(), &[], None).0,
            };
            let _scope = re_render::SceneTextScope::new(reg);
            let mut fb = Framebuffer::new(100, 32);
            re_render::composite::composite_dump_once(
                &map,
                &cam,
                &shade,
                &re_render::Fov::default(),
                &[],
                &mut fb,
            );
            let letters = (0..fb.rows())
                .flat_map(|r| (0..fb.cols()).map(move |c| (c, r)))
                .filter(|&(c, r)| {
                    fb.get(c, r)
                        .and_then(|cell| char::from_u32(cell.glyph))
                        .is_some_and(|ch| ch.is_ascii_alphabetic())
                })
                .count();
            best = best.max(letters);
            if std::env::var_os("TS_SHOW_NOTE_FRAME").is_some() {
                for r in 0..fb.rows() {
                    let line: String = (0..fb.cols())
                        .map(|c| char::from_u32(fb.get(c, r).unwrap().glyph).unwrap_or(' '))
                        .collect();
                    println!("{line}");
                }
            }
            if best > MIN_PLACARD_LETTERS {
                break;
            }
        }
        assert!(
            best > MIN_PLACARD_LETTERS,
            "06.lvl at reading distance showed only {best} alphabetic glyphs \
             (expected > {MIN_PLACARD_LETTERS}) — the placard is not rendering"
        );
        println!("[notes] 06.lvl placard letters at {READING_DISTANCE} cells: {best}");
    }

    /// How far out from the placard FACE the letter-count camera stands, in
    /// cells — the distance a player stops at to read a note.
    const READING_DISTANCE: f32 = 2.5;
    /// The legibility floor for that frame: a rendering placard puts dozens
    /// of letters on screen (the broken frame put zero).
    const MIN_PLACARD_LETTERS: usize = 50;
}
