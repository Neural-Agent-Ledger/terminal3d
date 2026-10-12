//! `emit_parts` — emitter-part material/animation resolution over a loaded
//! [`re_level::Level`]: the consumer for the `.lvl` `[emit SOURCE=1] parts = …`
//! clause. No seams.
//!
//! ## What the level authors
//!
//! A level declares one or more voxel `[part]` stamps plus an `[emit]` clause
//! binding each to a SOURCE NODE:
//!
//! ```text
//! [part red_block]
//! size   = 2 2 2
//! mask   =
//! ##
//! ##
//!
//! ##
//! ##
//! color  = #FF4040
//! offset = 0 0 0
//!
//! [emit output=1]
//! parts  = red_block
//! ```
//!
//! `re_level` parses this into [`re_level::Level::parts`] + [`re_level::Level::emits`]
//! and resolves the source name to a node index — but on its own nothing consumes
//! it. This module is that consumer: firing the source node materialises the part
//! in the world.
//!
//! ## The rule
//!
//! When the emit clause's source node's activation is `1`, the part's solid mask
//! cells are stamped into the world as [`CellKind::WallSolid`] carrying a synthesized
//! COLOUR-MATERIAL id in `side_data`, so the renderer shades them from the part's
//! authored `color`. When the activation falls back to `0`, exactly those cells are
//! restored. Nothing else changes — this module adds no puzzle rule, it only makes
//! existing puzzle state visible.
//!
//! ## Walk-through
//!
//! A materialised figure is light/projection, not furniture — the player walks
//! straight through it. Every stamp also marks the cell
//! [`Map::mark_walkthrough`], and every un-stamp clears the mark, so
//! `re_world::CameraPose`'s collision/gravity code treats it as air while
//! `cell_at`/`is_solid` (rendering) still see `WallSolid` — the look is
//! unchanged, only physical collision is.
//!
//! ## Where a part lands (the anchor rule)
//!
//! A `[part]` carries a size + mask + an `offset`, but no absolute position; the
//! `[pedestal]` is its anchor (that is the whole authored staging — the figure
//! materialises ON the pedestal). The mapping is:
//!
//! ```text
//! world = (ped.cx - ped.w/2 + off_x + px,
//!          ped.cy - ped.h/2 + off_y + py,
//!          ped.cz           + off_z + pz)
//! ```
//!
//! X/Y take the pedestal bounding box's MIN corner (`centre − extent/2`, the ordinary
//! centre+extent bbox convention, which is also what keeps L1's 2×2×2 block clear of
//! the door cell one step further `+x`). Z takes the pedestal's declared `cz`
//! directly rather than `cz − d/2`: every campaign level authors its pedestal centred
//! on the STANDING layer (`cz = 1`, or `cz = 5` on L1's second storey) with `d = 2`,
//! so the literal bbox minimum would be the floor slab and the whole base layer of
//! every two-layer figure (`offset z = 0` + `offset z = 1` — L2..L9 all use this) would
//! be swallowed by geometry. Anchoring at `cz` puts layer 0 on the deck and layer 1
//! above it, which is exactly the authored silhouette.
//!
//! ## Validate before mutate
//!
//! A part cell is bound at LOAD only if it is inside the room AND currently
//! [`CellKind::Empty`]. So a stamped figure can never clobber a wall, a node, a door,
//! a pipe trace, or a note placard — the same non-destructive rule
//! `re_level::stamp_pipes` uses. At runtime the stamp re-checks that the cell is still
//! Empty before writing, and an un-stamp restores ONLY the cells this clause actually
//! wrote (so two clauses whose parts overlap — L11/L12 author three/two beacons at the
//! same pedestal offset — resolve deterministically to first-lit-wins and neither one
//! erases the other's cell).
//!
//! ## Dirty flag
//!
//! [`emit_parts_sys`] reads the puzzle and the world through a shared borrow and only
//! takes `&mut Map` when a clause's lit state actually flips. So an idle frame never
//! marks the [`Map`] resource changed, and a frame that DOES change the world marks it
//! exactly once — which is the signal `re_render`'s composite already keys its
//! structural near-dirty + material-registry rebuild off (`map.is_changed()`).
//!
//! ## Reset
//!
//! [`EmitParts`] is rebuilt from scratch by every level load and cleared by level
//! teardown, so a reload starts with every part unlit and no stale clause can address
//! the next level's map.

use bevy::prelude::*;
use re_level::Level;
use re_puzzle::PuzzleState;
use re_world::{CellKind, Map, MaterialAnim, MaterialAnimKind, MaterialDef};

/// The five surface-kind material slots every level's material list starts with; the
/// per-cell colour palette follows them (`re_level::renderer_material_defs`). Mirrors
/// [`crate::notes`]'s constant of the same name (the note-placard binder) — both
/// append to the same palette.
const SURFACE_SLOTS: usize = 5;

/// Sentinel prefix on a synthesized emit-part material's name.
///
/// It can never collide with an authored `[material NAME]` (or with `level_text`'s note
/// materials) and is obvious in a material dump. Unit-separator framed, like
/// `re_level`'s preset prefix.
pub const EMIT_MATERIAL_PREFIX: &str = "\u{1f}emit\u{1f}";

/// The glyph a materialised part cell inks: a full block, so the authored `color`
/// carries the whole read and the figure holds its silhouette at any distance.
const EMIT_GLYPH: u32 = '█' as u32;

/// Fraction of the authored colour used for the cell background, so the block reads as
/// a lit volume with body rather than a flat sticker.
const EMIT_BG_SCALE: f32 = 0.22;

/// The materialised block BREATHES (a slow brightness pulse) — the authored beat is
/// "it lights up", not "a box appears". Cheap: these levels already carry animated
/// puzzle-node materials, so the near tier's animated-map classification is unchanged.
const EMIT_PULSE_SPEED: f32 = 1.1;
const EMIT_PULSE_AMPLITUDE: f32 = 0.35;

/// One bound `[emit]` clause: the node that lights it, the colour-material id its
/// cells shade from, and the world cells it owns.
#[derive(Clone, Debug)]
pub struct EmitClauseBinding {
    /// Index into [`PuzzleState::nodes`] — the source whose `activation == 1` lights
    /// this clause.
    pub source_node: usize,
    /// 1-based colour-material palette id (resolves to `Map::materials()[4 + id]`).
    pub color_id: u16,
    /// Every world cell of the part's solid mask that was in-bounds AND empty air at
    /// level load, with the `side_data` that cell carried then (restored on un-stamp).
    pub cells: Vec<(usize, usize, usize, u16)>,
    /// Whether this clause is currently materialised.
    lit: bool,
    /// The cells this clause actually wrote when it lit — the exact set an un-stamp
    /// restores. A subset of `cells` (a cell another clause already claimed is skipped).
    stamped: Vec<(usize, usize, usize, u16)>,
}

impl EmitClauseBinding {
    /// Is this clause currently materialised in the world?
    #[must_use]
    pub const fn is_lit(&self) -> bool {
        self.lit
    }

    /// How many cells this clause has currently stamped (`0` when unlit).
    #[must_use]
    pub const fn stamped_len(&self) -> usize {
        self.stamped.len()
    }
}

/// The level's bound `[emit]` clauses.
///
/// Rebuilt by every level load, cleared by teardown. A level with no `[emit]` section
/// binds nothing and the system is a no-op — byte-identical to before this module
/// existed.
#[derive(Resource, Default, Debug)]
pub struct EmitParts {
    /// One binding per authored `[emit]` clause, in level-authoring order.
    pub clauses: Vec<EmitClauseBinding>,
}

impl EmitParts {
    /// True when no clause is bound (an emit-less level, or a torn-down one).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Total cells currently materialised across every clause. Test/diagnostic hook.
    #[must_use]
    pub fn lit_cell_count(&self) -> usize {
        self.clauses
            .iter()
            .map(EmitClauseBinding::stamped_len)
            .sum()
    }
}

/// Scale a packed `0x00RRGGBB` colour per channel. Used for the block's darker body
/// background.
fn scale_rgb(rgb: u32, k: f32) -> u32 {
    let ch = |shift: u32| {
        let v = ((rgb >> shift) & 0xFF) as f32 * k;
        (v.round().clamp(0.0, 255.0) as u32) << shift
    };
    ch(16) | ch(8) | ch(0)
}

/// The material-name suffix that carries a part's [`re_level::PartLook`] to the renderer
/// ([`re_render::zband`]): empty for the default look, so every figure that authors no
/// `height` / `top` keeps its exact material name (and frame).
fn look_suffix(look: re_level::PartLook) -> String {
    if look.height >= 1.0 && look.top.is_none() {
        return String::new();
    }
    re_render::zband::suffix(0.0, look.height, look.top)
}

/// Resolve one `[part]`'s solid mask into absolute world cells, per the anchor rule in
/// the module docs. Pure — no map access, no clipping: the caller filters.
fn part_cells(level: &Level, part_idx: usize) -> Vec<(i32, i32, i32)> {
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
    // Z anchors on the pedestal's DECLARED z (its standing surface), not on the bbox
    // minimum — see the module docs for why.
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

/// Bind this level's `[emit]` clauses against `map`: append one synthesized colour
/// material per clause and resolve each clause's part cells.
///
/// Call at EVERY level-load site, after `level.build_map()` (and after the host's
/// `level_text::bind_level_text`, which appends to the same palette) and BEFORE
/// `re_render::registry_for_map(&map)` — the emit materials must already be on the
/// map when the render registry is derived.
///
/// Nothing is stamped here: a freshly-bound clause is UNLIT. [`emit_parts_sys`] lights
/// it on the first frame if its source node is already active at load, which keeps the
/// "activation drives it" rule single-sourced.
pub fn bind_emit_parts(level: &Level, map: &mut Map) -> EmitParts {
    let mut bound = EmitParts::default();
    if level.emits.is_empty() {
        return bound;
    }
    let (sx, sy, sz) = map.size();
    // The colour-material palette is `materials()[5..]`, in 1-based colour-id order
    // (id 1 ⇒ index 5). Appending keeps every existing id stable.
    let mut mats: Vec<MaterialDef> = map.materials().to_vec();
    // A map with no authored materials has an EMPTY list; the palette is defined
    // relative to the five surface slots, so those must exist before we append.
    if mats.len() < SURFACE_SLOTS {
        mats.resize(SURFACE_SLOTS, MaterialDef::fallback());
    }

    for (ci, emit) in level.emits.iter().enumerate() {
        let cells: Vec<(usize, usize, usize, u16)> = part_cells(level, emit.part)
            .into_iter()
            .filter(|&(x, y, z)| x >= 0 && y >= 0 && z >= 0)
            .filter(|&(x, y, z)| (x as usize) < sx && (y as usize) < sy && (z as usize) < sz)
            // NON-DESTRUCTIVE: only genuine air is ever claimed, so a part can never
            // replace a wall / node / door / pipe trace / note placard.
            .filter(|&(x, y, z)| map.cell_at(x, y, z) == CellKind::Empty)
            // …and never a cell RESERVED by the pipe network. A pipe path cell keeps its
            // `side_data` pipe tag even where a later `[cells]` pass demoted the trace
            // back to air (L1), so the tag — not just the kind — is what marks the cell
            // as spoken for. Claiming one would put the reward on top of the wiring.
            .filter(|&(x, y, z)| re_core::decode_pipe_tag(map.side_data_at(x, y, z)).is_none())
            .map(|(x, y, z)| {
                (
                    x as usize,
                    y as usize,
                    z as usize,
                    map.side_data_at(x, y, z),
                )
            })
            .collect();
        if cells.is_empty() {
            // Nothing to show (a part entirely out of the room or fully buried). Skip
            // the clause AND its material so the palette carries no dead slot.
            continue;
        }
        let color = level
            .parts
            .get(emit.part)
            .map_or(0x00FF_FFFF, |p| p.color_rgb & 0x00FF_FFFF);
        mats.push(MaterialDef {
            name: format!(
                "{EMIT_MATERIAL_PREFIX}{ci}{}",
                look_suffix(level.part_look(emit.part))
            ),
            fg: color,
            bg: scale_rgb(color, EMIT_BG_SCALE),
            glyph: Some(EMIT_GLYPH),
            // No recipe: the colour-palette base (the brick built-in) supplies one, and
            // the fixed full-block glyph + flat palette make the recipe's ramp moot —
            // the block reads as a solid lit volume.
            recipe: None,
            anim: MaterialAnim {
                kind: MaterialAnimKind::Pulse,
                speed: EMIT_PULSE_SPEED,
                amplitude: EMIT_PULSE_AMPLITUDE,
            },
            rank: 0,
            reflective: false,
            clarity: MaterialDef::DEFAULT_CLARITY,
            fallback: false,
            detail: false,
        });
        // Colour id = position within the palette (1-based).
        let color_id = (mats.len() - SURFACE_SLOTS) as u16;
        bound.clauses.push(EmitClauseBinding {
            source_node: emit.source_node,
            color_id,
            cells,
            lit: false,
            stamped: Vec::new(),
        });
    }

    if !bound.clauses.is_empty() {
        map.set_materials(mats);
    }
    bound
}

/// Materialise / de-materialise `clause` in `map`. Pure given the inputs (no ECS), so
/// the whole rule is unit-testable. Returns `true` when the world actually changed.
fn apply_clause(clause: &mut EmitClauseBinding, map: &mut Map, want_lit: bool) -> bool {
    if clause.lit == want_lit {
        return false;
    }
    if want_lit {
        clause.stamped.clear();
        for &(x, y, z, sd) in &clause.cells {
            // Re-validate: another clause may already own this cell (overlapping
            // authored parts). First lit wins; we simply do not claim it.
            if map.cell_at(x as i32, y as i32, z as i32) != CellKind::Empty {
                continue;
            }
            map.set_cell(x, y, z, CellKind::WallSolid, clause.color_id);
            // Visual reward figure, not furniture: the player walks straight
            // through it. Rendering is unaffected — `cell_at`/`is_solid` still
            // see `WallSolid`.
            map.mark_walkthrough(x, y, z, true);
            clause.stamped.push((x, y, z, sd));
        }
        clause.lit = true;
        // Even a clause that claimed nothing this time is now "lit" — but only report a
        // world change when a cell really moved.
        !clause.stamped.is_empty()
    } else {
        let changed = !clause.stamped.is_empty();
        for &(x, y, z, sd) in &clause.stamped {
            map.set_cell(x, y, z, CellKind::Empty, sd);
            map.mark_walkthrough(x, y, z, false);
        }
        clause.stamped.clear();
        clause.lit = false;
        changed
    }
}

/// Drive every bound clause from `puzzle` into `map`. Pure given the inputs; the ECS
/// system below is a thin wrapper so this is directly unit-testable.
///
/// Returns `true` when the world changed (the caller marks the [`Map`] resource dirty
/// only then).
pub fn sync_emit_parts(parts: &mut EmitParts, puzzle: &PuzzleState, map: &mut Map) -> bool {
    let mut changed = false;
    for clause in &mut parts.clauses {
        let want = puzzle
            .nodes
            .get(clause.source_node)
            .is_some_and(|n| n.activation == 1);
        changed |= apply_clause(clause, map, want);
    }
    changed
}

/// Does any clause's lit state disagree with the live puzzle? Read-only — lets the
/// system decide whether to take `&mut Map` at all, so an idle frame never marks the
/// map resource changed (and so never forces a structural re-cast).
fn needs_sync(parts: &EmitParts, puzzle: &PuzzleState) -> bool {
    parts.clauses.iter().any(|c| {
        let want = puzzle
            .nodes
            .get(c.source_node)
            .is_some_and(|n| n.activation == 1);
        want != c.lit
    })
}

/// `Update` system: stamp / un-stamp the authored `[emit]` parts to follow their
/// source nodes' activation.
///
/// Ordering: AFTER `re_puzzle::apply_interactions` (which performs the toggle + the
/// forward pass), so this frame's world reflects this frame's puzzle.
pub fn emit_parts_sys(
    puzzle: Res<PuzzleState>,
    mut parts: ResMut<EmitParts>,
    mut map: ResMut<Map>,
) {
    if parts.clauses.is_empty() {
        return;
    }
    // Read-only pre-check so an idle frame never touches `ResMut<Map>`'s change tick.
    if !needs_sync(&parts, &puzzle) {
        return;
    }
    // `parts` must advance its lit flags regardless, but that is bookkeeping no other
    // system reads — bypass its change tick. The MAP is deref-mutated inside
    // `sync_emit_parts`, which is what marks the render dirty, and we only get here on a
    // frame we already know flips a clause.
    //
    // The return value is deliberately dropped: `false` means the flags flipped but no
    // cell actually moved (a clause whose every cell was already claimed by an
    // overlapping one), in which case the map was marked dirty for one redundant frame.
    // Harmless and rare; never showing a stale frame beats saving that one re-cast.
    let parts = parts.bypass_change_detection();
    let _ = sync_emit_parts(parts, &puzzle, &mut map);
}

/// Registers the [`EmitParts`] resource + [`emit_parts_sys`]. Added by the binary
/// AFTER `PuzzlePlugin` so the `after(apply_interactions)` ordering is well-formed.
pub struct EmitPartsPlugin;

impl Plugin for EmitPartsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EmitParts>()
            .add_systems(Update, emit_parts_sys.after(re_puzzle::apply_interactions));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal level with one input node, one output node, one switchable pipe, a
    /// pedestal, and an `[emit]` clause lighting a 2×1×1 part.
    const EMIT_LEVEL: &str = "\
[meta]
name = emit-test
[room]
size = 10 8 4
ambient = solid
[node in]
pos = 2,4,1
glyph = I
is_input = yes
[node out]
pos = 7,4,1
glyph = O
is_output = yes
threshold = 1
[pipe in->out]
state = OFF
path = (2,4,1) (7,4,1)
switch_at = (4,4,1)
[part slab]
size = 2 1 1
mask =
##
color = #FF4040
offset = 0 0 0
[emit out=1]
parts = slab
[inputs]
in = 1
[pedestal]
center = 5 4 1
volume = 2 1 1
[door]
pos = 8 4 1
target_parts = out
[player]
spawn = 2 4 1
facing = +x
";

    fn load() -> (Level, Map) {
        let level =
            re_level::load_string(EMIT_LEVEL, "<emit_parts_test>").expect("test level parses");
        let map = level.build_map();
        (level, map)
    }

    #[test]
    fn part_cells_anchor_on_the_pedestal_min_corner_and_its_declared_z() {
        let (level, _) = load();
        // pedestal centre (5,4,1) extent (2,1,1) ⇒ x base 5 - 1 = 4, y base 4, z base 1.
        let cells = part_cells(&level, 0);
        assert_eq!(cells, vec![(4, 4, 1), (5, 4, 1)]);
    }

    #[test]
    fn binding_appends_one_colour_material_and_claims_only_empty_air() {
        let (level, mut map) = load();
        let before = map.materials().len();
        let bound = bind_emit_parts(&level, &mut map);
        assert_eq!(bound.clauses.len(), 1, "one [emit] clause bound");
        assert_eq!(
            map.materials().len(),
            before.max(SURFACE_SLOTS) + 1,
            "exactly one emit material appended"
        );
        let c = &bound.clauses[0];
        assert!(c.color_id >= 1, "a 1-based colour id was assigned");
        assert!(!c.is_lit(), "a freshly-bound clause is unlit");
        assert_eq!(c.cells.len(), 2, "both mask cells are empty air");
        for &(x, y, z, _) in &c.cells {
            assert_eq!(map.cell_at(x as i32, y as i32, z as i32), CellKind::Empty);
        }
    }

    #[test]
    fn binding_never_claims_a_non_empty_cell() {
        let (level, mut map) = load();
        // Wall off one of the two part cells before binding.
        map.set_cell(4, 4, 1, CellKind::WallSolid, 0);
        let bound = bind_emit_parts(&level, &mut map);
        assert_eq!(
            bound.clauses[0].cells,
            vec![(5, 4, 1, 0)],
            "the occupied cell is not claimed"
        );
    }

    #[test]
    fn firing_the_source_stamps_the_part_and_unfiring_removes_it() {
        let (level, mut map) = load();
        let mut parts = bind_emit_parts(&level, &mut map);
        let mut puzzle = PuzzleState::from_level(&level).expect("puzzle builds");

        // The pipe is authored OFF, so `out` has not fired: nothing is materialised.
        assert_eq!(puzzle.nodes[1].activation, 0);
        assert!(!sync_emit_parts(&mut parts, &puzzle, &mut map));
        assert_eq!(map.cell_at(4, 4, 1), CellKind::Empty);
        assert_eq!(parts.lit_cell_count(), 0);

        // Fire the output: the part materialises as coloured solid cells.
        puzzle.nodes[1].activation = 1;
        assert!(
            sync_emit_parts(&mut parts, &puzzle, &mut map),
            "world changed"
        );
        let color_id = parts.clauses[0].color_id;
        for &(x, y, z, _) in &parts.clauses[0].cells {
            assert_eq!(
                map.cell_at(x as i32, y as i32, z as i32),
                CellKind::WallSolid,
                "part cell materialised"
            );
            assert_eq!(
                map.side_data_at(x as i32, y as i32, z as i32),
                color_id,
                "part cell carries its colour-material id"
            );
        }
        assert_eq!(parts.lit_cell_count(), 2);

        // Idle: no further change (and so nothing marks the map dirty).
        assert!(!sync_emit_parts(&mut parts, &puzzle, &mut map));

        // Un-fire: exactly those cells go back to empty air with their original tags.
        puzzle.nodes[1].activation = 0;
        assert!(
            sync_emit_parts(&mut parts, &puzzle, &mut map),
            "world changed"
        );
        for &(x, y, z, sd) in &parts.clauses[0].cells {
            assert_eq!(map.cell_at(x as i32, y as i32, z as i32), CellKind::Empty);
            assert_eq!(map.side_data_at(x as i32, y as i32, z as i32), sd);
        }
        assert_eq!(parts.lit_cell_count(), 0);
    }

    #[test]
    fn a_lit_figure_is_walk_through_and_unlighting_clears_the_mark() {
        let (level, mut map) = load();
        let mut parts = bind_emit_parts(&level, &mut map);
        let mut puzzle = PuzzleState::from_level(&level).expect("puzzle builds");

        // Unlit: never marked (nothing to mark — cells are Empty).
        for &(x, y, z, _) in &parts.clauses[0].cells {
            assert!(!map.is_walkthrough(x as i32, y as i32, z as i32));
        }

        puzzle.nodes[1].activation = 1;
        sync_emit_parts(&mut parts, &puzzle, &mut map);
        for &(x, y, z, _) in &parts.clauses[0].cells {
            let (x, y, z) = (x as i32, y as i32, z as i32);
            assert!(
                map.is_walkthrough(x, y, z),
                "lit figure cell ({x},{y},{z}) must be walk-through"
            );
            // Rendering still sees the real, solid kind.
            assert_eq!(map.cell_at(x, y, z), CellKind::WallSolid);
            assert!(map.cell_at(x, y, z).is_solid());
            // Physics sees air.
            assert_eq!(map.physics_cell_at(x, y, z), CellKind::Empty);
        }

        puzzle.nodes[1].activation = 0;
        sync_emit_parts(&mut parts, &puzzle, &mut map);
        for &(x, y, z, _) in &parts.clauses[0].cells {
            assert!(
                !map.is_walkthrough(x as i32, y as i32, z as i32),
                "un-firing must clear the walk-through mark"
            );
        }
    }

    #[test]
    fn an_unbound_source_node_never_lights() {
        let (level, mut map) = load();
        let mut parts = bind_emit_parts(&level, &mut map);
        // Point the clause at a node index that does not exist.
        parts.clauses[0].source_node = 99;
        let puzzle = PuzzleState::from_level(&level).expect("puzzle builds");
        assert!(!sync_emit_parts(&mut parts, &puzzle, &mut map));
        assert_eq!(parts.lit_cell_count(), 0);
    }

    #[test]
    fn a_level_with_no_emit_binds_nothing_and_leaves_the_map_alone() {
        let src = EMIT_LEVEL.replace("[emit out=1]\nparts = slab\n", "");
        let level =
            re_level::load_string(&src, "<emit_parts_test>").expect("emit-less level parses");
        let mut map = level.build_map();
        let before = map.materials().len();
        let bound = bind_emit_parts(&level, &mut map);
        assert!(bound.is_empty());
        assert_eq!(map.materials().len(), before, "no material appended");
    }

    #[test]
    fn overlapping_clauses_resolve_first_lit_wins_and_do_not_erase_each_other() {
        let (level, mut map) = load();
        let mut parts = bind_emit_parts(&level, &mut map);
        // Duplicate the clause so two of them claim the SAME two cells.
        let mut dup = parts.clauses[0].clone();
        dup.color_id = parts.clauses[0].color_id + 1;
        parts.clauses.push(dup);
        let mut puzzle = PuzzleState::from_level(&level).expect("puzzle builds");
        puzzle.nodes[1].activation = 1;

        sync_emit_parts(&mut parts, &puzzle, &mut map);
        assert_eq!(
            parts.clauses[0].stamped_len(),
            2,
            "the first clause claimed both"
        );
        assert_eq!(parts.clauses[1].stamped_len(), 0, "the second claimed none");

        // Turning them off restores the world exactly — no cell is left stamped and no
        // cell is double-cleared.
        puzzle.nodes[1].activation = 0;
        sync_emit_parts(&mut parts, &puzzle, &mut map);
        assert_eq!(map.cell_at(4, 4, 1), CellKind::Empty);
        assert_eq!(map.cell_at(5, 4, 1), CellKind::Empty);
        assert_eq!(parts.lit_cell_count(), 0);
    }

    #[test]
    fn rebinding_a_fresh_load_starts_unlit_the_reset_contract() {
        // The reset path: teardown drops `EmitParts`, a reload re-binds from a fresh
        // map. Prove the rebuilt binding is unlit and the fresh map carries no stamp.
        let (level, mut map) = load();
        let mut parts = bind_emit_parts(&level, &mut map);
        let mut puzzle = PuzzleState::from_level(&level).expect("puzzle builds");
        puzzle.nodes[1].activation = 1;
        sync_emit_parts(&mut parts, &puzzle, &mut map);
        assert_eq!(parts.lit_cell_count(), 2);

        // Reload.
        parts = EmitParts::default();
        assert!(parts.is_empty(), "teardown clears the bindings");
        let (level2, mut map2) = load();
        let parts2 = bind_emit_parts(&level2, &mut map2);
        assert_eq!(parts2.lit_cell_count(), 0, "a reloaded level starts unlit");
        assert_eq!(map2.cell_at(4, 4, 1), CellKind::Empty);
        assert_eq!(map2.cell_at(5, 4, 1), CellKind::Empty);
    }

    #[test]
    fn scale_rgb_darkens_each_channel() {
        assert_eq!(scale_rgb(0x00FF_8040, 0.5), 0x0080_4020);
        assert_eq!(scale_rgb(0x0000_0000, 0.5), 0);
    }

    // ── the real ECS seam: the plugin's system, driven by a live Bevy App ──

    /// Records whether the [`Map`] resource was seen as CHANGED this frame — the exact
    /// signal `re_render`'s composite keys its structural near-dirty + material-registry
    /// rebuild off (`map.is_changed()`). Read from a system so the change tick is
    /// evaluated the way the renderer evaluates it, not from outside the schedule.
    #[derive(Resource, Default)]
    struct MapDirtySeen(bool);

    fn record_map_dirty(map: Res<Map>, mut seen: ResMut<MapDirtySeen>) {
        seen.0 = map.is_changed();
    }

    /// A tiny app wired like the play state: puzzle + map + the emit-parts plugin, plus
    /// the dirty-flag recorder ordered after the emit system (as the renderer is).
    fn emit_app(map: Map, parts: EmitParts, puzzle: PuzzleState) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(re_puzzle::PuzzlePlugin)
            .add_plugins(EmitPartsPlugin);
        app.insert_resource(map);
        app.insert_resource(parts);
        app.insert_resource(puzzle);
        app.init_resource::<MapDirtySeen>();
        app.add_systems(Update, record_map_dirty.after(emit_parts_sys));
        app
    }

    /// End-to-end through the SYSTEM (not the pure helper): a source node that reads 1
    /// materialises its part on the next frame, an idle frame leaves the `Map` resource
    /// UNCHANGED (so the renderer's `map.is_changed()` dirty-flag stays quiet), and the
    /// frame that flips it DOES mark the map changed.
    #[test]
    fn the_system_stamps_on_fire_and_only_dirties_the_map_when_it_changes() {
        let (level, mut map) = load();
        let parts = bind_emit_parts(&level, &mut map);
        let cells = parts.clauses[0].cells.clone();
        let puzzle = PuzzleState::from_level(&level).expect("puzzle builds");
        let mut app = emit_app(map, parts, puzzle);

        // Frame 1: the source is cold (the pipe is authored OFF) ⇒ nothing stamped.
        app.update();
        for &(x, y, z, _) in &cells {
            assert_eq!(
                app.world()
                    .resource::<Map>()
                    .cell_at(x as i32, y as i32, z as i32),
                CellKind::Empty,
                "nothing materialises while the source is cold"
            );
        }
        // Frame 2, still idle: the map must be quiet (frame 1 sees the insert itself).
        app.update();
        assert!(
            !app.world().resource::<MapDirtySeen>().0,
            "an idle frame must not mark the Map changed"
        );

        // Fire the output, then run a frame: the part appears AND the map is marked
        // changed (the render's structural dirty signal).
        app.world_mut().resource_mut::<PuzzleState>().nodes[1].activation = 1;
        app.update();
        assert!(
            app.world().resource::<MapDirtySeen>().0,
            "the frame that materialises a part must mark the Map changed"
        );
        for &(x, y, z, _) in &cells {
            assert_eq!(
                app.world()
                    .resource::<Map>()
                    .cell_at(x as i32, y as i32, z as i32),
                CellKind::WallSolid,
                "the part materialised"
            );
        }

        // An IDLE frame afterwards must NOT re-dirty the map.
        app.update();
        assert!(
            !app.world().resource::<MapDirtySeen>().0,
            "an idle frame after the stamp must not re-dirty the Map"
        );
    }

    // NB: the "every shipped campaign level's `[emit]` clauses bind to real cells" check
    // lives in `tests/campaign_pipes_and_emits.rs` — it reads `levels/`, so it belongs
    // with the other level-DATA assertions rather than in this module's pure unit tests.
}
