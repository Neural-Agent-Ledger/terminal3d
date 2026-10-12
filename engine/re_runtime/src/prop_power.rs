//! `prop_power` — prop power-gate/power-source live wiring: the runtime
//! consumer of [`re_level::Level::prop_power`], props that react to the live
//! puzzle. Seams: [`crate::session::PlayCamera`] (the host's play-camera
//! marker) and [`crate::unstuck`].
//!
//! ## What the level hands us
//!
//! `re_level` expands every `[prop]` with a `power = node:N | pipe:A->B | door |
//! budget:K | seq:I` binding (plus `invert = yes`) into a [`PowerGate`]: the RESOLVED
//! source, the state the built map already shows (`initial_lit`), and the concrete
//! structures that differ between "lit" and "unlit":
//!
//! | gate field | lit | unlit | delivery |
//! |---|---|---|---|
//! | `parts` | `WallSolid` carrying the lit palette id | `Empty` | map cells |
//! | `lines` | the lit palette id | the ink palette id | `Map::set_line_palette_ids` |
//! | `holos` | `Hologram` cells stamped | `Empty` | map cells |
//! | `emitters` | the lit spec | the unlit spec | `re_render::EmitterSet` slot |
//! | `lights` | the authored lamp | the same lamp at `intensity = 0` | map atmosphere |
//!
//! The built map is in the STATIC initial state — `re_level` cannot run the forward
//! pass, so a hidden node, the door and sequence steps start at 0. The first frame
//! after load reconciles every gate with the real [`PuzzleState`].
//!
//! ## Source value
//!
//! `s ∈ {0,1}` per [`source_value`]: a node's `activation == 1`, a pipe `On`, the
//! network solved (the door opens), at least `K` pipes `On` (the same count the power
//! budget spends), `[sequence]` step `I` locked. `invert` flips it.
//!
//! ## Edges only, events first
//!
//! [`prop_power_sys`] returns immediately unless [`PuzzleState`] (or the binding
//! itself, i.e. a fresh load) changed this frame — a toggle, a reset, the forward
//! pass. When it does run it first checks, read-only, whether any gate's wanted state
//! differs from its current one; only then does it write. The [`Map`] resource is
//! marked changed ONLY when a gate that touches the map actually flipped, and the
//! [`re_render::EmitterSet`] only when an emitter slot was swapped — so an idle frame,
//! or a toggle that changes nothing a prop listens to, never forces a re-cast.
//!
//! Every flip is also published as a [`PropPowerEdge`] message (the audio / FX hook).
//!
//! ## Non-destructive
//!
//! A lit slot (part or hologram cell) is bound only if at load it holds exactly what
//! this gate's initial state says (Empty when unlit; the gate's own stamp when lit).
//! Stamping re-validates that the cell is still Empty, and un-stamping restores only
//! the cells this gate wrote — the `emit_parts` contract.
//!
//! ## Not delivered here
//!
//! Whole-body material swaps (a `mode = skin` panel going `dead` → `lit`) are
//! not carried by [`PowerGate`] yet, so there is nothing to apply for them.

use bevy::prelude::*;
use re_level::props::{PowerGate, PowerSource};
use re_level::Level;
use re_puzzle::PuzzleState;
use re_world::{CameraPose, CellKind, EmitterSpec, HoloObject, Map, PointLight};

/// One world cell a gate owns: what goes there when lit, and what `side_data` it
/// returns to when unlit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellSlot {
    x: usize,
    y: usize,
    z: usize,
    /// `WallSolid` for a lit part, `Hologram` for a holo cell.
    kind: CellKind,
    /// The palette id (part) / holo style (hologram) written while lit.
    id: u16,
    /// The `side_data` the cell carries while Empty.
    restore: u16,
}

/// One bound gate: the level's [`PowerGate`] resolved against the live map.
#[derive(Clone, Debug)]
pub struct GateBinding {
    /// The owning `[prop NAME]`.
    pub prop: String,
    /// What this gate reads to decide lit/unlit (a node, a pipe, the door, …).
    pub source: PowerSource,
    /// True flips the source's raw reading before it drives `lit` — an
    /// authored `invert = yes` makes the gate light when its source is OFF.
    pub invert: bool,
    /// The state the world currently shows.
    lit: bool,
    /// Every cell this gate may occupy (parts, then hologram cells).
    slots: Vec<CellSlot>,
    /// Indices into `slots` this gate currently holds stamped.
    stamped: Vec<usize>,
    /// `(map line index, unlit palette id, lit palette id)`.
    lines: Vec<(usize, u16, u16)>,
    /// `(EmitterSet index, unlit spec, lit spec)`.
    emitters: Vec<(usize, EmitterSpec, EmitterSpec)>,
    /// `(atmosphere light index, the LIT lamp)`.
    lights: Vec<(usize, PointLight)>,
    /// Where the prop is (for positioned cues), if it has any geometry at all.
    pub anchor: Option<[f32; 3]>,
}

impl GateBinding {
    /// Is the gate currently showing its lit state?
    #[must_use]
    pub const fn is_lit(&self) -> bool {
        self.lit
    }

    /// Cells currently stamped by this gate.
    #[must_use]
    pub const fn stamped_len(&self) -> usize {
        self.stamped.len()
    }

    /// Cells this gate owns (lit or not).
    #[must_use]
    pub const fn slot_len(&self) -> usize {
        self.slots.len()
    }
}

/// The level's bound power gates. Rebuilt by every level load, cleared by teardown;
/// empty ⇒ [`prop_power_sys`] is a no-op.
#[derive(Resource, Default, Debug)]
pub struct PropPower {
    /// One binding per authored `[prop NAME]` power gate, in level-authoring
    /// order.
    pub gates: Vec<GateBinding>,
    /// The level's authored `[light]`s — what an emitter is rebuilt against on a
    /// swap (mirrors how a host's own level-enter handler builds an emitter set).
    emitter_lights: Vec<PointLight>,
}

impl PropPower {
    /// True when no gate is bound (a gate-less level, or a torn-down one).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Gates currently lit. Diagnostic / test hook.
    #[must_use]
    pub fn lit_count(&self) -> usize {
        self.gates.iter().filter(|g| g.lit).count()
    }
}

/// Published once per gate flip, in the frame it happens.
#[derive(Message, Clone, Debug, PartialEq)]
pub struct PropPowerEdge {
    /// Index into [`PropPower::gates`].
    pub gate: usize,
    /// The owning `[prop NAME]`, copied out for a subscriber that has no
    /// [`PropPower`] resource handy.
    pub prop: String,
    /// The gate's new state after this flip.
    pub lit: bool,
    /// Where the prop is (for positioned cues), if it has any geometry at all.
    pub at: Option<[f32; 3]>,
}

/// What one [`sync_prop_power`] call did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PowerSync {
    /// A cell, a line colour or a lamp changed — the render must re-cast.
    pub map_changed: bool,
    /// An emitter slot was swapped.
    pub emitters_changed: bool,
    /// `(gate index, now lit)` for every gate that flipped.
    pub edges: Vec<(usize, bool)>,
}

/// The live value of a power source, before `invert`.
#[must_use]
pub fn source_value(src: PowerSource, puzzle: &PuzzleState) -> bool {
    match src {
        PowerSource::Node(i) => puzzle.nodes.get(i).is_some_and(|n| n.activation == 1),
        PowerSource::Pipe(i) => puzzle.pipes.get(i).is_some_and(re_puzzle::Pipe::is_on),
        PowerSource::Door => puzzle.solved,
        PowerSource::Budget(k) => puzzle.power_used() >= k,
        PowerSource::Seq(i) => puzzle.sequence.progress as u64 >= u64::from(i),
    }
}

/// The state a gate should show for `puzzle`.
fn wanted(g: &GateBinding, puzzle: &PuzzleState) -> bool {
    source_value(g.source, puzzle) != g.invert
}

/// The world cells a hologram fills, via `re_world::stamp_holos` on a scratch map
/// the size of the object's box (the shape test itself is private to `re_world`).
fn holo_cells(h: &HoloObject) -> Vec<(i32, i32, i32)> {
    let (x0, x1) = (h.x0.min(h.x1), h.x0.max(h.x1));
    let (y0, y1) = (h.y0.min(h.y1), h.y0.max(h.y1));
    let (z0, z1) = (h.z0.min(h.z1), h.z0.max(h.z1));
    let (w, d, t) = (
        (x1 - x0 + 1) as usize,
        (y1 - y0 + 1) as usize,
        (z1 - z0 + 1) as usize,
    );
    let mut scratch = Map::new(w, d, t);
    let local = HoloObject {
        x0: 0,
        y0: 0,
        z0: 0,
        x1: x1 - x0,
        y1: y1 - y0,
        z1: z1 - z0,
        ..*h
    };
    re_world::stamp_holos(&mut scratch, &[local]);
    let mut out = Vec::new();
    for z in 0..t as i32 {
        for y in 0..d as i32 {
            for x in 0..w as i32 {
                if scratch.cell_at(x, y, z) == CellKind::Hologram {
                    out.push((x0 + x, y0 + y, z0 + z));
                }
            }
        }
    }
    out
}

/// Claim one slot against the map as built in `lit` state. Returns the slot and
/// whether it is currently stamped, or `None` when the cell is not ours.
fn claim(
    map: &Map,
    (x, y, z): (i32, i32, i32),
    kind: CellKind,
    id: u16,
    lit: bool,
) -> Option<(CellSlot, bool)> {
    if x < 0 || y < 0 || z < 0 || !map.in_bounds(x, y, z) {
        return None;
    }
    let here = map.cell_at(x, y, z);
    let sd = map.side_data_at(x, y, z);
    let slot = |restore| CellSlot {
        x: x as usize,
        y: y as usize,
        z: z as usize,
        kind,
        id,
        restore,
    };
    if lit && here == kind && sd == id {
        // Stamped by `re_level` at build; the air under it carried no tag (props
        // never claim a pipe-tagged cell — keep-out).
        Some((slot(0), true))
    } else if !lit && here == CellKind::Empty && re_core::decode_pipe_tag(sd).is_none() {
        Some((slot(sd), false))
    } else {
        None
    }
}

fn centroid(points: &[[f32; 3]]) -> Option<[f32; 3]> {
    if points.is_empty() {
        return None;
    }
    let n = points.len() as f32;
    let s = points
        .iter()
        .fold([0.0f32; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]);
    Some([s[0] / n, s[1] / n, s[2] / n])
}

/// Bind `level.prop_power` against the map `level.build_map()` produced.
///
/// Call after the text / emit binders — they only append palette slots, which leaves
/// every prop id stable. Nothing is written: the map already shows each gate's `initial_lit`
/// state, which is what the binding starts in; [`sync_prop_power`] reconciles it with
/// the live puzzle on the first frame.
#[must_use]
pub fn bind_prop_power(level: &Level, map: &Map) -> PropPower {
    let mut out = PropPower {
        gates: Vec::new(),
        emitter_lights: level.lights.clone(),
    };
    for g in &level.prop_power {
        out.gates.push(bind_gate(level, map, g));
    }
    out
}

fn bind_gate(level: &Level, map: &Map, g: &PowerGate) -> GateBinding {
    let lit = g.initial_lit;
    let mut slots = Vec::new();
    let mut stamped = Vec::new();
    let mut anchor_pts: Vec<[f32; 3]> = Vec::new();
    let mut push = |c: (i32, i32, i32), kind: CellKind, id: u16| {
        if let Some((slot, on)) = claim(map, c, kind, id, lit) {
            if on {
                stamped.push(slots.len());
            }
            slots.push(slot);
            anchor_pts.push([c.0 as f32 + 0.5, c.1 as f32 + 0.5, c.2 as f32 + 0.5]);
        }
    };
    for &(c, id) in &g.parts {
        push(c, CellKind::WallSolid, id);
    }
    for h in &g.holos {
        for c in holo_cells(h) {
            push(c, CellKind::Hologram, h.style);
        }
    }
    let n_lines = map.overlay().lines.len();
    let lines: Vec<(usize, u16, u16)> = g
        .lines
        .iter()
        .copied()
        .filter(|&(i, _, _)| i < n_lines)
        .collect();
    for &(i, _, _) in &lines {
        if let Some(p) = map.overlay().lines[i].path.points.first() {
            anchor_pts.push([p.x, p.y, p.z]);
        }
    }
    // Emitters: the level carries the INITIAL spec at the index, the gate the other.
    let emitters = g
        .emitters
        .iter()
        .filter_map(|(i, other)| {
            let now = level.emitters.get(*i)?.clone();
            Some(if lit {
                (*i, other.clone(), now)
            } else {
                (*i, now, other.clone())
            })
        })
        .collect();
    let n_lights = map.atmosphere().lights.len();
    let lights: Vec<(usize, PointLight)> = g
        .lights
        .iter()
        .copied()
        .filter(|&(i, _)| i < n_lights)
        .collect();
    for (_, l) in &lights {
        anchor_pts.push([l.pos.x, l.pos.y, l.pos.z]);
    }
    GateBinding {
        prop: g.prop.clone(),
        source: g.source,
        invert: g.invert,
        lit,
        slots,
        stamped,
        lines,
        emitters,
        lights,
        anchor: centroid(&anchor_pts),
    }
}

/// Write gate `g`'s `want` state into the map. Returns whether anything changed.
fn apply_map(g: &mut GateBinding, map: &mut Map, want: bool) -> bool {
    let mut changed = false;
    if want {
        g.stamped.clear();
        for (i, s) in g.slots.iter().enumerate() {
            // Re-validate: something else may have taken the cell meanwhile.
            if map.cell_at(s.x as i32, s.y as i32, s.z as i32) != CellKind::Empty {
                continue;
            }
            map.set_cell(s.x, s.y, s.z, s.kind, s.id);
            // A lit cell (a rack blade, a bank lamp, a reward figure) is
            // light/projection, not furniture — walk-through. Holograms are
            // already non-collidable by kind, so marking is scoped to the
            // `WallSolid` slots only.
            if s.kind == CellKind::WallSolid {
                map.mark_walkthrough(s.x, s.y, s.z, true);
            }
            g.stamped.push(i);
            changed = true;
        }
    } else {
        for &i in &g.stamped {
            let s = g.slots[i];
            map.set_cell(s.x, s.y, s.z, CellKind::Empty, s.restore);
            if s.kind == CellKind::WallSolid {
                map.mark_walkthrough(s.x, s.y, s.z, false);
            }
            changed = true;
        }
        g.stamped.clear();
    }
    if !g.lines.is_empty() {
        let n = map.overlay().lines.len();
        let mut ids: Vec<u16> = (0..n).map(|i| map.line_palette_id(i)).collect();
        let mut line_changed = false;
        for &(i, unlit, lit) in &g.lines {
            let id = if want { lit } else { unlit };
            if ids[i] != id {
                ids[i] = id;
                line_changed = true;
            }
        }
        if line_changed {
            map.set_line_palette_ids(ids);
            changed = true;
        }
    }
    if !g.lights.is_empty() {
        let mut atmo = map.atmosphere().clone();
        let mut light_changed = false;
        for &(i, lamp) in &g.lights {
            let l = if want {
                lamp
            } else {
                PointLight {
                    intensity: 0.0,
                    ..lamp
                }
            };
            if atmo.lights[i] != l {
                atmo.lights[i] = l;
                light_changed = true;
            }
        }
        if light_changed {
            // The renderer bakes its `LightEnv` from the map's atmosphere on every
            // cast (`re_render::light_env_for_map`), so a changed atmosphere plus the
            // map dirty flag IS the re-bake.
            map.set_atmosphere(atmo);
            changed = true;
        }
    }
    changed
}

/// Swap gate `g`'s emitter slots to the `want` spec. Returns whether any slot moved.
fn apply_emitters(
    g: &GateBinding,
    emitters: &mut [re_render::Emitter],
    lights: &[PointLight],
    want: bool,
) -> bool {
    let mut changed = false;
    for (i, unlit, lit) in &g.emitters {
        if let Some(slot) = emitters.get_mut(*i) {
            *slot = re_render::Emitter::from_spec_lit(if want { lit } else { unlit }, lights);
            changed = true;
        }
    }
    changed
}

/// Does any gate disagree with `puzzle`? Read-only.
#[must_use]
pub fn needs_sync(pp: &PropPower, puzzle: &PuzzleState) -> bool {
    pp.gates.iter().any(|g| wanted(g, puzzle) != g.lit)
}

/// Drive every gate from `puzzle`, acting only on edges.
///
/// Pure given the inputs (the ECS system is a thin wrapper), so the whole rule is
/// unit-testable. `emitters` is
/// the live particle set (`None` where there is none, e.g. a still dump).
pub fn sync_prop_power(
    pp: &mut PropPower,
    puzzle: &PuzzleState,
    map: &mut Map,
    mut emitters: Option<&mut Vec<re_render::Emitter>>,
) -> PowerSync {
    let mut out = PowerSync::default();
    let lights = std::mem::take(&mut pp.emitter_lights);
    for (gi, g) in pp.gates.iter_mut().enumerate() {
        let want = wanted(g, puzzle);
        if want == g.lit {
            continue;
        }
        out.map_changed |= apply_map(g, map, want);
        if let Some(set) = emitters.as_deref_mut() {
            out.emitters_changed |= apply_emitters(g, set, &lights, want);
        }
        g.lit = want;
        out.edges.push((gi, want));
    }
    pp.emitter_lights = lights;
    out
}

/// `Update` system: follow the live puzzle. Runs its body only on a frame where the
/// puzzle (or the binding — a fresh load) changed, and writes only on a real edge.
// Bevy system parameters are taken by value by construction; `Res` cannot be `&Res`.
#[allow(clippy::needless_pass_by_value)]
pub fn prop_power_sys(
    puzzle: Res<PuzzleState>,
    mut pp: ResMut<PropPower>,
    mut map: ResMut<Map>,
    emitters: Option<ResMut<re_render::EmitterSet>>,
    mut edges: MessageWriter<PropPowerEdge>,
    mut cams: Query<&mut CameraPose, With<crate::session::PlayCamera>>,
) {
    if pp.gates.is_empty() || !(puzzle.is_changed() || pp.is_changed()) {
        return;
    }
    if !needs_sync(&pp, &puzzle) {
        return;
    }
    let pp = pp.bypass_change_detection();
    // Change detection is re-armed by hand below, so a flip that only swaps an
    // emitter never marks the map dirty (and vice versa).
    let map_ref = map.bypass_change_detection();
    let mut emitters = emitters;
    let set = emitters
        .as_mut()
        .map(|e| &mut e.bypass_change_detection().emitters);
    let r = sync_prop_power(pp, &puzzle, map_ref, set);
    if r.map_changed {
        // RUNTIME SAFETY: a gate that just touched the map may — today only
        // via a not-yet-delivered mechanism, since every power-reactive cell
        // is marked walk-through — have left the player embedded in a
        // collider. Push them out deterministically rather than trust it.
        // Runs BEFORE `map.set_changed()` re-borrows `map` (the `map_ref`
        // borrow above must end first).
        for mut cam in &mut cams {
            crate::unstuck::ensure_camera_safe(map_ref, &mut cam);
        }
        map.set_changed();
    }
    if r.emitters_changed {
        if let Some(e) = emitters.as_mut() {
            e.set_changed();
        }
    }
    for (gi, lit) in r.edges {
        let g = &pp.gates[gi];
        edges.write(PropPowerEdge {
            gate: gi,
            prop: g.prop.clone(),
            lit,
            at: g.anchor,
        });
    }
}

/// Registers [`PropPower`], [`PropPowerEdge`] and [`prop_power_sys`]. Added AFTER
/// `PuzzlePlugin` so `after(apply_interactions)` is well-formed.
pub struct PropPowerPlugin;

impl Plugin for PropPowerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PropPower>()
            .add_message::<PropPowerEdge>()
            .add_systems(Update, prop_power_sys.after(re_puzzle::apply_interactions));
    }
}
