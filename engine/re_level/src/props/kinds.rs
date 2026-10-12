//! The 24 prop generators (set-dressing §3.8). Every function here is pure over
//! `(spec, level, prop-free map)`: it only pushes stamps into a [`BuiltProp`].

use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

use glam::{IVec3, Vec3};
use re_core::{RampDir, Shape};
use re_world::{CellKind, EmitterOrigin, EmitterSpec, HoloShape, Map, PointLight};

use super::{
    hash3, initial_value, lit_screen_material, pack_lit_rgb, pack_material, ramp_dir_of,
    resolve_power, shade, source_hue, unit3, wedge_toward, AbsPart, Cell3, CellMode, GalleryPosts,
    GateDraft, IssueLevel, KeepOut, LightMode, MatSlot, Mats, NoteMove, PowerSource, PowerSpec,
    PropCell, PropDecal, PropError, PropHolo, PropIssue, PropKind, PropLine, PropMode, PropSpec,
    PropStyle, Riser, Screen, StairForm, PROP_MAX_LIGHTS,
};
use crate::{Level, LevelMaterial};

/// Moss green of `overgrowth` (§3.8 #24).
const MOSS_RGB: u32 = 0x4E_8A3A;
/// Amber of the power budget (§3.6 colour law).
const BUDGET_RGB: u32 = 0xFF_B040;
/// The CCTV tally red.
const TALLY_RGB: u32 = 0xFF_3030;
/// A pendant's own lamp colour when `light = yes`.
const LAMP_RGB: u32 = 0xFF_D8A0;
/// Line lift off a surface so a stroke is never coplanar with the face it rides.
const EPS: f32 = 0.02;

/// State shared across every prop of one expansion.
#[derive(Default)]
pub(crate) struct Shared {
    /// `[light]` indices already bound by a `light = existing` pendant.
    existing_lights: HashSet<usize>,
    /// Lamps the kit added (`light = yes`).
    new_lights: usize,
}

/// The read-only context of one expansion.
pub(crate) struct Ctx<'a> {
    pub(crate) level: &'a Level,
    pub(crate) base: &'a Map,
    pub(crate) ko: &'a KeepOut,
    pub(crate) occ: &'a HashMap<Cell3, usize>,
    pub(crate) note_hosts: &'a HashSet<Cell3>,
}

/// One prop's output (indices prop-local; [`super::PropExpansion::absorb`]
/// rebases them).
#[derive(Default)]
pub(crate) struct BuiltProp {
    pub(crate) cells: Vec<PropCell>,
    pub(crate) lines: Vec<PropLine>,
    pub(crate) holos: Vec<PropHolo>,
    pub(crate) decals: Vec<PropDecal>,
    pub(crate) emitters: Vec<EmitterSpec>,
    pub(crate) lights: Vec<PointLight>,
    pub(crate) parts: Vec<AbsPart>,
    pub(crate) gates: Vec<GateDraft>,
    pub(crate) note_moves: Vec<NoteMove>,
    pub(crate) issues: Vec<PropIssue>,
    pub(crate) stair_cells: Vec<Cell3>,
}

/// The generator for one prop.
pub(crate) struct Builder<'a, 'b> {
    cx: &'b Ctx<'a>,
    s: &'b PropSpec,
    pi: usize,
    mats: &'b mut Mats,
    shared: &'b mut Shared,
    note: Option<usize>,
    out: BuiltProp,
    taken: HashSet<Cell3>,
    /// Per gate: (lit LINE material, lit CELL material).
    gate_mats: Vec<(String, String)>,
    /// Build cells skipped because the base map was not Empty / an earlier prop
    /// took them — reported once, with the first position.
    skipped: Vec<(Cell3, String)>,
    skin_missed: usize,
    main_gate: Option<usize>,
}

fn v3(c: Cell3) -> Vec3 {
    Vec3::new(c.0 as f32, c.1 as f32, c.2 as f32)
}

fn fvec(d: (i32, i32)) -> Vec3 {
    Vec3::new(d.0 as f32, d.1 as f32, 0.0)
}

/// A continuous frame: `p(U, V, h) = o + U·ru + V·bv + h·ẑ`.
#[derive(Clone, Copy)]
struct Frame {
    o: Vec3,
    ru: Vec3,
    bv: Vec3,
}

impl Frame {
    fn p(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.o + self.ru * u + self.bv * v + Vec3::Z * h
    }

    /// A HUMAN-HEIGHT point: `h` is in the kit's body units (the standing eye is 1.0
    /// above the floor), converted to world z by [`TRACE_H`].
    fn ph(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.p(u, v, h * TRACE_H)
    }
}

/// World z of one body unit. Trace furniture, rails and handrails were authored when
/// a `[line]` z put the standing eye 1.0 above the floor; `[line]` z is now WORLD z
/// (floor top at the cell's `z`, eye 0.5 above it), so every human-height constant —
/// a table top 0.45, a chair back 0.75, a rail 0.9 — is scaled by this to keep its
/// place relative to the eye. Cell-anchored strokes (a desk's keys, a rack's blade
/// lights, a lamp cord hung from the real ceiling) are already world geometry.
pub(crate) const TRACE_H: f32 = 0.5;

/// Handrail / stair-rail height above the walking surface, world z: 0.9 body units
/// (just under the eye).
pub(crate) const RAIL_H: f32 = 0.9 * TRACE_H;

/// A closed loop through `pts` (repeats the first point).
fn closed(mut pts: Vec<Vec3>) -> Vec<Vec3> {
    if let Some(&f) = pts.first() {
        pts.push(f);
    }
    pts
}

/// Drop interior points that are collinear with their neighbours.
fn simplify(pts: Vec<Vec3>) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::with_capacity(pts.len());
    for p in pts {
        if out.last().is_some_and(|l| l.distance(p) < 1e-4) {
            continue;
        }
        if out.len() >= 2 {
            let a = out[out.len() - 2];
            let b = out[out.len() - 1];
            if (b - a).cross(p - b).length() < 1e-4 && (b - a).dot(p - b) > 0.0 {
                out.pop();
            }
        }
        out.push(p);
    }
    out
}

fn em(origin: Vec3) -> EmitterSpec {
    let mut e = EmitterSpec::new().with_origin(EmitterOrigin::World(origin));
    e.synthesized = true;
    e
}

/// Frost falling from a cold surface (a level's own fiction can read cold as dead).
fn frost(o: Vec3) -> EmitterSpec {
    em(o)
        .with_colors(0xE8_F4FF, 0xA8_D0F0, 0x5A_7890)
        .with_family("haze_dot")
        .with_direction(Vec3::new(0.0, 0.0, -1.0), 25.0)
        .with_rate(5.0)
        .with_speed(0.35, 0.1)
        .with_lifetime(2.4, 0.5)
        .with_max_particles(60)
        .validated()
}

/// Steam rising from a warm surface (warm = alive).
fn steam(o: Vec3) -> EmitterSpec {
    em(o)
        .with_colors(0xF0_F0F0, 0xB8_C0C8, 0x60_6870)
        .with_family("haze_o")
        .with_direction(Vec3::new(0.0, 0.0, 1.0), 20.0)
        .with_rate(8.0)
        .with_speed(0.6, 0.15)
        .with_lifetime(2.2, 0.4)
        .with_max_particles(80)
        .validated()
}

fn haze(o: Vec3, dir: Vec3) -> EmitterSpec {
    em(o)
        .with_colors(0x90_98A0, 0x60_6870, 0x30_3438)
        .with_family("haze_wave")
        .with_direction(dir, 35.0)
        .with_rate(2.5)
        .with_speed(0.2, 0.05)
        .with_lifetime(3.5, 0.8)
        .with_max_particles(40)
        .validated()
}

fn dust(o: Vec3) -> EmitterSpec {
    em(o)
        .with_colors(0xA0_A0A0, 0x70_7070, 0x40_4040)
        .with_glyph_set(vec!['·'])
        .with_direction(Vec3::new(0.0, 0.0, -1.0), 30.0)
        .with_rate(1.5)
        .with_speed(0.12, 0.04)
        .with_lifetime(5.0, 1.0)
        .with_max_particles(60)
        .validated()
}

fn spores(o: Vec3) -> EmitterSpec {
    em(o)
        .with_colors(0x9A_D07A, 0x6A_9A4A, 0x3A_5A2A)
        .with_family("haze_dot")
        .with_direction(Vec3::new(0.0, 0.0, 1.0), 70.0)
        .with_rate(1.5)
        .with_speed(0.08, 0.03)
        .with_lifetime(6.0, 1.5)
        .with_max_particles(80)
        .validated()
}

/// The budget bank's cap burst (ON) and its idle twin (OFF).
fn sparks(o: Vec3, on: bool) -> EmitterSpec {
    let e = em(o)
        .with_colors(0xFF_E0A0, BUDGET_RGB, 0xA0_5010)
        .with_family("growth_dot")
        .with_direction(Vec3::new(0.0, 0.0, 1.0), 70.0)
        .with_speed(1.2, 0.4)
        .with_lifetime(0.6, 0.2)
        .with_max_particles(24);
    if on {
        e.with_burst(24).with_rate(0.0).validated()
    } else {
        e.with_burst(0).with_rate(0.0).validated()
    }
}

impl<'a, 'b> Builder<'a, 'b> {
    pub(crate) fn new(
        cx: &'b Ctx<'a>,
        s: &'b PropSpec,
        pi: usize,
        mats: &'b mut Mats,
        shared: &'b mut Shared,
        note: Option<usize>,
    ) -> Self {
        Builder {
            cx,
            s,
            pi,
            mats,
            shared,
            note,
            out: BuiltProp::default(),
            taken: HashSet::new(),
            gate_mats: Vec::new(),
            skipped: Vec::new(),
            skin_missed: 0,
            main_gate: None,
        }
    }

    // ── frame ────────────────────────────────────────────────────────────────

    fn at(&self) -> IVec3 {
        self.s.at_cell()
    }

    fn f(&self) -> (i32, i32) {
        self.s.facing.vec()
    }

    fn r(&self) -> (i32, i32) {
        self.s.facing.right()
    }

    /// `world(u, v, w) = at + u·r − v·f + w·ẑ` (§3.3).
    fn cell(&self, u: i32, v: i32, w: i32) -> Cell3 {
        let (a, f, r) = (self.at(), self.f(), self.r());
        (a.x + u * r.0 - v * f.0, a.y + u * r.1 - v * f.1, a.z + w)
    }

    /// The §3.3 frame continued to world points: local `U, V ∈ [0, W/D]` cover the
    /// cells `u = ⌊U⌋, v = ⌊V⌋`; `h` is height above the bottom of `at`'s cell.
    fn frame(&self) -> Frame {
        let a = self.at();
        let (f, r) = (fvec(self.f()), fvec(self.r()));
        Frame {
            o: Vec3::new(a.x as f32 + 0.5, a.y as f32 + 0.5, a.z as f32) + f * 0.5 - r * 0.5,
            ru: r,
            bv: -f,
        }
    }

    fn pt(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.frame().p(u, v, h)
    }

    fn size_i(&self) -> (i32, i32, i32) {
        let z = self.s.size;
        (z[0] as i32, z[1] as i32, z[2] as i32)
    }

    // ── map queries ─────────────────────────────────────────────────────────

    fn solid_at(&self, c: Cell3) -> bool {
        let k = self.cx.base.cell_at(c.0, c.1, c.2);
        k.is_solid() || matches!(k, CellKind::Floor | CellKind::Ceiling)
    }

    fn in_bounds(&self, c: Cell3) -> bool {
        self.cx.base.in_bounds(c.0, c.1, c.2)
    }

    /// The first solid cell strictly above `c` in its column (the ceiling), or
    /// the room's height when there is none.
    fn ceil_above(&self, c: Cell3) -> i32 {
        let (_, _, sz) = self.cx.base.size();
        ((c.2 + 1).max(0)..sz as i32)
            .find(|&z| self.solid_at((c.0, c.1, z)))
            .unwrap_or(sz as i32)
    }

    /// Solid in the room OR already built by a prop (an earlier `[prop]`, or this one).
    fn solid_or_prop(&self, c: Cell3) -> bool {
        self.solid_at(c) || self.cx.occ.contains_key(&c) || self.taken.contains(&c)
    }

    /// [`floor_at`](Self::floor_at) that also stands on prop cells.
    fn floor_or_prop_at(&self, c: Cell3) -> i32 {
        let mut z = c.2;
        while z >= 0 {
            let p = (c.0, c.1, z);
            if self.cx.occ.contains_key(&p) || self.taken.contains(&p) {
                return z;
            }
            if self.solid_at(p)
                && re_core::decode_pipe_tag(self.cx.base.side_data_at(p.0, p.1, p.2)).is_none()
            {
                return z;
            }
            z -= 1;
        }
        -1
    }

    /// The first solid (non-pipe) cell at or below `c` in its column, or `-1`.
    fn floor_at(&self, c: Cell3) -> i32 {
        let m = self.cx.base;
        let mut z = c.2;
        while z >= 0 {
            if self.solid_at((c.0, c.1, z))
                && re_core::decode_pipe_tag(m.side_data_at(c.0, c.1, z)).is_none()
            {
                return z;
            }
            z -= 1;
        }
        -1
    }

    /// The note binder's eye layer for a reader standing at world z `z`.
    fn eye_layer(&self, z: i32) -> i32 {
        let (_, _, sz) = self.cx.base.size();
        let h = self.cx.level.storey_height().max(1) as i32;
        let top = (sz as i32 - 2).max(1);
        let mut storey = z.max(0) / h;
        while storey > 0 && storey * h + 1 > top {
            storey -= 1;
        }
        (storey * h + 1).clamp(1, top)
    }

    // ── issues ──────────────────────────────────────────────────────────────

    fn issue(&mut self, level: IssueLevel, msg: impl Into<String>, pos: Option<Cell3>) {
        self.out.issues.push(PropIssue {
            prop: self.s.name.clone(),
            level,
            msg: msg.into(),
            pos,
        });
    }

    fn err(&self, key: &str, msg: impl Into<String>) -> PropError {
        PropError::new(
            self.s.line_of(key),
            format!("[prop {}] {}: {}", self.s.name, key, msg.into()),
        )
    }

    // ── materials ───────────────────────────────────────────────────────────

    /// The material for a pack slot of `style`, honouring `mat.SLOT` overrides
    /// when `style` is the prop's own.
    fn mat_of(&mut self, style: PropStyle, slot: MatSlot) -> Result<String, PropError> {
        if style == self.s.style {
            if let Some((_, v, line)) = self.s.mats.iter().find(|(s, _, _)| *s == slot) {
                let (v, line) = (v.clone(), *line);
                return self.mats.resolve_value(&v).ok_or_else(|| {
                    PropError::new(
                        line,
                        format!(
                            "[prop {}] mat.{}: '{v}' is not a level material, #RRGGBB, catalog preset or recipe",
                            self.s.name,
                            slot.as_str()
                        ),
                    )
                });
            }
        }
        Ok(self.mats.intern(pack_material(style, slot)))
    }

    fn mat(&mut self, slot: MatSlot) -> Result<String, PropError> {
        self.mat_of(self.s.style, slot)
    }

    fn flat(&mut self, c: u32) -> String {
        self.mats.intern(super::flat_material(c))
    }

    fn preset(&mut self, name: &str) -> String {
        self.mats.resolve_value(name).unwrap_or_default()
    }

    /// The ink colour a stroke draws in (a flat material, so the stroke has one
    /// hue — a preset slot's catalog palette would evaluate to its mid tone).
    fn ink(&mut self) -> Result<String, PropError> {
        self.mat(MatSlot::Ink)
    }

    /// Look up a material definition by name (authored first, then synthesized).
    fn def_of(&self, name: &str) -> Option<LevelMaterial> {
        self.cx
            .level
            .materials
            .iter()
            .find(|m| m.name == name)
            .or_else(|| self.mats.synth.iter().find(|m| m.name == name))
            .cloned()
    }

    // ── power ───────────────────────────────────────────────────────────────

    /// Resolve a power source into a prop-local gate (deduplicated). `None` for
    /// `power = none` and for the `dead` style, which never lights.
    fn gate_with(
        &mut self,
        p: &PowerSpec,
        invert: bool,
        hue: Option<u32>,
    ) -> Result<Option<usize>, PropError> {
        if p.is_none() || self.s.style == PropStyle::Dead {
            return Ok(None);
        }
        if let Some(i) = self
            .out
            .gates
            .iter()
            .position(|g| g.spec == *p && g.invert == invert)
        {
            return Ok(Some(i));
        }
        let level = self.cx.level;
        let src = resolve_power(level, p)
            .map_err(|m| self.err("power", m))?
            .ok_or_else(|| self.err("power", "unresolved source"))?;
        let initial_lit = initial_value(level, src) != invert;
        // `mat.lit` overrides the colour-law hue: its cells and lines draw in it, and
        // its own colour becomes the gate's `lit_rgb` — the hue the lamps, emitters
        // and holograms of the gate light in, and the one lvlcheck's colour law
        // checks. (A `door` / `seq:` gate otherwise has no puzzle hue and falls back to
        // the pack colour, which the colour law rejects.) Otherwise lines use the flat
        // colour-law hue and cells a lit screen.
        let lit_override = if self.s.mats.iter().any(|(s, _, _)| *s == MatSlot::Lit) {
            let m = self.mat(MatSlot::Lit)?;
            let fg = self
                .def_of(&m)
                .map(|d| d.fg & 0xFF_FFFF)
                .filter(|&c| c != 0);
            Some((m, fg))
        } else {
            None
        };
        let hue = hue
            .or_else(|| lit_override.as_ref().and_then(|(_, fg)| *fg))
            .unwrap_or_else(|| source_hue(level, src, self.s.style));
        let (line_lit, cell_lit) = match lit_override {
            Some((m, _)) => (m.clone(), m),
            None => (self.flat(hue), self.mats.intern(lit_screen_material(hue))),
        };
        self.out.gates.push(GateDraft {
            prop: self.s.name.clone(),
            spec: p.clone(),
            source: src,
            invert,
            initial_lit,
            lit_rgb: hue,
            lights: Vec::new(),
            emitters: Vec::new(),
        });
        self.gate_mats.push((line_lit, cell_lit));
        Ok(Some(self.out.gates.len() - 1))
    }

    fn gate(&mut self, p: &PowerSpec) -> Result<Option<usize>, PropError> {
        self.gate_with(p, self.s.invert, None)
    }

    fn lit(&self, g: Option<usize>) -> bool {
        g.is_none_or(|g| self.out.gates[g].initial_lit)
    }

    // ── stamps ──────────────────────────────────────────────────────────────

    /// Stamp one BUILD cell (Empty only). A cell inside the hard keep-out is a
    /// load error naming both; a ring cell is stamped and reported (lvlcheck
    /// FAIL); an occupied cell is skipped and reported.
    fn solid(&mut self, c: Cell3, shape: Shape, mat: Option<String>) -> Result<bool, PropError> {
        if self.taken.contains(&c) {
            return Ok(false);
        }
        if !self.in_bounds(c) {
            self.skipped.push((c, "outside the room".into()));
            return Ok(false);
        }
        if let Some(k) = self.cx.ko.get(c) {
            if k.hard {
                return Err(PropError::new(
                    self.s.line,
                    format!(
                        "[prop {}] solid cell ({},{},{}) enters the {} (puzzle keep-out)",
                        self.s.name, c.0, c.1, c.2, k.what
                    ),
                ));
            }
            let what = k.what.clone();
            self.issue(
                IssueLevel::Fail,
                format!(
                    "solid cell ({},{},{}) is inside the {what} keep-out",
                    c.0, c.1, c.2
                ),
                Some(c),
            );
        }
        let k = self.cx.base.cell_at(c.0, c.1, c.2);
        if k != CellKind::Empty {
            self.skipped.push((c, format!("{k:?}")));
            return Ok(false);
        }
        if let Some(&other) = self.cx.occ.get(&c) {
            let name = self
                .cx
                .level
                .props
                .get(other)
                .map_or("?", |p| p.name.as_str());
            self.skipped.push((c, format!("prop {name}")));
            return Ok(false);
        }
        self.taken.insert(c);
        self.out.cells.push(PropCell {
            pos: c,
            kind: CellKind::WallSolid,
            shape,
            mat,
            mode: CellMode::Build,
            prop: self.pi,
        });
        Ok(true)
    }

    fn full(&mut self, c: Cell3, mat: &str) -> Result<bool, PropError> {
        self.solid(c, Shape::Full, Some(mat.to_string()))
    }

    /// Re-material an EXISTING geometry cell (`mode = skin`). Puzzle-owned cells
    /// (pipe traces, nodes, portals, holograms) and note placard walls are left
    /// alone.
    fn skin(&mut self, c: Cell3, mat: &str) -> bool {
        let m = self.cx.base;
        let k = m.cell_at(c.0, c.1, c.2);
        let geometry = matches!(
            k,
            CellKind::WallSolid
                | CellKind::WallTrace
                | CellKind::WallMatrix
                | CellKind::Floor
                | CellKind::Ceiling
        );
        if !self.in_bounds(c)
            || !geometry
            || re_core::decode_pipe_tag(m.side_data_at(c.0, c.1, c.2)).is_some()
            || self.cx.note_hosts.contains(&c)
            || self.taken.contains(&c)
        {
            self.skin_missed += 1;
            return false;
        }
        self.taken.insert(c);
        self.out.cells.push(PropCell {
            pos: c,
            kind: k,
            shape: m.shape_at(c.0, c.1, c.2),
            mat: Some(mat.to_string()),
            mode: CellMode::Skin,
            prop: self.pi,
        });
        true
    }

    /// An R1 lit slot: Empty now, filled with the gate's lit material while lit.
    fn part(&mut self, c: Cell3, gate: usize) -> Result<(), PropError> {
        let mat = self.gate_mats[gate].1.clone();
        self.part_mat(c, gate, mat)
    }

    /// A RACK's lit slot: the gate's hue as an LED / scanline emitter
    /// ([`super::rack_led_material`]), not the gate's flat lit screen. A lit blade in
    /// a flat neon of its source's hue reads as the brick of a wall of the same hue
    /// instead of a powered object, so the slot uses the LED/scanline look instead.
    fn rack_part(&mut self, c: Cell3, gate: usize) -> Result<(), PropError> {
        let hue = self.out.gates[gate].lit_rgb;
        let mat = self.mats.intern(super::rack_led_material(hue));
        self.part_mat(c, gate, mat)
    }

    fn part_mat(&mut self, c: Cell3, gate: usize, mat: String) -> Result<(), PropError> {
        if let Some(k) = self.cx.ko.get(c) {
            if k.hard {
                return Err(PropError::new(
                    self.s.line,
                    format!(
                        "[prop {}] lit slot ({},{},{}) enters the {} (puzzle keep-out)",
                        self.s.name, c.0, c.1, c.2, k.what
                    ),
                ));
            }
        }
        if !self.in_bounds(c)
            || self.cx.base.cell_at(c.0, c.1, c.2) != CellKind::Empty
            || self.cx.occ.contains_key(&c)
        {
            self.skipped.push((c, "lit slot not Empty".into()));
            return Ok(());
        }
        self.taken.insert(c);
        self.out.parts.push(AbsPart {
            pos: c,
            mat,
            prop: self.pi,
            gate,
        });
        Ok(())
    }

    fn line(&mut self, pts: Vec<Vec3>, mat: &str) {
        self.line_g(pts, mat, None);
    }

    /// A stroke; with a gate it switches `ink` ↔ the gate's lit colour.
    fn line_g(&mut self, pts: Vec<Vec3>, ink: &str, gate: Option<usize>) {
        if pts.len() < 2 {
            return;
        }
        let lit_mat = gate.map(|g| self.gate_mats[g].0.clone());
        let lit_now = gate.is_some_and(|g| self.out.gates[g].initial_lit);
        self.out.lines.push(PropLine {
            points: pts,
            mat: ink.to_string(),
            prop: self.pi,
            gate,
            lit_mat,
            lit_now,
        });
    }

    /// An emitter; with a gate, `(lit, unlit)` specs swap with the source.
    fn emitter(&mut self, lit: EmitterSpec, unlit: EmitterSpec, gate: Option<usize>) {
        let idx = self.out.emitters.len();
        match gate {
            Some(g) => {
                let on = self.out.gates[g].initial_lit;
                let (now, other) = if on { (lit, unlit) } else { (unlit, lit) };
                self.out.emitters.push(now);
                self.out.gates[g].emitters.push((idx, other));
            }
            None => self.out.emitters.push(unlit),
        }
    }

    fn holo(&mut self, shape: HoloShape, min: Cell3, max: Cell3, style: u16, gate: Option<usize>) {
        for z in min.2..=max.2 {
            for y in min.1..=max.1 {
                for x in min.0..=max.0 {
                    if self.cx.ko.is_switch_column((x, y, z)) {
                        self.issue(
                            IssueLevel::Fail,
                            format!("hologram cell ({x},{y},{z}) stands in a switch column"),
                            Some((x, y, z)),
                        );
                        return;
                    }
                }
            }
        }
        let visible = self.lit(gate);
        self.out.holos.push(PropHolo {
            shape,
            min,
            max,
            style,
            prop: self.pi,
            visible,
            gate,
        });
    }

    // ── driver ──────────────────────────────────────────────────────────────

    pub(crate) fn run(&mut self) -> Result<(), PropError> {
        let s = self.s;
        // A followed pipe powers its tray / cable unless `power` says otherwise.
        let power = match (&s.follow, s.key_lines.contains_key("power")) {
            (Some((a, b)), false) if matches!(s.kind, PropKind::CableRun | PropKind::CableTray) => {
                PowerSpec::Pipe(a.clone(), b.clone())
            }
            _ => s.power.clone(),
        };
        if !matches!(s.kind, PropKind::BudgetBank | PropKind::Cctv) {
            self.main_gate = self.gate(&power)?;
        }
        match s.kind {
            PropKind::Rack => self.rack(),
            PropKind::RackRow => self.rack_row(),
            PropKind::Console => self.console(),
            PropKind::Shelving => self.shelving(),
            PropKind::LockerBank => self.locker_bank(),
            PropKind::Tank => self.tank(),
            PropKind::BudgetBank => self.budget_bank(),
            PropKind::Column => self.column(),
            PropKind::Partition => self.partition(),
            PropKind::Debris => self.debris(),
            PropKind::Cctv => self.cctv(),
            PropKind::WallPanel => self.wall_panel(),
            PropKind::NoticeBoard => self.notice_board(),
            PropKind::Stair => self.stair(),
            PropKind::Gallery => self.gallery(),
            PropKind::CableTray => self.cable_tray(),
            PropKind::Conduit => self.conduit(),
            PropKind::HangingPanels => self.hanging_panels(),
            PropKind::CableRun => self.cable_run(),
            PropKind::CableDrop => self.cable_drop(),
            PropKind::LampPendant => self.lamp_pendant(),
            PropKind::FurnitureTrace => self.furniture_trace(),
            PropKind::HoloDisplay => self.holo_display(),
            PropKind::Overgrowth => self.overgrowth(),
        }
    }

    pub(crate) fn finish(mut self) -> BuiltProp {
        if let Some((c, why)) = self.skipped.first().cloned() {
            let n = self.skipped.len();
            self.issue(
                IssueLevel::Warn,
                format!(
                    "{n} cell(s) not stamped (Empty-only); first ({},{},{}) holds {why}",
                    c.0, c.1, c.2
                ),
                Some(c),
            );
        }
        if self.skin_missed > 0 {
            let n = self.skin_missed;
            self.issue(
                IssueLevel::Warn,
                format!("mode = skin: {n} footprint cell(s) had no plain geometry to re-material"),
                None,
            );
        }
        self.out
    }

    // ── 1 · rack ────────────────────────────────────────────────────────────

    /// One rack at local column `u` (shared by `rack` and `rack_row`).
    fn rack_unit(
        &mut self,
        u: i32,
        style: PropStyle,
        gate: Option<usize>,
        d: i32,
        h: i32,
    ) -> Result<(), PropError> {
        let front = self.cell(u, -1, 0);
        if self.solid_at(front) {
            return Err(self.err(
                "facing",
                format!(
                    "the rack front at column {u} faces a wall at ({},{},{}); it must face walkable air",
                    front.0, front.1, front.2
                ),
            ));
        }
        let ink = self.ink()?;
        if self.s.mode == PropMode::Skin {
            let face = self.mat_of(style, MatSlot::Face)?;
            for v in 0..d {
                for w in 0..h {
                    self.skin(self.cell(u, v, w), &face);
                }
            }
        } else if d >= 2 {
            let body = self.mat_of(style, MatSlot::Body)?;
            let trim = self.mat_of(style, MatSlot::Trim)?;
            for w in 0..h {
                self.full(self.cell(u, 1, w), &body)?;
            }
            if h >= 3 {
                self.full(self.cell(u, 0, h - 1), &trim)?;
            }
            // The blade slot sits in the recess.
            let blade = self.cell(u, 0, if h >= 2 { 1 } else { 0 });
            match gate {
                Some(g) => self.rack_part(blade, g)?,
                None => {
                    let dead = self.mat_of(style, MatSlot::Dead)?;
                    self.full(blade, &dead)?;
                }
            }
        } else {
            // A one-deep unit: a powered unit's blade is an R1 lit slot in the face
            // itself (dark = an empty bay), so a `rack_row` — always one deep — shows
            // its payoff as lit cells, not only as LED strokes.
            let face = self.mat_of(style, MatSlot::Face)?;
            let blade_w = if h >= 2 { 1 } else { 0 };
            for w in 0..h {
                match gate {
                    Some(g) if w == blade_w => self.rack_part(self.cell(u, 0, w), g)?,
                    _ => {
                        self.full(self.cell(u, 0, w), &face)?;
                    }
                }
            }
        }
        // LED strips on the face the viewer sees (the recessed body face when D=2).
        let vf = if d >= 2 { 1.0 - EPS } else { -EPS };
        let uf = u as f32;
        let hf = h as f32;
        for du in [0.25, 0.75] {
            let pts = vec![self.pt(uf + du, vf, 0.1), self.pt(uf + du, vf, hf - 0.1)];
            self.line_g(pts, &ink, gate);
        }
        let bar_v = if d >= 2 && h < 3 { vf } else { -EPS };
        let bar = vec![
            self.pt(uf + 0.15, bar_v, hf - 0.12),
            self.pt(uf + 0.85, bar_v, hf - 0.12),
        ];
        self.line_g(bar, &ink, gate);
        Ok(())
    }

    fn rack(&mut self) -> Result<(), PropError> {
        let (_, d, h) = self.size_i();
        self.rack_unit(0, self.s.style, self.main_gate, d, h)
    }

    fn rack_row(&mut self) -> Result<(), PropError> {
        let (w, _, h) = self.size_i();
        for i in 0..w {
            if self.s.gap.is_some_and(|k| (i + 1) % k as i32 == 0) {
                continue;
            }
            let dead = unit3(self.s.seed, i as u32, 11) < self.s.dead;
            let (style, gate) = if dead {
                (PropStyle::Dead, None)
            } else if let Some(p) = self.s.power_each.get(i as usize).cloned() {
                (self.s.style, self.gate(&p)?)
            } else {
                (self.s.style, self.main_gate)
            };
            self.rack_unit(i, style, gate, 1, h)?;
        }
        if self.s.tray {
            let top = self.cell(0, 0, h);
            if h < self.ceil_above(self.cell(0, 0, 0)) - self.at().z {
                let trim = self.mat(MatSlot::Trim)?;
                let ink = self.ink()?;
                for i in 0..w {
                    self.full(self.cell(i, 0, h), &trim)?;
                }
                let hf = h as f32;
                for side in [0.2, 0.8] {
                    let pts = self.sag_run(
                        (0..=w)
                            .map(|i| self.pt(i as f32, side, hf + 1.0 + EPS))
                            .collect(),
                        3,
                        0.12,
                    );
                    self.line_g(pts, &ink, self.main_gate);
                }
            } else {
                self.issue(
                    IssueLevel::Warn,
                    "tray = yes: no room under the ceiling",
                    Some(top),
                );
            }
        }
        Ok(())
    }

    /// Make a polyline sag between every `every`-th point by `sag`.
    fn sag_run(&self, pts: Vec<Vec3>, every: usize, sag: f32) -> Vec<Vec3> {
        let mut out = Vec::new();
        for (i, w) in pts.windows(2).enumerate() {
            if i == 0 {
                out.push(w[0]);
            }
            let mid = (w[0] + w[1]) * 0.5;
            if (i + 1) % every.max(1) != 0 {
                out.push(mid - Vec3::Z * sag);
            }
            out.push(w[1]);
        }
        out
    }

    // ── 3 · console ─────────────────────────────────────────────────────────

    fn console(&mut self) -> Result<(), PropError> {
        let (w, d, _) = self.size_i();
        let face = self.mat(MatSlot::Face)?;
        let body = self.mat(MatSlot::Body)?;
        let ink = self.ink()?;
        // The sloped panel is a RampZ rising away from the reader (toward −f, the
        // prop's depth axis): the vertical pass draws a ramp cell exactly on every
        // plane, so the desk top reads as a slanted panel instead of the head-high
        // block a Full cell made (it used to be one because the old single-slab caster
        // drew an eye-layer ramp as a floor-to-sky column). The scan strokes below
        // ride this surface (`h = v`).
        let (fx, fy) = self.f();
        let panel = Shape::RampZ(ramp_dir_of((-fx, -fy)));
        for u in 0..w {
            self.solid(self.cell(u, 0, 0), panel, Some(face.clone()))?;
            if d >= 2 {
                self.full(self.cell(u, 1, 0), &body)?;
            }
        }
        if self.s.monitor {
            if d < 2 {
                return Err(self.err(
                    "monitor",
                    "monitor = yes needs a 2-deep console (size = W 2 1)",
                ));
            }
            let crt = self.cell(w / 2, 1, 1);
            match self.main_gate {
                Some(g) => self.part(crt, g)?,
                None => {
                    let dead = self.mat(MatSlot::Dead)?;
                    self.full(crt, &dead)?;
                }
            }
            // The bezel around the CRT.
            let (u0, v0) = ((w / 2) as f32, 1.0 - EPS);
            let bez = closed(vec![
                self.pt(u0 + 0.05, v0, 1.05),
                self.pt(u0 + 0.95, v0, 1.05),
                self.pt(u0 + 0.95, v0, 1.95),
                self.pt(u0 + 0.05, v0, 1.95),
            ]);
            self.line(bez, &ink);
        }
        // Scan glow on the slanted panel: the ramp surface is h = V.
        for u in 0..w {
            for k in 0..3 {
                let v = 0.25 + 0.25 * k as f32;
                let pts = vec![
                    self.pt(u as f32 + 0.2, v, v + 0.03),
                    self.pt(u as f32 + 0.8, v, v + 0.03),
                ];
                self.line_g(pts, &ink, self.main_gate);
            }
        }
        if self.s.chair {
            // A chair in front of the desk, facing it.
            let c = self.pt(w as f32 * 0.5, -0.8, 0.0);
            let fr = self.frame();
            let fl = c + fr.ru * 0.3 + fr.bv * 0.3; // the chair faces −f: its right is −r
            let chair = Frame {
                o: Vec3::new(fl.x, fl.y, self.at().z as f32),
                ru: -fr.ru,
                bv: -fr.bv,
            };
            self.furniture("chair", chair, 0.6, 0.6, 0.0, &ink);
        }
        Ok(())
    }

    // ── 4 · shelving ────────────────────────────────────────────────────────

    fn shelving(&mut self) -> Result<(), PropError> {
        let (w, d, h) = self.size_i();
        let ink = self.ink()?;
        let mut layer = Vec::new();
        for lw in 0..h {
            let slot = if lw == 0 {
                MatSlot::Trim
            } else if h >= 3 && lw == h - 1 {
                if self.s.style == PropStyle::Vault {
                    MatSlot::Face
                } else {
                    MatSlot::Trim
                }
            } else if lw % 2 == 1 {
                MatSlot::Face
            } else {
                MatSlot::Body
            };
            let mut m = self.mat(slot)?;
            if slot == MatSlot::Face {
                if let Some(ch) = self.s.hatch {
                    m = self.hatched(&m, ch);
                }
            }
            layer.push(m);
        }
        for u in 0..w {
            for v in 0..d {
                for lw in 0..h {
                    let c = self.cell(u, v, lw);
                    let m = layer[lw as usize].clone();
                    if self.s.mode == PropMode::Skin {
                        self.skin(c, &m);
                    } else {
                        self.full(c, &m)?;
                    }
                }
            }
        }
        // Drive LED ticks on the odd (face) layers.
        let faces: &[f32] = if self.s.faces_both {
            &[-EPS, 0.0]
        } else {
            &[-EPS]
        };
        for (fi, &vf) in faces.iter().enumerate() {
            let vf = if fi == 1 { d as f32 + EPS } else { vf };
            for u in 0..w {
                for lw in (1..h).step_by(2) {
                    if h >= 3 && lw == h - 1 {
                        continue;
                    }
                    let key = (u * 64 + lw) as u32 + fi as u32 * 4096;
                    if unit3(self.s.seed, key, 3) >= 0.7 {
                        continue;
                    }
                    let x = u as f32 + 0.15 + 0.55 * unit3(self.s.seed, key, 4);
                    let z = lw as f32 + 0.5;
                    let pts = vec![self.pt(x, vf, z), self.pt(x + 0.18, vf, z)];
                    self.line_g(pts, &ink, self.main_gate);
                }
            }
        }
        Ok(())
    }

    /// A glyph-fixed twin of material `name` (`hatch = / | \`).
    fn hatched(&mut self, name: &str, ch: char) -> String {
        match self.def_of(name) {
            Some(mut m) => {
                m.name = format!("{name}_hatch_{:x}", ch as u32);
                m.glyph = Some(ch as u32);
                self.mats.intern(m)
            }
            None => name.to_string(),
        }
    }

    // ── 5 · locker_bank ─────────────────────────────────────────────────────

    fn locker_bank(&mut self) -> Result<(), PropError> {
        let (w, _, _) = self.size_i();
        let body = self.mat(MatSlot::Body)?;
        let fg = self
            .def_of(&body)
            .map_or(0x6A_7480, |m| if m.fg == 0 { 0x6A_7480 } else { m.fg });
        let locker = self.mats.intern(LevelMaterial {
            name: format!(
                "{}{}_locker",
                super::PROP_MATERIAL_PREFIX,
                self.s.style.as_str()
            ),
            fg,
            bg: shade(fg, 0.12),
            recipe: crate::parse_recipe("brick aspect=0.5 mortar_w=0.10 grain=0.15"),
            ..LevelMaterial::default()
        });
        let ink = self.ink()?;
        let host = self.note_host(w)?;
        for u in 0..w {
            if self.s.open == Some(u) {
                // An open unit: a coat hangs inside it.
                let fr = self.frame();
                let o = fr.p(u as f32, 0.0, 0.0);
                self.furniture("coat", Frame { o, ..fr }, 1.0, 1.0, 0.0, &ink);
                continue;
            }
            for lw in 0..2 {
                let c = self.cell(u, 0, lw);
                let m = if host == Some((u, lw)) {
                    None
                } else {
                    Some(locker.clone())
                };
                self.solid(c, Shape::Full, m)?;
            }
            let uf = u as f32;
            self.line(
                vec![
                    self.pt(uf + 0.82, -EPS, 0.85),
                    self.pt(uf + 0.82, -EPS, 1.15),
                ],
                &ink,
            );
            self.line(
                vec![
                    self.pt(uf + 0.25, -EPS, 1.78),
                    self.pt(uf + 0.75, -EPS, 1.78),
                ],
                &ink,
            );
        }
        Ok(())
    }

    /// Re-home the claimed note onto the prop's front at column 0 (or the first
    /// unit that is not open): returns the `(u, w)` of the placard host cell,
    /// which must stay UNTAGGED masonry so the note binder paints it.
    fn note_host(&mut self, w: i32) -> Result<Option<(i32, i32)>, PropError> {
        let Some(ni) = self.note else { return Ok(None) };
        let u = (0..w).find(|&u| self.s.open != Some(u)).unwrap_or(0);
        let eye = self.eye_layer(self.at().z);
        let hw = eye - self.at().z;
        let reader = self.cell(u, -1, hw);
        if self.solid_at(reader) {
            return Err(self.err(
                "note",
                "the cell in front of the board is not air, so the placard has no reader",
            ));
        }
        if !(0..2).contains(&hw) {
            return Err(self.err(
                "note",
                format!(
                    "the placard eye layer z={eye} is not on this prop (its cells span z={}..{})",
                    self.at().z,
                    self.at().z + 1
                ),
            ));
        }
        // The note's `pos` is the WALL cell the placard mounts on (the binder's
        // convention): the board's own front cell at the eye layer.
        self.out.note_moves.push(NoteMove {
            note: ni,
            pos: self.cell(u, 0, hw),
            facing: self.s.facing.yaw_col(),
        });
        Ok(Some((u, hw)))
    }

    // ── 6 · tank ────────────────────────────────────────────────────────────

    fn tank(&mut self) -> Result<(), PropError> {
        let (w, d, h) = self.size_i();
        let trim = self.mat(MatSlot::Trim)?;
        let ink = self.ink()?;
        let (f, r) = (self.f(), self.r());
        let chamfer = w >= 2 && d >= 2;
        for u in 0..w {
            for v in 0..d {
                // Corner columns are NOT wedged: WedgeXY cells in the eye slab render as
                // floor-to-sky column artefacts in the single-slab raycaster. The
                // chamfer is carried by the octagonal cap ring instead (documented gap).
                for lw in 0..h {
                    self.solid(self.cell(u, v, lw), Shape::Full, Some(trim.clone()))?;
                }
            }
        }
        let (wf, df, hf) = (w as f32, d as f32, h as f32);
        let top = hf + EPS;
        let ring = if chamfer {
            vec![
                self.pt(1.0, 0.0, top),
                self.pt(wf - 1.0, 0.0, top),
                self.pt(wf, 1.0, top),
                self.pt(wf, df - 1.0, top),
                self.pt(wf - 1.0, df, top),
                self.pt(1.0, df, top),
                self.pt(0.0, df - 1.0, top),
                self.pt(0.0, 1.0, top),
            ]
        } else {
            vec![
                self.pt(0.0, 0.0, top),
                self.pt(wf, 0.0, top),
                self.pt(wf, df, top),
                self.pt(0.0, df, top),
            ]
        };
        self.line(closed(ring), &ink);
        // The gauge on the front: a column with a level tick.
        let gu = wf * 0.5;
        self.line_g(
            vec![self.pt(gu, -EPS, 0.3), self.pt(gu, -EPS, hf - 0.3)],
            &ink,
            self.main_gate,
        );
        let lvl = 0.3 + (hf - 0.6) * (0.35 + 0.4 * unit3(self.s.seed, 0, 21));
        self.line_g(
            vec![self.pt(gu - 0.15, -EPS, lvl), self.pt(gu + 0.15, -EPS, lvl)],
            &ink,
            self.main_gate,
        );
        // A stub pipe to the nearest wall (left, right or back), at mid height.
        let mid = (h / 2).min(h - 1);
        let mut best: Option<(i32, Vec3, Vec3)> = None;
        let sides: [(i32, i32, Vec3, Vec3); 3] = [
            (
                -1,
                0,
                self.pt(0.0, df * 0.5, mid as f32 + 0.5),
                fvec((-r.0, -r.1)),
            ),
            (w, 0, self.pt(wf, df * 0.5, mid as f32 + 0.5), fvec(r)),
            (
                0,
                d,
                self.pt(wf * 0.5, df, mid as f32 + 0.5),
                fvec((-f.0, -f.1)),
            ),
        ];
        for (k, (du, dv, start, dir)) in sides.into_iter().enumerate() {
            let (u0, v0) = if k == 2 { (w / 2, dv) } else { (du, d / 2) };
            for step in 0..6 {
                let (uu, vv) = match k {
                    0 => (u0 - step, v0),
                    1 => (u0 + step, v0),
                    _ => (u0, v0 + step),
                };
                if self.solid_at(self.cell(uu, vv, mid)) {
                    if best.is_none_or(|b| step < b.0) {
                        best = Some((step, start, start + dir * (step as f32)));
                    }
                    break;
                }
            }
        }
        if let Some((_, a, b)) = best {
            if a.distance(b) > 0.05 {
                self.line(vec![a, b], &ink);
            }
        }
        if self.s.emitter {
            let o = self.pt(wf * 0.5, df * 0.5, hf + 0.2);
            self.emitter(steam(o), frost(o), self.main_gate);
        }
        Ok(())
    }

    // ── 7 · budget_bank ─────────────────────────────────────────────────────

    fn budget_bank(&mut self) -> Result<(), PropError> {
        let k = self.cx.level.power_budget as i32;
        if k <= 0 {
            return Err(self.err("kind", "budget_bank needs the level's [room] power_budget"));
        }
        if self.at().z < 3 {
            return Err(self.err(
                "at",
                "budget_bank is wall-mounted at head height or above (at.z >= 3)",
            ));
        }
        let trim = self.mat(MatSlot::Trim)?;
        let ink = self.ink()?;
        let dead = self.mat(MatSlot::Dead)?;
        for i in 0..k {
            let back = self.cell(i, 1, 0);
            if !self.solid_at(back) {
                self.full(back, &trim)?;
            }
            let slot = self.cell(i, 0, 0);
            match self.gate_with(&PowerSpec::Budget(i as u32 + 1), false, Some(BUDGET_RGB))? {
                Some(g) => self.part(slot, g)?,
                None => {
                    self.full(slot, &dead)?;
                }
            }
            let fi = i as f32;
            let ring = closed(vec![
                self.pt(fi + 0.12, 1.0 - EPS, 0.12),
                self.pt(fi + 0.88, 1.0 - EPS, 0.12),
                self.pt(fi + 0.88, 1.0 - EPS, 0.88),
                self.pt(fi + 0.12, 1.0 - EPS, 0.88),
            ]);
            self.line(ring, &ink);
        }
        let cap = self.gate_with(&PowerSpec::Budget(k as u32), false, Some(BUDGET_RGB))?;
        let o = self.pt(k as f32 * 0.5, -0.3, 0.5);
        self.emitter(sparks(o, true), sparks(o, false), cap);
        Ok(())
    }

    // ── 8 · column ──────────────────────────────────────────────────────────

    fn column(&mut self) -> Result<(), PropError> {
        let (w, d, h0) = self.size_i();
        let at = self.at();
        let skin = self.s.mode == PropMode::Skin;
        let h = if h0 > 0 {
            h0
        } else if skin {
            // The existing pillar: from `at` up while the column is geometry.
            let (_, _, sz) = self.cx.base.size();
            (at.z..sz as i32 - 1)
                .take_while(|&z| self.solid_at((at.x, at.y, z)))
                .count() as i32
        } else {
            (self.ceil_above(self.cell(0, 0, 0)) - at.z).max(1)
        };
        let body = self.mat(MatSlot::Body)?;
        let trim = self.mat(MatSlot::Trim)?;
        for u in 0..w {
            for v in 0..d {
                for lw in 0..h {
                    let m = if lw == 0 || lw == h - 1 { &trim } else { &body };
                    let c = self.cell(u, v, lw);
                    if skin {
                        self.skin(c, m);
                    } else {
                        self.full(c, m)?;
                    }
                }
            }
        }
        if self.s.ribs && h >= 3 {
            let ink = self.ink()?;
            let (wf, df, hf) = (w as f32, d as f32, h as f32);
            // (fixed coordinate, along-U?) per face: front, back, left, right.
            let faces = [
                (-EPS, true, (0, -1)),
                (df + EPS, true, (0, d)),
                (-EPS, false, (-1, 0)),
                (wf + EPS, false, (w, 0)),
            ];
            for (fixed, along_u, probe) in faces {
                let out = self.cell(probe.0, probe.1, h / 2);
                if self.solid_at(out) {
                    continue;
                }
                let span = if along_u { wf } else { df };
                for k in 1..=4 {
                    let t = span * k as f32 / 5.0;
                    let (a, b) = if along_u {
                        (self.pt(t, fixed, 1.0), self.pt(t, fixed, hf - 1.0))
                    } else {
                        (self.pt(fixed, t, 1.0), self.pt(fixed, t, hf - 1.0))
                    };
                    self.line_g(vec![a, b], &ink, self.main_gate);
                }
            }
        }
        Ok(())
    }

    // ── 9 · partition ───────────────────────────────────────────────────────

    fn partition(&mut self) -> Result<(), PropError> {
        let (w, _, h) = self.size_i();
        if w >= 3 && self.s.door_gap.is_none() {
            return Err(self.err("door_gap", "a partition 3 or more wide needs door_gap = U0..U1 (it must not wall off the room)"));
        }
        if let Some((a, b)) = self.s.door_gap {
            if b >= w {
                return Err(self.err(
                    "door_gap",
                    format!("gap {a}..{b} runs past the partition width {w}"),
                ));
            }
        }
        let m = if self.s.mats.iter().any(|(s, _, _)| *s == MatSlot::Body) {
            self.mat(MatSlot::Body)?
        } else {
            match self.s.style {
                PropStyle::Lab => self.preset("fabric"),
                PropStyle::Range => self.preset("lattice"),
                _ => self.mat(MatSlot::Face)?,
            }
        };
        let ink = self.ink()?;
        for u in 0..w {
            if self.s.door_gap.is_some_and(|(a, b)| (a..=b).contains(&u)) {
                continue;
            }
            for lw in 0..h {
                self.full(self.cell(u, 0, lw), &m)?;
            }
        }
        let (wf, hf) = (w as f32, h as f32);
        self.line(
            vec![self.pt(0.0, 0.5, hf + EPS), self.pt(wf, 0.5, hf + EPS)],
            &ink,
        );
        if let Some((a, b)) = self.s.door_gap {
            for x in [a as f32, (b + 1) as f32] {
                self.line(vec![self.pt(x, -EPS, 0.0), self.pt(x, -EPS, hf)], &ink);
            }
        }
        Ok(())
    }

    // ── 10 · debris ─────────────────────────────────────────────────────────

    fn debris(&mut self) -> Result<(), PropError> {
        let (w, d, _) = self.size_i();
        let n = self.s.count.unwrap_or(((w * d) / 3).max(1) as u32);
        let dead = self.mat(MatSlot::Dead)?;
        let decay = self.preset("decay");
        let mut placed = 0;
        for i in 0..n * 6 {
            if placed >= n {
                break;
            }
            let hsh = hash3(self.s.seed, i, 31);
            let u = (hsh % w as u32) as i32;
            let v = ((hsh >> 8) % d as u32) as i32;
            let c = self.cell(u, v, 0);
            // Keep-out cells are skipped by construction (§3.8 #10).
            if self.taken.contains(&c)
                || self.cx.ko.get(c).is_some()
                || !self.in_bounds(c)
                || self.cx.base.cell_at(c.0, c.1, c.2) != CellKind::Empty
                || self.cx.occ.contains_key(&c)
            {
                continue;
            }
            // Rubble lies at the eye slab, where RampZ/WedgeXY cells render as column
            // artefacts; chunks are Full cells varied by material only (documented gap).
            let shape = Shape::Full;
            let m = if (hsh >> 24) & 1 == 0 {
                dead.clone()
            } else {
                decay.clone()
            };
            if self.solid(c, shape, Some(m))? {
                placed += 1;
            }
        }
        if placed < n {
            self.issue(
                IssueLevel::Warn,
                format!("debris: placed {placed} of {n} pieces (the box is crowded)"),
                None,
            );
        }
        Ok(())
    }

    // ── 11 · cctv ───────────────────────────────────────────────────────────

    fn cctv(&mut self) -> Result<(), PropError> {
        let c = {
            let a = self.at();
            (a.x, a.y, a.z)
        };
        let lv = self.cx.level;
        let look = self.s.look.unwrap_or(Vec3::new(
            lv.player.x as f32 + 0.5,
            lv.player.y as f32 + 0.5,
            lv.player.z as f32 + 0.6,
        ));
        let wall = |dx: i32, dy: i32| self.solid_at((c.0 + dx, c.1 + dy, c.2));
        let mut wx = if wall(1, 0) {
            1
        } else if wall(-1, 0) {
            -1
        } else {
            0
        };
        let mut wy = if wall(0, 1) {
            1
        } else if wall(0, -1) {
            -1
        } else {
            0
        };
        if wx == 0 && wy == 0 {
            return Err(self.err(
                "at",
                "cctv must sit against a wall (no solid neighbour at its z)",
            ));
        }
        let centre = v3(c) + Vec3::new(0.5, 0.5, 0.5);
        if wx == 0 {
            wx = if look.x > centre.x { -1 } else { 1 };
        }
        if wy == 0 {
            wy = if look.y > centre.y { -1 } else { 1 };
        }
        let trim = self.mat(MatSlot::Trim)?;
        self.solid(c, wedge_toward((wx, wy)), Some(trim))?;
        let ink = self.ink()?;
        let p0 = centre + Vec3::new(-wx as f32, -wy as f32, 0.0) * 0.2 - Vec3::Z * 0.1;
        let dir = (look - p0).normalize_or_zero();
        self.line(vec![p0, p0 + dir * 0.6], &ink);
        // The tally: lit while unsolved, dark on `door` (unless `power` says otherwise).
        let gate = if self.s.key_lines.contains_key("power") {
            self.gate_with(&self.s.power.clone(), self.s.invert, Some(TALLY_RGB))?
        } else {
            self.gate_with(&PowerSpec::Door, true, Some(TALLY_RGB))?
        };
        let dark = self.flat(shade(TALLY_RGB, 0.3));
        let t0 = p0 + Vec3::Z * 0.18;
        self.line_g(vec![t0, t0 + Vec3::new(0.07, 0.07, 0.0)], &dark, gate);
        Ok(())
    }

    // ── 12 · wall_panel ─────────────────────────────────────────────────────

    fn wall_panel(&mut self) -> Result<(), PropError> {
        let (w, _, h) = self.size_i();
        let skin = self.s.mode == PropMode::Skin;
        let face = match self.s.face.as_deref() {
            Some("codewall") => self.mats.resolve_value("codewall").unwrap_or_default(),
            Some(p) => self.preset(p),
            None => self.mat(MatSlot::Face)?,
        };
        let dead = self.mat(MatSlot::Dead)?;
        let lit_static = self.mat(MatSlot::Lit)?;
        let stat = self.preset("static_noise");
        let ink = self.ink()?;
        let gate = self.main_gate;
        for u in 0..w {
            for lw in 0..h {
                let c = self.cell(u, 0, lw);
                let r = unit3(self.s.seed, (u * 97 + lw) as u32, 13);
                let pick = match self.s.screen {
                    Screen::None => 0,
                    Screen::Live => 1,
                    Screen::Dead => 2,
                    Screen::Mixed if r < 0.5 => 1,
                    Screen::Mixed if r < 0.8 => 2,
                    Screen::Mixed => 3,
                };
                match pick {
                    1 if !skin && gate.is_some() => {
                        // Bank mode: a live screen is an R1 slot (dark = empty bezel).
                        self.part(c, gate.unwrap_or_default())?;
                    }
                    _ => {
                        let m = match pick {
                            0 => face.clone(),
                            1 => {
                                if self.lit(gate) {
                                    gate.map_or(lit_static.clone(), |g| self.gate_mats[g].1.clone())
                                } else {
                                    dead.clone()
                                }
                            }
                            2 => dead.clone(),
                            _ => stat.clone(),
                        };
                        if skin {
                            self.skin(c, &m);
                        } else {
                            self.full(c, &m)?;
                        }
                    }
                }
            }
        }
        let (wf, hf) = (w as f32, h as f32);
        let vf = -EPS;
        let bez = closed(vec![
            self.pt(0.0, vf, 0.0),
            self.pt(wf, vf, 0.0),
            self.pt(wf, vf, hf),
            self.pt(0.0, vf, hf),
        ]);
        self.line(bez, &ink);
        if !skin && !self.solid_at(self.cell(0, 1, 0)) {
            self.issue(
                IssueLevel::Warn,
                "wall_panel: no wall behind the bank",
                Some(self.cell(0, 1, 0)),
            );
        }
        if self.s.emitter {
            let o = self.pt(wf * 0.5, -0.15, hf * 0.5);
            let e = haze(o, fvec(self.f()));
            self.emitter(e.clone(), e, None);
        }
        Ok(())
    }

    // ── 13 · notice_board ───────────────────────────────────────────────────

    fn notice_board(&mut self) -> Result<(), PropError> {
        let (w, _, _) = self.size_i();
        let body = self.mat(MatSlot::Body)?;
        let ink = self.ink()?;
        let host = self.note_host(w)?;
        for u in 0..w {
            for lw in 0..2 {
                let m = if host == Some((u, lw)) {
                    None
                } else {
                    Some(body.clone())
                };
                self.solid(self.cell(u, 0, lw), Shape::Full, m)?;
            }
        }
        let wf = w as f32;
        let frame = closed(vec![
            self.pt(0.04, -EPS, 0.04),
            self.pt(wf - 0.04, -EPS, 0.04),
            self.pt(wf - 0.04, -EPS, 1.96),
            self.pt(0.04, -EPS, 1.96),
        ]);
        self.line(frame, &ink);
        Ok(())
    }

    // ── 14 · stair ──────────────────────────────────────────────────────────

    /// A straight run: tread 1 at `at`, climbing along `f`, `n` treads plus the
    /// landing (`staircase::straight_staircase` with `base = at − f − ẑ`).
    fn stair_run(
        &mut self,
        at: IVec3,
        f: (i32, i32),
        n: u32,
        mat: &str,
    ) -> Result<Vec<Cell3>, PropError> {
        let base = at - IVec3::new(f.0, f.1, 1);
        let stamps = crate::staircase::straight_staircase(base, ramp_dir_of(f), n);
        let mut cells = Vec::new();
        for st in stamps {
            let c = (st.pos.x, st.pos.y, st.pos.z);
            self.solid(c, st.shape, Some(mat.to_string()))?;
            self.out.stair_cells.push(c);
            cells.push(c);
        }
        // Headroom: the generator clears nothing; a blocked tread is a FAIL.
        for &c in &cells {
            for dz in 1..=2 {
                let hc = (c.0, c.1, c.2 + dz);
                if self.in_bounds(hc) && self.solid_at(hc) {
                    self.issue(
                        IssueLevel::Fail,
                        format!(
                            "stair headroom: ({},{},{}) above the tread is not air",
                            hc.0, hc.1, hc.2
                        ),
                        Some(hc),
                    );
                }
            }
        }
        Ok(cells)
    }

    /// Stringer rails at [`RAIL_H`] along a straight run (open sides only).
    fn stair_rails(&mut self, at: IVec3, f: (i32, i32), n: u32, sides: &[f32], ink: &str) {
        let (fv, rv) = (fvec(f), fvec((f.1, -f.0)));
        let start = Vec3::new(at.x as f32 + 0.5, at.y as f32 + 0.5, at.z as f32) - fv * 0.5;
        let nf = n as f32;
        let mid = n as i32 / 2;
        for &s in sides {
            let probe = (
                at.x + f.0 * mid + (rv.x * s.signum()) as i32,
                at.y + f.1 * mid + (rv.y * s.signum()) as i32,
                at.z + mid,
            );
            if self.solid_at(probe) {
                continue;
            }
            let off = rv * s;
            self.line(
                vec![
                    start + off + Vec3::Z * RAIL_H,
                    start + fv * nf + off + Vec3::Z * (nf + RAIL_H),
                    start + fv * (nf + 1.0) + off + Vec3::Z * (nf + RAIL_H),
                ],
                ink,
            );
        }
    }

    fn stair(&mut self) -> Result<(), PropError> {
        let rise = self.s.rise.unwrap_or(self.cx.level.storey_height());
        let at = self.at();
        let f = self.f();
        let body = self.mat(MatSlot::Body)?;
        let ink = self.ink()?;
        match self.s.form {
            StairForm::Straight => {
                self.stair_run(at, f, rise, &body)?;
                self.stair_rails(at, f, rise, &[-0.45, 0.45], &ink);
            }
            StairForm::Switchback => {
                if rise < 2 {
                    return Err(self.err("rise", "a switchback needs rise >= 2"));
                }
                let n1 = rise.div_ceil(2);
                let n2 = rise / 2;
                let r = self.r();
                // Run 1 up along f; its landing plus the cell to its right form
                // the 2×1 mid landing; run 2 climbs back along −f one column over.
                let c1 = self.stair_run(at, f, n1, &body)?;
                let land = *c1.last().unwrap_or(&(at.x, at.y, at.z));
                let land2 = (land.0 + r.0, land.1 + r.1, land.2);
                self.full(land2, &body)?;
                self.out.stair_cells.push(land2);
                let back = (-f.0, -f.1);
                let at2 = IVec3::new(land2.0 - f.0, land2.1 - f.1, land2.2 + 1);
                self.stair_run(at2, back, n2, &body)?;
                self.stair_rails(at, f, n1, &[-0.45], &ink);
                self.stair_rails(at2, back, n2, &[-0.45], &ink);
            }
            StairForm::Spiral => {
                let r = self.s.radius;
                let per = 8 * r as u32;
                if !rise.is_multiple_of(per) || rise == 0 {
                    let near = ((rise as f32 / per as f32).round().max(1.0) as u32) * per;
                    return Err(self.err(
                        "rise",
                        format!("a spiral of radius {r} rises 8·r = {per} per turn; rise {rise} is not a whole number of turns (nearest valid rise: {near})"),
                    ));
                }
                let stamps = crate::staircase::spiral_staircase(
                    at,
                    r,
                    rise / per,
                    self.cx.level.storey_height(),
                );
                for st in stamps {
                    let c = (st.pos.x, st.pos.y, st.pos.z);
                    self.solid(c, st.shape, Some(body.clone()))?;
                    self.out.stair_cells.push(c);
                }
            }
        }
        Ok(())
    }

    // ── 15 · gallery ────────────────────────────────────────────────────────

    fn gallery(&mut self) -> Result<(), PropError> {
        let (w, d, _) = self.size_i();
        let at = self.at();
        if at.z < 3 || at.z == 2 {
            return Err(self.err(
                "at",
                "a gallery deck sits at z >= 3 (never on the z=2 node/pipe plane)",
            ));
        }
        let lv = self.cx.level;
        let dress = self.s.mode == PropMode::Dress;
        let trim = self.mat(MatSlot::Trim)?;
        let ink = self.ink()?;
        let mut deck: HashSet<(i32, i32)> = HashSet::new();
        for u in 0..w {
            for v in 0..d {
                let c = self.cell(u, v, 0);
                if lv
                    .nodes
                    .iter()
                    .any(|n| n.x == c.0 && n.y == c.1 && n.z < at.z)
                {
                    return Err(self.err(
                        "at",
                        format!("deck cell ({},{},{}) sits over a node", c.0, c.1, c.2),
                    ));
                }
                if lv
                    .pipes
                    .iter()
                    .filter_map(|p| p.switch_cell.and_then(|i| p.path.get(i)))
                    .any(|s| s.0 == c.0 && s.1 == c.1 && s.2 < at.z)
                {
                    return Err(self.err(
                        "at",
                        format!(
                            "deck cell ({},{},{}) sits over a switch column",
                            c.0, c.1, c.2
                        ),
                    ));
                }
                for n in &lv.nodes {
                    if (n.x - c.0).abs() <= 1
                        && (n.y - c.1).abs() <= 1
                        && (at.z - n.z).abs() <= 2
                        && n.z != at.z
                    {
                        self.issue(
                            IssueLevel::Fail,
                            format!(
                                "deck cell ({},{},{}) is within jump reach of node {}",
                                c.0, c.1, c.2, n.name
                            ),
                            Some(c),
                        );
                    }
                }
                if dress {
                    if !self.solid_at(c) {
                        self.skin_missed += 1;
                    }
                } else {
                    self.full(c, &trim)?;
                }
                deck.insert((c.0, c.1));
            }
        }
        // The optional stair; its landing must touch the deck.
        let mut landing: Option<Cell3> = None;
        if let Some((sat, sf, srise)) = self.s.stair {
            let fv = sf.vec();
            let land = (
                sat.x + fv.0 * srise as i32,
                sat.y + fv.1 * srise as i32,
                sat.z + srise as i32 - 1,
            );
            let touches = land.2 == at.z
                && [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|(dx, dy)| deck.contains(&(land.0 + dx, land.1 + dy)));
            if !touches {
                return Err(self.err(
                    "stair",
                    format!(
                        "the stair landing ({},{},{}) does not touch the deck at z={}",
                        land.0, land.1, land.2, at.z
                    ),
                ));
            }
            let body = self.mat(MatSlot::Body)?;
            self.stair_run(sat, fv, srise, &body)?;
            self.stair_rails(sat, fv, srise, &[-0.45, 0.45], &ink);
            landing = Some(land);
        }
        // Rails + balusters on every open edge; posts every `post_every` cells.
        let top = at.z as f32 + 1.0;
        let edges: [(i32, (i32, i32)); 4] = [(w, (0, -1)), (w, (0, d)), (d, (-1, 0)), (d, (w, 0))];
        let mut posts: Vec<(i32, i32)> = Vec::new();
        for (ei, (len, _)) in edges.iter().enumerate() {
            let mut run: Vec<(f32, f32, f32, f32)> = Vec::new(); // (U0,V0,U1,V1)
            for t in 0..*len {
                let (u, v, ou, ov) = match ei {
                    0 => (t, 0, t, -1),
                    1 => (t, d - 1, t, d),
                    2 => (0, t, -1, t),
                    _ => (w - 1, t, w, t),
                };
                let out_c = self.cell(ou, ov, 0);
                let out_up = self.cell(ou, ov, 1);
                let against = self.solid_at(out_c)
                    || self.solid_at(out_up)
                    || landing.is_some_and(|l| l.0 == out_c.0 && l.1 == out_c.1);
                if against {
                    continue;
                }
                let tf = t as f32;
                let seg = match ei {
                    0 => (tf, 0.1, tf + 1.0, 0.1),
                    1 => (tf, d as f32 - 0.1, tf + 1.0, d as f32 - 0.1),
                    2 => (0.1, tf, 0.1, tf + 1.0),
                    _ => (w as f32 - 0.1, tf, w as f32 - 0.1, tf + 1.0),
                };
                match run.last_mut() {
                    Some(last)
                        if (last.2 - seg.0).abs() < 1e-4 && (last.3 - seg.1).abs() < 1e-4 =>
                    {
                        last.2 = seg.2;
                        last.3 = seg.3;
                    }
                    _ => run.push(seg),
                }
                let cadence = t % self.s.post_every == 0;
                let end = t == len - 1;
                let stand = match &self.s.posts {
                    GalleryPosts::Auto => cadence || end,
                    GalleryPosts::NoEnds => cadence && !end && t != 0,
                    GalleryPosts::None | GalleryPosts::At(_) => false,
                };
                if stand {
                    posts.push((u, v));
                }
            }
            for (u0, v0, u1, v1) in run {
                let a = self.pt(u0, v0, 0.0);
                let b = self.pt(u1, v1, 0.0);
                let a = Vec3::new(a.x, a.y, top + RAIL_H);
                let b = Vec3::new(b.x, b.y, top + RAIL_H);
                self.line(vec![a, b], &ink);
                let n = a.distance(b).round() as i32;
                for k in 0..=n {
                    let p = a.lerp(b, if n == 0 { 0.0 } else { k as f32 / n as f32 });
                    self.line(vec![Vec3::new(p.x, p.y, top), p], &ink);
                }
            }
        }
        if let GalleryPosts::At(list) = &self.s.posts {
            for &(u, v) in list {
                if !(0..w).contains(&u) || !(0..d).contains(&v) {
                    return Err(self.err(
                        "posts",
                        format!("post ({u} {v}) is outside the {w}x{d} deck"),
                    ));
                }
                posts.push((u, v));
            }
        }
        posts.sort_unstable();
        posts.dedup();
        for (u, v) in posts {
            let c = self.cell(u, v, 0);
            // The post stands on whatever is really under it — the room's floor OR a
            // prop stamped before this gallery (a console, a crate stack) — and is
            // dropped rather than driven through one.
            let fl = self.floor_or_prop_at((c.0, c.1, at.z - 1));
            let col: Vec<Cell3> = (fl + 1..at.z).map(|z| (c.0, c.1, z)).collect();
            if col.is_empty()
                || col
                    .iter()
                    .any(|&p| self.cx.ko.get(p).is_some() || self.solid_or_prop(p))
            {
                self.skipped
                    .push((c, "gallery post: no clear column to the floor".into()));
                continue;
            }
            for p in col {
                self.full(p, &trim)?;
            }
        }
        if dress && self.skin_missed > 0 {
            let n = self.skin_missed;
            self.skin_missed = 0;
            self.issue(
                IssueLevel::Warn,
                format!("mode = dress: {n} deck cell(s) have no existing geometry"),
                None,
            );
        }
        Ok(())
    }

    // ── routes (16, 17, 19) ─────────────────────────────────────────────────

    /// The route cells of a linear kind: the followed pipe's path (from its switch
    /// for `cable_run`) lifted by `lift`, or the axis-aligned run `at → to`.
    fn route(&self, from_switch: bool) -> Result<Vec<Cell3>, PropError> {
        if let Some((a, b)) = &self.s.follow {
            let p = self
                .cx
                .level
                .pipes
                .iter()
                .find(|p| &p.from == a && &p.to == b)
                .ok_or_else(|| self.err("follow", format!("unknown pipe '{a}->{b}'")))?;
            let start = if from_switch {
                p.switch_cell.unwrap_or(0)
            } else {
                0
            };
            return Ok(p.path[start.min(p.path.len())..]
                .iter()
                .map(|c| (c.0, c.1, c.2 + self.s.lift))
                .collect());
        }
        let a = self.at();
        let t = self.s.to.unwrap_or(self.s.at);
        let t = IVec3::new(t.x.floor() as i32, t.y.floor() as i32, t.z.floor() as i32);
        let diff = [(a.x != t.x), (a.y != t.y), (a.z != t.z)]
            .iter()
            .filter(|&&b| b)
            .count();
        if diff > 1 {
            return Err(self.err(
                "to",
                "an at → to run must be axis-aligned (differ in one coordinate)",
            ));
        }
        let n = (t - a).abs().max_element();
        let step = (t - a).signum();
        Ok((0..=n)
            .map(|i| {
                let c = a + step * i;
                (c.x, c.y, c.z)
            })
            .collect())
    }

    // ── 16 · cable_tray ─────────────────────────────────────────────────────

    fn cable_tray(&mut self) -> Result<(), PropError> {
        let cells = self.route(false)?;
        let trim = self.mat(MatSlot::Trim)?;
        let ink = self.ink()?;
        // Contiguous runs; a keep-out cell breaks the tray.
        let mut runs: Vec<Vec<Cell3>> = vec![Vec::new()];
        for c in cells {
            if self.cx.ko.get(c).is_some()
                || !self.in_bounds(c)
                || self.cx.base.cell_at(c.0, c.1, c.2) != CellKind::Empty
            {
                if runs.last().is_some_and(|r| !r.is_empty()) {
                    runs.push(Vec::new());
                }
                continue;
            }
            self.full(c, &trim)?;
            if let Some(r) = runs.last_mut() {
                r.push(c);
            }
        }
        for run in runs.into_iter().filter(|r| !r.is_empty()) {
            let centres: Vec<Vec3> = run
                .iter()
                .map(|&c| v3(c) + Vec3::new(0.5, 0.5, 0.0))
                .collect();
            for side in [-0.58f32, 0.58] {
                let mut pts = Vec::new();
                for (i, &p) in centres.iter().enumerate() {
                    let d = if i + 1 < centres.len() {
                        centres[i + 1] - p
                    } else if i > 0 {
                        p - centres[i - 1]
                    } else {
                        Vec3::X
                    };
                    let perp = if d.z.abs() > 0.5 {
                        Vec3::X
                    } else {
                        Vec3::new(-d.y, d.x, 0.0).normalize_or_zero()
                    };
                    pts.push(p + perp * side + Vec3::Z * 0.55);
                }
                let pts = self.sag_run(pts, 3, 0.25);
                self.line_g(pts, &ink, self.main_gate);
            }
            // Hangers every 3 cells, from the tray top to the ceiling.
            for (i, &c) in run.iter().enumerate().step_by(3) {
                let ceil = self.ceil_above(c);
                let (_, _, sz) = self.cx.base.size();
                if ceil < sz as i32 && ceil - c.2 <= 6 {
                    let p = centres[i];
                    self.line(
                        vec![p + Vec3::Z * 1.0, Vec3::new(p.x, p.y, ceil as f32)],
                        &ink,
                    );
                }
            }
        }
        Ok(())
    }

    // ── 17 · conduit ────────────────────────────────────────────────────────

    fn conduit(&mut self) -> Result<(), PropError> {
        let cells = self.route(false)?;
        let m = if self.s.mats.iter().any(|(s, _, _)| *s == MatSlot::Body) {
            self.mat(MatSlot::Body)?
        } else {
            match self.s.style {
                PropStyle::Lab | PropStyle::Range => self.preset("metal"),
                PropStyle::Vault | PropStyle::Archive => self.preset("rust"),
                _ => self.mat(MatSlot::Trim)?,
            }
        };
        let ink = self.ink()?;
        let mut all = cells.clone();
        let mut riser_ends = Vec::new();
        if matches!(self.s.riser, Riser::Start | Riser::Both) {
            riser_ends.push(cells[0]);
        }
        if matches!(self.s.riser, Riser::End | Riser::Both) {
            riser_ends.push(*cells.last().unwrap_or(&cells[0]));
        }
        for e in riser_ends {
            let fl = self.floor_at((e.0, e.1, e.2 - 1));
            for z in (fl + 1..e.2).rev() {
                all.push((e.0, e.1, z));
            }
        }
        let last = *cells.last().unwrap_or(&cells[0]);
        for &c in &all {
            let mm = if self.s.end_vent && c == last {
                self.preset("lattice")
            } else {
                m.clone()
            };
            self.full(c, &mm)?;
        }
        // Valve wheels (a cross in a ring) on the endpoints and every turn.
        let mut joints = vec![cells[0], last];
        for w in cells.windows(3) {
            let d1 = (w[1].0 - w[0].0, w[1].1 - w[0].1, w[1].2 - w[0].2);
            let d2 = (w[2].0 - w[1].0, w[2].1 - w[1].1, w[2].2 - w[1].2);
            if d1 != d2 {
                joints.push(w[1]);
            }
        }
        joints.dedup();
        for j in joints {
            let o = v3(j) + Vec3::new(0.5, 0.5, 1.12);
            let ring: Vec<Vec3> = (0..8)
                .map(|k| {
                    let a = k as f32 * PI / 4.0;
                    o + Vec3::new(a.cos(), a.sin(), 0.0) * 0.3
                })
                .collect();
            self.line(closed(ring), &ink);
            self.line(vec![o - Vec3::X * 0.3, o + Vec3::X * 0.3], &ink);
            self.line(vec![o - Vec3::Y * 0.3, o + Vec3::Y * 0.3], &ink);
        }
        if self.s.end_vent || self.s.emitter {
            let o = v3(last) + Vec3::new(0.5, 0.5, 1.1);
            self.emitter(steam(o), frost(o), self.main_gate);
        }
        Ok(())
    }

    // ── 18 · hanging_panels ─────────────────────────────────────────────────

    fn hanging_panels(&mut self) -> Result<(), PropError> {
        let (w, d, _) = self.size_i();
        let trim = self.mat(MatSlot::Trim)?;
        let ink = self.ink()?;
        let dirs = [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY];
        let mut gaps = Vec::new();
        let mut low = 0;
        let mut ceil_z = 0;
        for u in 0..w {
            for v in 0..d {
                let floor_c = self.cell(u, v, 0);
                let ceil = self.ceil_above(floor_c);
                let fl = self.floor_at(floor_c);
                let z = ceil - 1;
                ceil_z = ceil_z.max(ceil);
                if z - fl < 4 {
                    low += 1;
                    continue;
                }
                let c = (floor_c.0, floor_c.1, z);
                let key = (u * 97 + v) as u32;
                let r = unit3(self.s.seed, key, 5);
                if r < 0.6 {
                    self.full(c, &trim)?;
                } else if r < 0.85 {
                    gaps.push((c, ceil));
                } else {
                    let dir = dirs[(hash3(self.s.seed, key, 6) % 4) as usize];
                    self.solid(c, Shape::RampZ(dir), Some(trim.clone()))?;
                }
            }
        }
        if low > 0 {
            self.issue(
                IssueLevel::Warn,
                format!("{low} panel(s) skipped: the room is under 4 cells tall there"),
                None,
            );
        }
        let strands = 1 + hash3(self.s.seed, 0, 7) % 3;
        for (k, (c, ceil)) in gaps.iter().take(strands as usize).enumerate() {
            let top = v3(*c) + Vec3::new(0.5, 0.5, 0.0);
            let top = Vec3::new(top.x, top.y, *ceil as f32);
            let len = 1.2 + 0.8 * unit3(self.s.seed, k as u32, 8);
            let sway = (unit3(self.s.seed, k as u32, 9) - 0.5) * 0.2;
            let pts = (0..=4)
                .map(|i| {
                    let t = i as f32 / 4.0;
                    top + Vec3::new(sway * (t * PI).sin(), sway * 0.5 * (t * PI).sin(), -len * t)
                })
                .collect();
            self.line(pts, &ink);
        }
        if self.s.emitter {
            let c = self.cell(w / 2, d / 2, 0);
            let o = Vec3::new(c.0 as f32 + 0.5, c.1 as f32 + 0.5, ceil_z as f32 - 1.2);
            let e = dust(o);
            self.emitter(e.clone(), e, None);
        }
        Ok(())
    }

    // ── 19 · cable_run ──────────────────────────────────────────────────────

    fn cable_run(&mut self) -> Result<(), PropError> {
        let lv = self.cx.level;
        let (mut pts, hue) = if self.s.follow.is_some() {
            let cells = self.route(true)?;
            let (a, b) = self.s.follow.clone().unwrap_or_default();
            let pipe = lv.pipes.iter().find(|p| p.from == a && p.to == b);
            let hue = pipe.map_or(pack_lit_rgb(self.s.style), |p| {
                source_hue(
                    lv,
                    PowerSource::Pipe(
                        lv.pipes
                            .iter()
                            .position(|q| std::ptr::eq(q, p))
                            .unwrap_or(0),
                    ),
                    self.s.style,
                )
            });
            let mut pts: Vec<Vec3> = Vec::new();
            for c in cells {
                let fz = self.floor_at((c.0, c.1, c.2 - 1 - self.s.lift)) as f32 + 1.0 + EPS;
                let p = Vec3::new(c.0 as f32 + 0.5, c.1 as f32 + 0.5, fz);
                if let Some(&l) = pts.last() {
                    if (l.z - p.z).abs() > 1e-3 {
                        // Run up the wall / tread edge between the two cells.
                        let edge = (l + p) * 0.5;
                        pts.push(Vec3::new(edge.x, edge.y, l.z));
                        pts.push(Vec3::new(edge.x, edge.y, p.z));
                    }
                }
                pts.push(p);
            }
            (pts, hue)
        } else {
            let a = self.s.at;
            let t = self.s.to.unwrap_or(a);
            (
                vec![a, Vec3::new(t.x, a.y, a.z), t],
                pack_ink_hue(self.s.style),
            )
        };
        if let Some(e) = self.s.extend {
            if let Some(&l) = pts.last() {
                let ec = (e.x.floor() as i32, e.y.floor() as i32, e.z.floor() as i32);
                let fz = self.floor_at((ec.0, ec.1, ec.2 - 1)) as f32 + 1.0 + EPS;
                let t = Vec3::new(ec.0 as f32 + 0.5, ec.1 as f32 + 0.5, fz);
                pts.push(Vec3::new(t.x, l.y, l.z));
                pts.push(Vec3::new(t.x, t.y, l.z));
                if (t.z - l.z).abs() > 1e-3 {
                    pts.push(t);
                }
            }
        }
        let pts = simplify(pts);
        let ink = if self.s.mats.iter().any(|(s, _, _)| *s == MatSlot::Ink) {
            self.ink()?
        } else {
            self.flat(shade(hue, 0.55))
        };
        let n = self.s.bundle.max(1);
        for k in 0..n {
            let o = (k as f32 - (n - 1) as f32 * 0.5) * 0.12;
            let off = Vec3::new(o, o, 0.0);
            self.line_g(pts.iter().map(|p| *p + off).collect(), &ink, self.main_gate);
        }
        Ok(())
    }

    // ── 20 · cable_drop ─────────────────────────────────────────────────────

    fn cable_drop(&mut self) -> Result<(), PropError> {
        let a = self.s.at;
        let t = self.s.to.unwrap_or(a - Vec3::Z);
        let ink = self.ink()?;
        let n = self.s.strands.max(1);
        let horiz = Vec3::new(t.x - a.x, t.y - a.y, 0.0).length();
        for k in 0..n {
            let o = (k as f32 - (n - 1) as f32 * 0.5) * 0.08;
            let sway = (unit3(self.s.seed, k, 41) - 0.5) * 0.2;
            let pts = (0..=6)
                .map(|i| {
                    let tt = i as f32 / 6.0;
                    let bell = (tt * PI).sin();
                    let sag = if horiz > 0.5 {
                        -0.25 * 4.0 * tt * (1.0 - tt)
                    } else {
                        0.0
                    };
                    a.lerp(t, tt) + Vec3::new(o + sway * bell, o * 0.5 - sway * 0.5 * bell, sag)
                })
                .collect();
            self.line_g(pts, &ink, self.main_gate);
        }
        Ok(())
    }

    // ── 21 · lamp_pendant ───────────────────────────────────────────────────

    fn lamp_pendant(&mut self) -> Result<(), PropError> {
        let a = self.s.at;
        let ac = (a.x.floor() as i32, a.y.floor() as i32, a.z.floor() as i32);
        let ceil = self.ceil_above(ac) as f32;
        let gate = self.main_gate;
        let colour = match self.s.light {
            LightMode::Existing => {
                let lv = self.cx.level;
                let found = lv
                    .lights
                    .iter()
                    .enumerate()
                    .filter(|(i, l)| {
                        !self.shared.existing_lights.contains(i) && l.pos.distance(a) < 0.75
                    })
                    .min_by(|x, y| x.1.pos.distance(a).total_cmp(&y.1.pos.distance(a)));
                let Some((i, l)) = found else {
                    return Err(self.err(
                        "light",
                        format!(
                            "light = existing, but no [light] within 0.75 of ({}, {}, {})",
                            a.x, a.y, a.z
                        ),
                    ));
                };
                self.shared.existing_lights.insert(i);
                l.color
            }
            LightMode::Yes => {
                self.shared.new_lights += 1;
                let total = self.cx.level.lights.len() + self.shared.new_lights;
                if total > PROP_MAX_LIGHTS {
                    self.issue(
                        IssueLevel::Fail,
                        format!("light = yes makes {total} lights; the renderer drops everything past {PROP_MAX_LIGHTS}"),
                        Some(ac),
                    );
                }
                let lit = PointLight {
                    pos: a - Vec3::Z * 0.3,
                    color: LAMP_RGB,
                    radius: 6.0,
                    intensity: 1.0,
                };
                let idx = self.out.lights.len();
                match gate {
                    Some(g) if !self.out.gates[g].initial_lit => {
                        self.out.lights.push(PointLight {
                            intensity: 0.0,
                            ..lit
                        });
                        self.out.gates[g].lights.push((idx, lit));
                    }
                    Some(g) => {
                        self.out.lights.push(lit);
                        self.out.gates[g].lights.push((idx, lit));
                    }
                    None => self.out.lights.push(lit),
                }
                LAMP_RGB
            }
            LightMode::No => 0,
        };
        let shade_mat = if colour == 0 {
            self.ink()?
        } else {
            self.flat(colour)
        };
        let ink = self.ink()?;
        let cone = self.s.shape.as_deref() == Some("cone");
        let cord_end = a.z + if cone { 0.25 } else { 0.18 };
        if ceil > cord_end {
            self.line(
                vec![Vec3::new(a.x, a.y, ceil), Vec3::new(a.x, a.y, cord_end)],
                &ink,
            );
        }
        let sq = |r: f32, z: f32| {
            closed(vec![
                Vec3::new(a.x - r, a.y - r, z),
                Vec3::new(a.x + r, a.y - r, z),
                Vec3::new(a.x + r, a.y + r, z),
                Vec3::new(a.x - r, a.y + r, z),
            ])
        };
        if cone {
            let apex = Vec3::new(a.x, a.y, cord_end);
            for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                self.line_g(
                    vec![apex, Vec3::new(a.x + dx * 0.25, a.y + dy * 0.25, a.z)],
                    &shade_mat,
                    gate,
                );
            }
            self.line_g(sq(0.25, a.z), &shade_mat, gate);
        } else {
            self.line_g(sq(0.2, a.z), &shade_mat, gate);
            self.line_g(sq(0.12, a.z + 0.18), &shade_mat, gate);
        }
        Ok(())
    }

    // ── 22 · furniture_trace ────────────────────────────────────────────────

    fn furniture_trace(&mut self) -> Result<(), PropError> {
        let [w, d, h] = self.s.size;
        let shape = self.s.shape.clone().unwrap_or_default();
        let (fv, rv) = (fvec(self.f()), fvec(self.r()));
        let (ex, ey) = if fv.x != 0.0 { (d, w) } else { (w, d) };
        let a = self.s.at;
        let centre = Vec3::new(a.x + ex * 0.5, a.y + ey * 0.5, a.z);
        let fl = centre - rv * (w * 0.5) + fv * (d * 0.5);
        let fr = Frame {
            o: fl,
            ru: rv,
            bv: -fv,
        };
        let ink = self.ink()?;
        if self.s.solid {
            return self.furniture_solid(&shape, fr, w, d, &ink);
        }
        self.furniture(&shape, fr, w, d, h, &ink);
        Ok(())
    }

    /// `solid = yes`: the piece is a LOW SOLID BLOCK — one `WallSolid` per cell
    /// whose centre the footprint covers
    /// (or, along an extent under a cell, the one cell holding its middle), in a
    /// z-band material ([`super::solid_furniture_material`]) the renderer draws solid
    /// only from the floor to [`super::solid_furniture_band`] with a bright top and
    /// lid. A chair keeps its back as strokes rising from the seat; the other shapes
    /// are the block alone.
    ///
    /// COLLISION: the cells are ordinary solid prop cells, so the piece blocks
    /// movement like a rack does (a full cell — the grid has no half-height walls to
    /// step onto), and every cell passes the same keep-out / Empty-only rules as any
    /// solid kind: entering a puzzle cell is a load error naming both.
    fn furniture_solid(
        &mut self,
        shape: &str,
        fr: Frame,
        w: f32,
        d: f32,
        ink: &str,
    ) -> Result<(), PropError> {
        let ink_rgb = self
            .def_of(ink)
            .map_or(super::pack_ink(self.s.style), |m| m.fg & 0xFF_FFFF);
        let corners = [
            fr.p(0.0, 0.0, 0.0),
            fr.p(w, 0.0, 0.0),
            fr.p(w, d, 0.0),
            fr.p(0.0, d, 0.0),
        ];
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in corners {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        // The cells whose CENTRE the footprint covers; an extent narrower than a cell
        // takes the one cell holding its middle (a 0.7-wide stool straddling a cell line
        // is ONE block, not four).
        let cells_of = |a: f32, b: f32| -> (i32, i32) {
            let (c0, c1) = ((a - 0.5).ceil() as i32, (b - 0.5).floor() as i32);
            if c0 <= c1 {
                (c0, c1)
            } else {
                let m = ((a + b) * 0.5).floor() as i32;
                (m, m)
            }
        };
        let (x0, x1) = cells_of(lo.x, hi.x);
        let (y0, y1) = cells_of(lo.y, hi.y);
        let above = ((lo.z - 0.5).floor() as i32).max(0);
        let fl = self.floor_at((x0, y0, above));
        // The block stands in the walking cell on top of the floor under its corner.
        let cz = if fl >= 0 { fl + 1 } else { lo.z.floor() as i32 };
        let m = self
            .mats
            .intern(super::solid_furniture_material(ink_rgb, shape));
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.full((x, y, cz), &m)?;
            }
        }
        if shape == "chair" {
            let seat = super::solid_furniture_band(shape) / TRACE_H;
            for u in [0.05, w - 0.05] {
                self.line(
                    vec![fr.ph(u, d - 0.05, seat), fr.ph(u, d - 0.05, 0.95)],
                    ink,
                );
            }
            self.line(
                vec![fr.ph(0.05, d - 0.05, 0.95), fr.ph(w - 0.05, d - 0.05, 0.95)],
                ink,
            );
            self.line(
                vec![fr.ph(0.05, d - 0.05, 0.70), fr.ph(w - 0.05, d - 0.05, 0.70)],
                ink,
            );
        }
        Ok(())
    }

    /// Draw one outline furniture piece in frame `fr` (front-left corner at `o`,
    /// `U` to the right, `V` to the back, heights above `o.z`).
    fn furniture(&mut self, shape: &str, fr: Frame, w: f32, d: f32, h: f32, ink: &str) {
        if matches!(
            shape,
            "table" | "bench" | "chair" | "cot" | "planter" | "pier" | "stack"
        ) {
            self.furniture_base(fr, w, d, ink);
        }
        let rect = |z: f32, i: f32| {
            closed(vec![
                fr.ph(i, i, z),
                fr.ph(w - i, i, z),
                fr.ph(w - i, d - i, z),
                fr.ph(i, d - i, z),
            ])
        };
        let legs = |this: &mut Self, top: f32, i: f32| {
            for (u, v) in [(i, i), (w - i, i), (w - i, d - i), (i, d - i)] {
                this.line(vec![fr.ph(u, v, 0.0), fr.ph(u, v, top)], ink);
            }
        };
        match shape {
            "table" => {
                self.line(rect(0.45, 0.0), ink);
                legs(self, 0.45, 0.08);
            }
            "bench" => {
                self.line(rect(0.30, 0.0), ink);
                legs(self, 0.30, 0.08);
            }
            "chair" => {
                self.line(rect(0.30, 0.0), ink);
                legs(self, 0.30, 0.05);
                // The back rises from the rear edge.
                for u in [0.05, w - 0.05] {
                    self.line(
                        vec![fr.ph(u, d - 0.05, 0.30), fr.ph(u, d - 0.05, 0.75)],
                        ink,
                    );
                }
                self.line(
                    vec![fr.ph(0.05, d - 0.05, 0.75), fr.ph(w - 0.05, d - 0.05, 0.75)],
                    ink,
                );
                self.line(
                    vec![fr.ph(0.05, d - 0.05, 0.55), fr.ph(w - 0.05, d - 0.05, 0.55)],
                    ink,
                );
            }
            "cot" => {
                self.line(rect(0.35, 0.0), ink);
                legs(self, 0.35, 0.06);
                self.line(
                    vec![fr.ph(0.1, d - 0.25, 0.42), fr.ph(w - 0.1, d - 0.25, 0.42)],
                    ink,
                );
                self.line(
                    vec![fr.ph(0.1, d * 0.5, 0.36), fr.ph(w - 0.1, d * 0.5, 0.36)],
                    ink,
                );
            }
            "planter" => {
                self.line(rect(0.0, 0.05), ink);
                self.line(rect(0.35, 0.05), ink);
                legs(self, 0.35, 0.05);
                let c = fr.ph(w * 0.5, d * 0.5, 0.35);
                let top = c + Vec3::Z * (0.55 * TRACE_H);
                self.line(vec![c, top], ink);
                self.line(
                    vec![
                        c + Vec3::Z * (0.25 * TRACE_H),
                        c + Vec3::Z * (0.35 * TRACE_H) + fr.ru * 0.15,
                    ],
                    ink,
                );
                self.line(
                    vec![
                        c + Vec3::Z * (0.4 * TRACE_H),
                        c + Vec3::Z * (0.5 * TRACE_H) - fr.ru * 0.15,
                    ],
                    ink,
                );
            }
            "stack" => {
                // A short shelf / crate stack: a box to 0.85 with two shelf lines.
                self.line(rect(0.0, 0.03), ink);
                self.line(rect(0.85, 0.03), ink);
                legs(self, 0.85, 0.03);
                for z in [0.30, 0.58] {
                    self.line(vec![fr.ph(0.03, 0.03, z), fr.ph(w - 0.03, 0.03, z)], ink);
                }
            }
            "rail" => {
                let v = d * 0.5;
                self.line(vec![fr.ph(0.0, v, 0.9), fr.ph(w, v, 0.9)], ink);
                let n = w.round().max(1.0) as i32;
                for k in 0..=n {
                    let u = w * k as f32 / n as f32;
                    self.line(vec![fr.ph(u, v, 0.0), fr.ph(u, v, 0.9)], ink);
                }
            }
            "frame" => {
                let hh = if h > 0.0 { h } else { 0.8 };
                let r = closed(vec![
                    fr.ph(0.0, 0.0, 0.0),
                    fr.ph(w, 0.0, 0.0),
                    fr.ph(w, 0.0, hh),
                    fr.ph(0.0, 0.0, hh),
                ]);
                self.line(r, ink);
                let c = fr.ph(w * 0.5, 0.0, hh * 0.5);
                let k = (w.min(hh)) * 0.25;
                self.line(vec![c - fr.ru * k, c + fr.ru * k], ink);
                self.line(vec![c - Vec3::Z * k, c + Vec3::Z * k], ink);
            }
            "pier" => {
                let n = (d / 0.25).round().max(1.0) as i32;
                for k in 0..=n {
                    let v = d * k as f32 / n as f32;
                    self.line(vec![fr.ph(0.0, v, 0.03), fr.ph(w, v, 0.03)], ink);
                }
                for (u, v) in [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)] {
                    self.line(vec![fr.ph(u, v, -0.3), fr.ph(u, v, 0.3)], ink);
                }
            }
            "coat" => {
                // Hangs against the back plane between 0.6 and 1.4.
                let v = d - 0.05;
                let cu = w * 0.5;
                self.line(
                    vec![
                        fr.ph(cu, v, 1.55),
                        fr.ph(cu, v, 1.42),
                        fr.ph(cu + 0.08, v, 1.42),
                    ],
                    ink,
                );
                self.line(
                    vec![
                        fr.ph(cu - 0.25, v, 1.32),
                        fr.ph(cu, v, 1.42),
                        fr.ph(cu + 0.25, v, 1.32),
                    ],
                    ink,
                );
                self.line(
                    vec![fr.ph(cu - 0.25, v, 1.32), fr.ph(cu - 0.28, v, 0.62)],
                    ink,
                );
                self.line(
                    vec![fr.ph(cu + 0.25, v, 1.32), fr.ph(cu + 0.28, v, 0.62)],
                    ink,
                );
                self.line(
                    vec![fr.ph(cu - 0.28, v, 0.62), fr.ph(cu + 0.28, v, 0.62)],
                    ink,
                );
                self.line(vec![fr.ph(cu, v, 1.40), fr.ph(cu, v, 0.70)], ink);
            }
            _ => {
                // hook
                let v = d - 0.05;
                let cu = w * 0.5;
                self.line(
                    vec![
                        fr.ph(cu, v, 1.45),
                        fr.ph(cu, v - 0.12, 1.45),
                        fr.ph(cu, v - 0.12, 1.55),
                    ],
                    ink,
                );
            }
        }
    }

    /// The solid low BASE under a floor-standing trace piece: a floor decal over the
    /// cells its footprint covers, in a dark wash of its ink
    /// ([`super::trace_base_material`]). A trace alone is a handful of 1-px strokes on
    /// a textured floor and reads as invisible from spawn distance; the base gives
    /// each chair, table, planter or pier a mass in the bg channel (§5 R6) that reads
    /// as its footprint. Floor cells only, so the piece stays non-colliding.
    fn furniture_base(&mut self, fr: Frame, w: f32, d: f32, ink: &str) {
        let ink_rgb = self
            .def_of(ink)
            .map_or(super::pack_ink(self.s.style), |m| m.fg & 0xFF_FFFF);
        let corners = [
            fr.p(0.0, 0.0, 0.0),
            fr.p(w, 0.0, 0.0),
            fr.p(w, d, 0.0),
            fr.p(0.0, d, 0.0),
        ];
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in corners {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        // A footprint edge sitting exactly on a cell line belongs to the cell inside.
        let x0 = (lo.x + 0.02).floor() as i32;
        let y0 = (lo.y + 0.02).floor() as i32;
        let x1 = ((hi.x - 0.02).floor() as i32).max(x0);
        let y1 = ((hi.y - 0.02).floor() as i32).max(y0);
        let above = ((lo.z - 0.5).floor() as i32).max(0);
        let fl = self.floor_at((x0, y0, above));
        if fl < 0 {
            return;
        }
        let m = self.mats.intern(super::trace_base_material(ink_rgb));
        self.out.decals.push(PropDecal {
            min: (x0, y0, fl),
            max: (x1, y1, fl),
            mat: m,
            prop: self.pi,
        });
    }

    // ── 23 · holo_display ───────────────────────────────────────────────────

    fn holo_display(&mut self) -> Result<(), PropError> {
        let (w, d, h) = self.size_i();
        let shape = self
            .s
            .shape
            .as_deref()
            .and_then(HoloShape::from_str_opt)
            .unwrap_or(HoloShape::Box);
        let mut lift = 0;
        if self.s.base {
            let plinth: Vec<Cell3> = (0..w)
                .flat_map(|u| (0..d).map(move |v| (u, v)))
                .map(|(u, v)| self.cell(u, v, 0))
                .collect();
            let fits = plinth.iter().all(|&c| {
                self.cx.ko.get(c).is_none()
                    && self.in_bounds(c)
                    && self.cx.base.cell_at(c.0, c.1, c.2) == CellKind::Empty
            });
            if fits {
                let trim = self.mat(MatSlot::Trim)?;
                for c in plinth {
                    self.full(c, &trim)?;
                }
                lift = 1;
            } else {
                self.issue(
                    IssueLevel::Warn,
                    "base = yes: the plinth would enter the keep-out; omitted",
                    None,
                );
            }
        }
        let a = self.cell(0, 0, lift);
        let b = self.cell(w - 1, d - 1, lift + h - 1);
        let min = (a.0.min(b.0), a.1.min(b.1), a.2.min(b.2));
        let max = (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2));
        // Tint: an authored `style` (number, hue name or #RRGGBB) wins; a powered
        // display with no style projects in its source's colour-law hue ("a node's
        // thought"), so the dome reads as that node's. Unpowered + unstyled keeps
        // the default hologram blue.
        let style = match (self.s.holo_style, self.main_gate) {
            (0, Some(g)) => re_core::encode_holo_rgb(self.out.gates[g].lit_rgb),
            (s, _) => s,
        };
        self.holo(shape, min, max, style, self.main_gate);
        Ok(())
    }

    // ── 24 · overgrowth ─────────────────────────────────────────────────────

    fn overgrowth(&mut self) -> Result<(), PropError> {
        let a = self.s.at;
        let [w, _, h0] = self.s.size;
        let h = self.s.height.unwrap_or(h0);
        let (fv, rv) = (fvec(self.f()), fvec(self.r()));
        let moss = if self.s.mats.iter().any(|(s, _, _)| *s == MatSlot::Ink) {
            self.ink()?
        } else {
            self.flat(MOSS_RGB)
        };
        let base = a + fv * 0.03;
        let seed = self.s.seed;
        let stems = 2 + hash3(seed, 0, 51) % 3;
        for s in 0..stems {
            let mut u = w * (s as f32 + 0.5) / stems as f32 + (unit3(seed, s, 52) - 0.5) * 0.2;
            let top = h * (0.6 + 0.4 * unit3(seed, s, 53));
            let steps = (top / 0.3).ceil().max(1.0) as u32;
            let mut pts = Vec::new();
            for k in 0..=steps {
                let z = (k as f32 * 0.3).min(top);
                if k > 0 {
                    u += (unit3(seed, s * 1000 + k, 54) - 0.5) * 0.24;
                }
                pts.push(base + rv * u + Vec3::Z * z);
            }
            // Leaf ticks every 0.3, alternating sides.
            for (k, &p) in pts.iter().enumerate().skip(1) {
                let side = if k % 2 == 0 { 1.0 } else { -1.0 };
                self.line(vec![p, p + rv * (0.12 * side) + Vec3::Z * 0.08], &moss);
            }
            // One branch from a seeded point on the stem.
            if pts.len() > 3 {
                let at = 1 + (hash3(seed, s, 55) as usize % (pts.len() - 2));
                let p = pts[at];
                let side = if hash3(seed, s, 56) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                self.line(
                    vec![
                        p,
                        p + rv * (0.25 * side) + Vec3::Z * 0.2,
                        p + rv * (0.4 * side) + Vec3::Z * 0.45,
                    ],
                    &moss,
                );
            }
            self.line(pts, &moss);
        }
        if self.s.floor {
            let c0 = (a.x.floor() as i32, a.y.floor() as i32, a.z.floor() as i32);
            let far = a + rv * w + fv * 0.9;
            let c1 = (far.x.floor() as i32, far.y.floor() as i32, c0.2);
            let fl = self.floor_at((c0.0, c0.1, c0.2 - 1));
            let m = self.preset("moss");
            self.out.decals.push(PropDecal {
                min: (c0.0.min(c1.0), c0.1.min(c1.1), fl),
                max: (c0.0.max(c1.0), c0.1.max(c1.1), fl),
                mat: m,
                prop: self.pi,
            });
        }
        if self.s.emitter {
            let e = spores(base + rv * (w * 0.5) + fv * 0.5 + Vec3::Z * 0.8);
            self.emitter(e.clone(), e, None);
        }
        Ok(())
    }
}

/// The ink hue of a pack (for a free `cable_run` with no pipe to take a hue from).
fn pack_ink_hue(style: PropStyle) -> u32 {
    super::pack_ink(style)
}
