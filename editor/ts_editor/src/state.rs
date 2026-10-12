//! The editor's data model + UI cursor state.
use bevy::prelude::*;
use re_core::{DiagXY, RampDir, Shape};
use re_world::{CameraPose, Map};

/// Per-frame mouse-look tracker for the preview free-look DUAL PATH (mirrors the
/// game's `MouseTracker`). `last` holds the previous absolute cell `(col, row)`
/// from the crossterm `MouseMessage` stream so a per-step delta can be differenced
/// (path 2 — Windows Terminal / Wayland / grab-off). `grab_yielded` records that
/// the OS cursor-grab (path 1) actually produced a delta at least once since
/// capture began; while it has, a `poll_delta()` returning `None` is "a real grab,
/// just no motion" and we DON'T fall through to the cell path. Until it has (the
/// Windows-Terminal case where the grab is nominally `active` but never yields),
/// we DO fall through, which is the whole point of the fix.
#[derive(Resource, Default)]
pub struct EditorMouseTracker {
    /// Previous absolute cell position from the crossterm motion stream.
    pub last: Option<(u16, u16)>,
    /// `true` once the OS grab has yielded a real delta since capture started.
    pub grab_yielded: bool,
}

/// The eight authorable 45° orientations, in the order the shape brush cycles
/// through them: the four [`Shape::WedgeXY`] diagonals (NE, NW, SE, SW) then the
/// four [`Shape::RampZ`] rise directions (+X, -X, +Y, -Y). Wrapping.
pub const SHAPE_CYCLE: [Shape; 8] = [
    Shape::WedgeXY(DiagXY::NE),
    Shape::WedgeXY(DiagXY::NW),
    Shape::WedgeXY(DiagXY::SE),
    Shape::WedgeXY(DiagXY::SW),
    Shape::RampZ(RampDir::PosX),
    Shape::RampZ(RampDir::NegX),
    Shape::RampZ(RampDir::PosY),
    Shape::RampZ(RampDir::NegY),
];

/// Step the shape selection by `delta` (±1) through [`SHAPE_CYCLE`], wrapping.
/// A shape not in the cycle (e.g. `Full`) restarts at index 0.
pub fn cycle_shape(current: Shape, delta: i32) -> Shape {
    let n = SHAPE_CYCLE.len() as i32;
    match SHAPE_CYCLE.iter().position(|&s| s == current) {
        // In the cycle: step by `delta`, wrapping.
        Some(i) => {
            let next = (i as i32 + delta).rem_euclid(n) as usize;
            SHAPE_CYCLE[next]
        }
        // Not in the cycle (e.g. Full): restart at the first orientation.
        None => SHAPE_CYCLE[0],
    }
}

/// A short readable label for a [`Shape`] orientation, for the panel readout.
pub fn shape_label(shape: Shape) -> &'static str {
    match shape {
        Shape::Full => "full",
        Shape::WedgeXY(DiagXY::NE) => "wedge NE",
        Shape::WedgeXY(DiagXY::NW) => "wedge NW",
        Shape::WedgeXY(DiagXY::SE) => "wedge SE",
        Shape::WedgeXY(DiagXY::SW) => "wedge SW",
        Shape::RampZ(RampDir::PosX) => "ramp +X",
        Shape::RampZ(RampDir::NegX) => "ramp -X",
        Shape::RampZ(RampDir::PosY) => "ramp +Y",
        Shape::RampZ(RampDir::NegY) => "ramp -Y",
    }
}

/// A placeable entity kind for the OBJECT SELECTOR. `o` enters/cycles the
/// selector; `space` places the current kind at the cursor (or selects an
/// existing one for editing).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntityKind {
    /// The single win-condition door anchor (`level.door`).
    Door,
    /// The player spawn pose (`level.player`), with a cyclable facing.
    Spawn,
    /// The pedestal output volume (`level.pedestal`), with a growable extent.
    Pedestal,
    /// A named PART (`level.parts`): a `w*h*d` voxel volume with a solid mask —
    /// the pedestal OUTPUT model that an `[emit NODE=1] parts = …` clause lights
    /// when a node fires. Its size/offset/colour are focused-field editable and
    /// its solid mask is authored in a dedicated mask-edit modal (`q`).
    Part,
    /// A puzzle node (`level.nodes`), with cyclable io + threshold.
    Node,
    /// A particle emitter (`level.emitters`), with a full focused-field parameter
    /// editor (rate/speed/spread/lifetime/size/growth/direction/color_mode/…).
    Emitter,
    /// A trigger FIELD (`level.triggers`): a box region + an action (ActivateNode /
    /// Event) + an `once` flag. The region corners, action type + target/label, and
    /// `once` are all focused-field editable (see [`EntityField`]).
    Trigger,
    /// A PIPE (`level.pipes`): a wired connection between two nodes carrying signal.
    /// Pipes are NOT placed at a cursor cell like the other kinds — they are wired
    /// by a two-node connect flow (the `P` key → pick source, pick target). This
    /// variant exists so the focused-field editor (`state`/`switch`/`density`/
    /// `color`) and the object tree can select + edit an existing pipe.
    Pipe,
    /// A ZONE INPUT (`level.zone_inputs`): a box region that drives a named INPUT
    /// node HIGH while the player stands inside it. Like a [`EntityKind::Trigger`]
    /// it is a REGION entity — its six region corners are focused-field editable —
    /// plus a linked input-node selector (the `node` field, by name).
    ZoneInput,
    /// A NOTE (`level.notes`): a POINT placard carrying verbatim body text. Its
    /// kind (Researcher / SystemLog / ModelThought), cardinal facing, position, and
    /// a single-line body buffer are all focused-field editable.
    Note,
    /// A WALL DECAL (`level.decals`): a box region painted with a surface texture
    /// (matrix-rain / circuit-trace / plain solid). Like a [`EntityKind::Trigger`]
    /// or [`EntityKind::ZoneInput`] it is a REGION entity — its six region corners
    /// are focused-field editable — plus a cyclable `surface`, a `density` scalar,
    /// and two palette colours (`color head` / `color tail`).
    WallDecal,
    /// A HOLOGRAM object (`level.holos`): a box region filled by a chosen
    /// [`re_world::HoloShape`] (Box / Pyramid / Sphere) and stamped as
    /// `CellKind::Hologram`. Like a [`EntityKind::Trigger`] / [`EntityKind::ZoneInput`]
    /// / [`EntityKind::WallDecal`] it is a REGION entity — its six region corners are
    /// focused-field editable — plus a cyclable `shape` and a numeric `style` tag.
    Holo,
    /// A WARP PORTAL (`level.portals`): a PAIR of linked box regions (Region A and
    /// Region B) that re-base a ray crossing one onto the other. Unlike every other
    /// region entity it carries TWO regions — both A's six corners and B's six
    /// corners are focused-field editable — plus an extra-yaw `rotate` (degrees,
    /// A→B) and an optional `skin` material name (empty ⇒ none). The cell brush
    /// (`CellKind::Portal`, glyph `@`) is separate from this paired entity.
    Portal,
    /// A SCALABLE MATERIAL OVERLAY ZONE (`level.overlay.zones`): a box region that
    /// binds a material name and a scale rule (None / Crisp / SelfSimilar), plus an
    /// optional list of inset references. Like the other region entities its six
    /// region corners are focused-field editable — plus a material-name cycle, a
    /// scale-rule cycle with three SelfSimilar sub-fields (arms / octave_warp /
    /// twist), and an editable inset-name membership list (cursor + `c` toggle).
    Zone,
    /// A SCALABLE MATERIAL OVERLAY LINE (`level.overlay.lines`): a world-space
    /// polyline (waypoints appended at the cursor like a pipe path) carrying a
    /// line-material name and a scale rule (None / Crisp / SelfSimilar with the same
    /// arms / octave_warp / twist sub-fields as [`EntityKind::Zone`]).
    Line,
    /// A SCALABLE MATERIAL OVERLAY INSET / "врезка" (`level.overlay.insets`): a box
    /// region at an inclusive octave range `(lo, hi)` that overrides the scale rule
    /// with a referenced child scene. Its octave lo/hi + six region corners are
    /// focused-field editable, plus a `child` scene-path text buffer (typed like a
    /// note body).
    Inset,
    /// A STANDALONE SOUND FIELD (`level.field_defs`, `FieldHost::Standalone`):
    /// a `[field NAME]` region that bends the level's audio mix. The editor
    /// authors the POINT shape (`at`, like a Node) and the ROOM shape (no
    /// geometry); a field ATTACHED inline to a node/pipe/door (`field =` on
    /// that section) is authored today by hand-editing the `.lvl` text — see
    /// the module doc on `crate::field` for the full scope note.
    /// `Box`/`Capsule` (`region`/`follow`) standalone
    /// shapes are also not yet authorable here; a hand-edited `[field]`
    /// section using either still round-trips losslessly (a key this editor
    /// carries no row for is preserved verbatim — see `crate::field`).
    Field,
}

impl EntityKind {
    /// Cycle Door → Spawn → Pedestal → Part → Node → Emitter → Trigger → … → Note
    /// → Door (object-selector order).
    pub fn next(self) -> EntityKind {
        match self {
            EntityKind::Door => EntityKind::Spawn,
            EntityKind::Spawn => EntityKind::Pedestal,
            EntityKind::Pedestal => EntityKind::Part,
            EntityKind::Part => EntityKind::Node,
            EntityKind::Node => EntityKind::Emitter,
            EntityKind::Emitter => EntityKind::Trigger,
            EntityKind::Trigger => EntityKind::Pipe,
            EntityKind::Pipe => EntityKind::ZoneInput,
            EntityKind::ZoneInput => EntityKind::Note,
            EntityKind::Note => EntityKind::WallDecal,
            EntityKind::WallDecal => EntityKind::Holo,
            EntityKind::Holo => EntityKind::Portal,
            EntityKind::Portal => EntityKind::Zone,
            EntityKind::Zone => EntityKind::Line,
            EntityKind::Line => EntityKind::Inset,
            EntityKind::Inset => EntityKind::Field,
            EntityKind::Field => EntityKind::Door,
        }
    }

    /// A short panel label.
    pub fn label(self) -> &'static str {
        match self {
            EntityKind::Door => "Door",
            EntityKind::Spawn => "Spawn",
            EntityKind::Pedestal => "Pedestal",
            EntityKind::Part => "Part",
            EntityKind::Node => "Node",
            EntityKind::Emitter => "Emitter",
            EntityKind::Trigger => "Trigger",
            EntityKind::Pipe => "Pipe",
            EntityKind::ZoneInput => "Zone Input",
            EntityKind::Note => "Note",
            EntityKind::WallDecal => "Wall Decal",
            EntityKind::Holo => "Hologram",
            EntityKind::Portal => "Portal",
            EntityKind::Zone => "Zone",
            EntityKind::Line => "Line",
            EntityKind::Inset => "Inset",
            EntityKind::Field => "Field",
        }
    }
}

/// Read a readable label for an [`re_level::OverlayScaleRule`] — `none` / `crisp` /
/// `self_similar`, matching the serializer's keyword.
pub fn scale_rule_label(rule: &re_level::OverlayScaleRule) -> &'static str {
    match rule {
        re_level::OverlayScaleRule::None => "none",
        re_level::OverlayScaleRule::Crisp => "crisp",
        re_level::OverlayScaleRule::SelfSimilar(_) => "self_similar",
    }
}

/// Step an [`re_level::OverlayScaleRule`] to the next case (None → Crisp →
/// SelfSimilar → None), wrapping. Switching INTO `SelfSimilar` seeds it with the
/// inert default params (the dedicated arms/warp/twist fields then edit them);
/// switching AWAY discards the params (re-entering reseeds the default). Switching
/// into SelfSimilar PRESERVES the passed `params` so a round-trip through
/// None/Crisp/SelfSimilar can keep authored params if the caller threads them.
pub fn cycle_scale_rule(
    rule: &re_level::OverlayScaleRule,
    params: re_level::OverlaySelfSimilarParams,
) -> re_level::OverlayScaleRule {
    use re_level::OverlayScaleRule as Sr;
    match rule {
        Sr::None => Sr::Crisp,
        Sr::Crisp => Sr::SelfSimilar(params),
        Sr::SelfSimilar(_) => Sr::None,
    }
}

/// The [`re_world::CellKind`] surfaces a wall decal cycles through (`c` on its
/// `surface` field), paired with the readable label that matches the parser's
/// keyword. This list is the EDITOR half of a two-sided contract:
///
/// * `re_level`'s `[wall_decal] surface` parser accepts exactly six spellings —
///   `matrix`, `trace`, `solid`, `hologram`, `portal`, `object`;
/// * `re_voximport::to_lvl_text` emits all six, and `vox_to_level` mints the last
///   three from `.vox` tags 13/14/15.
///
/// So all six are kinds a level on disk can legitimately carry, and every one of
/// them must be on this cycle: a kind the cycle omits falls through
/// [`cycle_decal_surface`]'s `None` arm and is REWRITTEN to `trace` on the first
/// `c` press — silent destruction of authored data. `decal_surface_tests` pins
/// both directions (every label parses to its paired kind; every writable kind is
/// listed), so the two lists cannot drift apart again.
///
/// Order is cycle order; `trace` is first and is the default a freshly-placed
/// decal carries.
pub const DECAL_SURFACES: [(re_world::CellKind, &str); 6] = [
    (re_world::CellKind::WallTrace, "trace"),
    (re_world::CellKind::WallMatrix, "matrix"),
    (re_world::CellKind::WallSolid, "solid"),
    (re_world::CellKind::Hologram, "hologram"),
    (re_world::CellKind::Portal, "portal"),
    (re_world::CellKind::Object, "object"),
];

/// Step a wall-decal `surface_kind` to the next/prev entry in [`DECAL_SURFACES`],
/// wrapping. Every kind the `.lvl` format can carry on a decal is on that list, so
/// the `None` arm is reached only by a `CellKind` that is not a decal surface at
/// all (a corrupt or hand-forged in-memory value); it lands on `trace`, the
/// placement default.
pub fn cycle_decal_surface(current: re_world::CellKind, delta: i32) -> re_world::CellKind {
    let n = DECAL_SURFACES.len() as i32;
    match DECAL_SURFACES.iter().position(|&(k, _)| k == current) {
        Some(i) => DECAL_SURFACES[(i as i32 + delta).rem_euclid(n) as usize].0,
        None => DECAL_SURFACES[0].0,
    }
}

/// A readable label for a wall-decal `surface_kind` — the parser keyword for any
/// of the six surfaces a decal can carry, else the lowercase debug name for a
/// `CellKind` that is not a decal surface at all.
pub fn decal_surface_label(kind: re_world::CellKind) -> String {
    DECAL_SURFACES
        .iter()
        .find(|&&(k, _)| k == kind)
        .map(|&(_, l)| l.to_string())
        .unwrap_or_else(|| format!("{kind:?}").to_lowercase())
}

/// The three [`re_world::HoloShape`] interior fills a hologram object cycles
/// through (`c` on its `shape` field), paired with the readable label that matches
/// the parser's keyword (`re_world::HoloShape::from_str_opt` reads
/// `box`/`pyramid`/`sphere`). `box` is the default a freshly-placed hologram carries.
pub const HOLO_SHAPES: [(re_world::HoloShape, &str); 3] = [
    (re_world::HoloShape::Box, "box"),
    (re_world::HoloShape::Pyramid, "pyramid"),
    (re_world::HoloShape::Sphere, "sphere"),
];

/// Step a hologram `shape` to the next/prev entry in [`HOLO_SHAPES`], wrapping. A
/// shape not on the list snaps to the first entry (`box`).
pub fn cycle_holo_shape(current: re_world::HoloShape, delta: i32) -> re_world::HoloShape {
    let n = HOLO_SHAPES.len() as i32;
    match HOLO_SHAPES.iter().position(|&(k, _)| k == current) {
        Some(i) => HOLO_SHAPES[(i as i32 + delta).rem_euclid(n) as usize].0,
        None => HOLO_SHAPES[0].0,
    }
}

/// A readable label for a hologram `shape` (`box`/`pyramid`/`sphere`, else the
/// lowercase debug name).
pub fn holo_shape_label(shape: re_world::HoloShape) -> String {
    HOLO_SHAPES
        .iter()
        .find(|&&(k, _)| k == shape)
        .map(|&(_, l)| l.to_string())
        .unwrap_or_else(|| format!("{shape:?}").to_lowercase())
}

/// The preset event labels the `Event` action's label cycles through (`c` on the
/// target field of an Event trigger). Free text entry is impractical in this TUI,
/// so the author picks from this small vocabulary; each is a valid single token so
/// it round-trips through `action = event LABEL`.
pub const TRIGGER_EVENT_LABELS: [&str; 6] =
    ["event", "enter", "exit", "reveal", "alarm", "checkpoint"];

/// [`TRIGGER_EVENT_LABELS`] as owned strings, for [`crate::vocab`].
#[must_use]
pub fn event_label_canon() -> Vec<String> {
    TRIGGER_EVENT_LABELS
        .iter()
        .map(|l| (*l).to_string())
        .collect()
}

/// Step an event `label` `delta` places along [`event_label_canon`] plus the
/// labels this session has seen, wrapping.
///
/// `re_level` accepts FREE TEXT here (`action = event WHATEVER`), so the preset
/// list can never be complete. It used to snap anything off-list to `event`,
/// which would have deleted an imported label outright; now the label is
/// remembered and stays in the cycle.
pub fn cycle_event_label(state: &mut EditorState, current: &str, delta: i32) -> String {
    let canon = event_label_canon();
    let cur = current.to_string();
    crate::vocab::remember(&canon, &mut state.vocab.event_label, &cur);
    crate::vocab::step(&canon, &state.vocab.event_label, &cur, delta)
}

/// One named GROUP the emitter focused-field cycle is paginated into. The panel
/// shows the focused field's group label so a long flat list stays navigable —
/// the user reads "\[Rates\] rate" / "\[Forces\] accelZ" etc. while `;`/`'` step
/// through every field across every group in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitterGroup {
    /// High-level render behaviour: kind.
    Kind,
    /// Spawn frequency: rate, burst, max_particles.
    Rates,
    /// Velocity: speed, speed_spread.
    Motion,
    /// Direction + cone spread.
    Aim,
    /// Lifetime + its spread.
    Life,
    /// Size + growth.
    Shape,
    /// Per-step forces: accel x/y/z, swirl, path_jitter.
    Forces,
    /// Colour stops + colour mode.
    Colour,
    /// Glyph family / glyph_set / bold.
    Glyphs,
    /// Determinism: seed, blend, origin nudges.
    Misc,
}

impl EmitterGroup {
    /// Short bracketed panel label for the group banner.
    pub fn label(self) -> &'static str {
        match self {
            EmitterGroup::Kind => "Kind",
            EmitterGroup::Rates => "Rates",
            EmitterGroup::Motion => "Motion",
            EmitterGroup::Aim => "Aim",
            EmitterGroup::Life => "Life",
            EmitterGroup::Shape => "Shape",
            EmitterGroup::Forces => "Forces",
            EmitterGroup::Colour => "Colour",
            EmitterGroup::Glyphs => "Glyphs",
            EmitterGroup::Misc => "Misc",
        }
    }
}

/// Which parameter of the selected emitter the focused-field editor targets.
/// The `;` key cycles forward and `'` cycles backward through these; `+`/`-`
/// adjust the focused NUMERIC param, and `c` cycles the focused ENUM param.
///
/// EXHAUSTIVE: every field of [`re_world::EmitterSpec`] is reachable here — the
/// numeric scalars, the three colour stops, all enum/axis fields, the glyph_set
/// toggle, and the origin fine-tune nudges. Grouped (see [`EmitterGroup`]) so the
/// long list stays navigable.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EmitterField {
    // ── Kind ──
    /// high-level render behaviour — `c` cycles Standard/GodRays (enum).
    Kind,
    // ── Rates ──
    /// particles/sec — `+`/`-` step (numeric).
    Rate,
    /// one-shot burst count (`None`/`Some(n)`) — `+`/`-` step; -1 below 0 ⇒ None.
    Burst,
    /// hard particle cap — `+`/`-` step, min 1 (numeric).
    MaxParticles,
    // ── Motion ──
    /// initial speed — `+`/`-` step (numeric).
    Speed,
    /// symmetric speed spread — `+`/`-` step, min 0 (numeric).
    SpeedSpread,
    // ── Aim ──
    /// which world axis the emission points along — `c` cycles +Z/-Z/+X/-X/+Y/-Y.
    DirectionAxis,
    /// cone half-angle degrees — `+`/`-` step, clamped 0..=180 (numeric).
    SpreadDeg,
    // ── Life ──
    /// mean lifetime seconds — `+`/`-` step (numeric).
    Lifetime,
    /// symmetric lifetime spread — `+`/`-` step, min 0 (numeric).
    LifetimeSpread,
    // ── Shape ──
    /// birth size — `+`/`-` step (numeric).
    Size,
    /// size-over-age scale — `+`/`-` step (numeric).
    Growth,
    // ── Forces ──
    /// acceleration X (cells/s²) — `+`/`-` step (numeric).
    AccelX,
    /// acceleration Y — `+`/`-` step (numeric).
    AccelY,
    /// acceleration Z (gravity/buoyancy) — `+`/`-` step (numeric).
    AccelZ,
    /// curl/rotational swirl strength — `+`/`-` step (numeric, signed).
    Swirl,
    /// per-step turbulence amplitude — `+`/`-` step, min 0 (numeric).
    PathJitter,
    // ── Colour ──
    /// colour stop 0 (birth) — `+`/`-` step a palette (colour).
    Color0,
    /// colour stop 1 (mid-life) — `+`/`-` step a palette (colour).
    Color1,
    /// colour stop 2 (death) — `+`/`-` step a palette (colour).
    Color2,
    /// colour application — `c` cycles RampOverAge/RandomPick/Fixed.
    ColorMode,
    // ── Glyphs ──
    /// glyph family name — `c` cycles the canonical family list.
    Family,
    /// explicit glyph-set override (`None`/`Some(set)`) — `c` cycles preset sets.
    GlyphSet,
    /// bold policy — `c` cycles Off/On/ByProximity/ByAge.
    Bold,
    // ── Misc ──
    /// composite blend — `c` cycles KeepBg/Opaque.
    Blend,
    /// deterministic RNG seed — `+`/`-` step (numeric).
    Seed,
    /// origin X fine-tune (cells) — `+`/`-` nudge by 0.1 (numeric).
    OriginX,
    /// origin Y fine-tune — `+`/`-` nudge by 0.1 (numeric).
    OriginY,
    /// origin Z fine-tune — `+`/`-` nudge by 0.1 (numeric).
    OriginZ,
}

/// The focused-field cycle order. `;`/`'` step through it (wrapping). Ordered by
/// [`EmitterGroup`] so paging reads coherently.
pub const EMITTER_FIELDS: [EmitterField; 29] = [
    // Kind
    EmitterField::Kind,
    // Rates
    EmitterField::Rate,
    EmitterField::Burst,
    EmitterField::MaxParticles,
    // Motion
    EmitterField::Speed,
    EmitterField::SpeedSpread,
    // Aim
    EmitterField::DirectionAxis,
    EmitterField::SpreadDeg,
    // Life
    EmitterField::Lifetime,
    EmitterField::LifetimeSpread,
    // Shape
    EmitterField::Size,
    EmitterField::Growth,
    // Forces
    EmitterField::AccelX,
    EmitterField::AccelY,
    EmitterField::AccelZ,
    EmitterField::Swirl,
    EmitterField::PathJitter,
    // Colour
    EmitterField::Color0,
    EmitterField::Color1,
    EmitterField::Color2,
    EmitterField::ColorMode,
    // Glyphs
    EmitterField::Family,
    EmitterField::GlyphSet,
    EmitterField::Bold,
    // Misc
    EmitterField::Blend,
    EmitterField::Seed,
    EmitterField::OriginX,
    EmitterField::OriginY,
    EmitterField::OriginZ,
];

impl EmitterField {
    /// Step to the next/prev field in [`EMITTER_FIELDS`], wrapping.
    pub fn step(self, delta: i32) -> EmitterField {
        let n = EMITTER_FIELDS.len() as i32;
        let i = EMITTER_FIELDS.iter().position(|&f| f == self).unwrap_or(0) as i32;
        EMITTER_FIELDS[(i + delta).rem_euclid(n) as usize]
    }

    /// Short panel label for the focused field.
    pub fn label(self) -> &'static str {
        match self {
            EmitterField::Kind => "kind",
            EmitterField::Rate => "rate",
            EmitterField::Burst => "burst",
            EmitterField::MaxParticles => "max_particles",
            EmitterField::Speed => "speed",
            EmitterField::SpeedSpread => "speed_spread",
            EmitterField::DirectionAxis => "dir",
            EmitterField::SpreadDeg => "spread",
            EmitterField::Lifetime => "life",
            EmitterField::LifetimeSpread => "life_spread",
            EmitterField::Size => "size",
            EmitterField::Growth => "growth",
            EmitterField::AccelX => "accelX",
            EmitterField::AccelY => "accelY",
            EmitterField::AccelZ => "accelZ",
            EmitterField::Swirl => "swirl",
            EmitterField::PathJitter => "path_jitter",
            EmitterField::Color0 => "color0",
            EmitterField::Color1 => "color1",
            EmitterField::Color2 => "color2",
            EmitterField::ColorMode => "color_mode",
            EmitterField::Family => "family",
            EmitterField::GlyphSet => "glyph_set",
            EmitterField::Bold => "bold",
            EmitterField::Blend => "blend",
            EmitterField::Seed => "seed",
            EmitterField::OriginX => "originX",
            EmitterField::OriginY => "originY",
            EmitterField::OriginZ => "originZ",
        }
    }

    /// Which [`EmitterGroup`] this field belongs to (for the panel banner).
    pub fn group(self) -> EmitterGroup {
        match self {
            EmitterField::Kind => EmitterGroup::Kind,
            EmitterField::Rate | EmitterField::Burst | EmitterField::MaxParticles => {
                EmitterGroup::Rates
            }
            EmitterField::Speed | EmitterField::SpeedSpread => EmitterGroup::Motion,
            EmitterField::DirectionAxis | EmitterField::SpreadDeg => EmitterGroup::Aim,
            EmitterField::Lifetime | EmitterField::LifetimeSpread => EmitterGroup::Life,
            EmitterField::Size | EmitterField::Growth => EmitterGroup::Shape,
            EmitterField::AccelX
            | EmitterField::AccelY
            | EmitterField::AccelZ
            | EmitterField::Swirl
            | EmitterField::PathJitter => EmitterGroup::Forces,
            EmitterField::Color0
            | EmitterField::Color1
            | EmitterField::Color2
            | EmitterField::ColorMode => EmitterGroup::Colour,
            EmitterField::Family | EmitterField::GlyphSet | EmitterField::Bold => {
                EmitterGroup::Glyphs
            }
            EmitterField::Blend
            | EmitterField::Seed
            | EmitterField::OriginX
            | EmitterField::OriginY
            | EmitterField::OriginZ => EmitterGroup::Misc,
        }
    }

    /// True when `+`/`-` adjust this field (numeric/colour); false when `c` does
    /// (the enum/cycle fields). Colour stops are adjusted with `+`/`-` (palette
    /// step), so they count as numeric here.
    pub fn is_numeric(self) -> bool {
        !matches!(
            self,
            EmitterField::Kind
                | EmitterField::DirectionAxis
                | EmitterField::ColorMode
                | EmitterField::Family
                | EmitterField::GlyphSet
                | EmitterField::Bold
                | EmitterField::Blend
        )
    }
}

// The glyph-family vocabulary used to live here as a hand-copied `[&str; 13]`.
// It is now `crate::vocab::emitter_families()`, enumerated from
// `re_render::glyph_family::LIBRARY` — the copy had drifted three families behind
// the library, so `state_node` / `state_door` / `state_trace` were unauthorable
// and any level using one would have been snapped to `haze_o`.

/// The four cardinal player-spawn facings, in cycle order, as legacy `yaw_col`
/// values (`+y=0, +x=120, -y=240, -x=360` — matching `re_level`'s
/// `facing_to_yaw_col`). The `f` key steps the spawn facing through these.
pub const SPAWN_FACINGS: [(i32, &str); 4] = [(0, "+y"), (120, "+x"), (240, "-y"), (360, "-x")];

/// The four cardinal `yaw_col` values of [`SPAWN_FACINGS`], for [`crate::vocab`].
#[must_use]
pub fn facing_canon() -> Vec<i32> {
    SPAWN_FACINGS.iter().map(|&(v, _)| v).collect()
}

/// The list a spawn-facing `c`/`f` press walks: the four cardinals plus every
/// non-cardinal yaw this session has seen, plus `current`.
#[must_use]
pub fn facing_options(state: &EditorState, current: i32) -> Vec<i32> {
    crate::vocab::options_at(&facing_canon(), &state.vocab.facing, &current).0
}

/// Step a `yaw_col` one place along [`facing_options`], wrapping.
///
/// `re_level::facing_to_yaw_col` falls through to `other.parse()`, so a level may
/// author ANY integer yaw. It used to snap a non-cardinal one to `+y`; now the
/// yaw is remembered and stays in the cycle.
pub fn cycle_facing(state: &mut EditorState, yaw_col: i32) -> i32 {
    let canon = facing_canon();
    crate::vocab::remember(&canon, &mut state.vocab.facing, &yaw_col);
    crate::vocab::step(&canon, &state.vocab.facing, &yaw_col, 1)
}

/// A readable label for a spawn `yaw_col` facing (`+y`/`+x`/`-y`/`-x`, else the
/// raw number).
pub fn facing_label(yaw_col: i32) -> String {
    SPAWN_FACINGS
        .iter()
        .find(|&&(v, _)| v == yaw_col)
        .map(|&(_, l)| l.to_string())
        .unwrap_or_else(|| yaw_col.to_string())
}

/// The [`re_level::NoteKind`] variants in cycle order, paired with a
/// readable label. The `c` key on a Note's kind field steps through these. Kept
/// here next to the spawn-facing helpers so the editor owns its enum vocabularies
/// in one place; the on-disk keyword map (`researcher`/`system_log`/
/// `model_thought`/`stencil`) lives in `re_voximport` and is NOT duplicated.
pub const NOTE_KINDS: [(re_level::NoteKind, &str); 4] = [
    (re_level::NoteKind::Researcher, "Researcher"),
    (re_level::NoteKind::SystemLog, "System Log"),
    (re_level::NoteKind::ModelThought, "Model Thought"),
    (re_level::NoteKind::Stencil, "Stencil"),
];

/// Step a [`re_level::NoteKind`] to the next/prev entry in [`NOTE_KINDS`],
/// wrapping. An unknown variant snaps to the first entry (`Researcher`).
pub fn cycle_note_kind(current: re_level::NoteKind, delta: i32) -> re_level::NoteKind {
    let n = NOTE_KINDS.len() as i32;
    match NOTE_KINDS.iter().position(|&(k, _)| k == current) {
        Some(i) => NOTE_KINDS[(i as i32 + delta).rem_euclid(n) as usize].0,
        None => NOTE_KINDS[0].0,
    }
}

/// A readable label for a [`re_level::NoteKind`].
pub fn note_kind_label(kind: re_level::NoteKind) -> &'static str {
    NOTE_KINDS
        .iter()
        .find(|&&(k, _)| k == kind)
        .map(|&(_, l)| l)
        .unwrap_or("Researcher")
}

/// Which FOCUSED FIELD of the active non-emitter entity (Door / Node / Pedestal /
/// Spawn) the `;`/`'` keys page through and the `c` / `+`/`-` keys edit. This is
/// the SAME focused-field-editor pattern the emitter uses ([`EmitterField`]),
/// extended so every authored field of a door / node / pedestal / spawn is
/// reachable from the keyboard (the "exhaustive object settings" the design asks
/// for). Each variant belongs to exactly one [`EntityKind`]; [`entity_fields_for`]
/// returns the ordered field list for a kind and [`step_entity_field`] pages
/// within it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EntityField {
    // ── Door (re_level::Door) ──
    /// Door anchor X — `+`/`-` step, clamped to the room (numeric).
    DoorPosX,
    /// Door anchor Y — `+`/`-` step (numeric).
    DoorPosY,
    /// Door anchor Z — `+`/`-` step (numeric).
    DoorPosZ,
    /// The node the target-CURSOR points at — `+`/`-` move it through the level's
    /// nodes (selection); `c` toggles that node in `target_parts` (must-FIRE).
    DoorTargetOn,
    /// The node the target-CURSOR points at — `+`/`-` move it; `c` toggles that
    /// node in `target_off` (must-be-OFF, the L12 `exit 0` inverted win).
    DoorTargetOff,
    // ── Node (re_level::Node) ──
    /// Node X — `+`/`-` step, clamped to the room (numeric).
    NodePosX,
    /// Node Y — `+`/`-` step (numeric).
    NodePosY,
    /// Node Z — `+`/`-` step (numeric).
    NodePosZ,
    /// Top-down glyph — `c` cycles a candidate glyph list.
    NodeGlyph,
    /// `color` RGB — `+`/`-` step the shared palette (colour).
    NodeColor,
    /// io role — `c` cycles neither → input → output (enum).
    NodeIo,
    /// activation threshold — `+`/`-` step (numeric, signed).
    NodeThreshold,
    /// `anchor` pool-word index — `+`/`-` step, min 0 (numeric).
    NodeAnchor,
    /// `pool` word count — `+`/`-` add/remove an auto-named word (selection).
    NodePool,
    /// `pool_alt` word count — `+`/`-` add/remove an auto-named word (selection).
    NodePoolAlt,
    /// The PART the emit-CURSOR points at — `+`/`-` move it through `level.parts`
    /// (selection); `c` toggles a `[emit]` clause linking the selected node to
    /// that part (the node lights it when it fires). Mirrors `DoorTargetOn`.
    NodeEmits,
    // ── Pedestal (re_level::Pedestal) ──
    /// Center X — `+`/`-` step, clamped to the room (numeric).
    PedCenterX,
    /// Center Y — `+`/`-` step (numeric).
    PedCenterY,
    /// Center Z — `+`/`-` step (numeric).
    PedCenterZ,
    /// Volume width — `+`/`-` step, clamped 1..=max (numeric).
    PedVolW,
    /// Volume height — `+`/`-` step (numeric).
    PedVolH,
    /// Volume depth — `+`/`-` step (numeric).
    PedVolD,
    // ── Part (re_level::Part) ──
    /// Volume width — `+`/`-` step, clamped `>= 1`; resizes the mask (numeric).
    PartSizeW,
    /// Volume height — `+`/`-` step, clamped `>= 1`; resizes the mask (numeric).
    PartSizeH,
    /// Volume depth — `+`/`-` step, clamped `>= 1`; resizes the mask (numeric).
    PartSizeD,
    /// World anchor X (`off_x`) — `+`/`-` step, clamped to the room (numeric).
    PartOffX,
    /// World anchor Y (`off_y`) — `+`/`-` step (numeric).
    PartOffY,
    /// World anchor Z (`off_z`) — `+`/`-` step (numeric).
    PartOffZ,
    /// `color` RGB — `+`/`-` step the shared palette (colour).
    PartColor,
    // ── Spawn (re_level::PlayerSpawn) ──
    /// Spawn X — `+`/`-` step, clamped to the room (numeric).
    SpawnPosX,
    /// Spawn Y — `+`/`-` step (numeric).
    SpawnPosY,
    /// Spawn Z — `+`/`-` step (numeric).
    SpawnPosZ,
    /// Facing — `c` cycles the four cardinals (enum).
    SpawnFacing,
    // ── Trigger (re_core::Trigger) ──
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    TrigMinX,
    /// Region min Y — `+`/`-` step (numeric).
    TrigMinY,
    /// Region min Z — `+`/`-` step (numeric).
    TrigMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    TrigMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    TrigMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    TrigMaxZ,
    /// Action TYPE — `c` cycles ActivateNode ↔ Event (enum).
    TrigActionType,
    /// Action target — `c` cycles the node (ActivateNode) or the preset event
    /// label (Event); `+`/`-` also step it (selection).
    TrigTarget,
    /// `once` flag — `c` toggles fire-once vs re-fire (enum).
    TrigOnce,
    // ── Pipe (re_level::Pipe) ──
    /// Authoring mode — `c` toggles `path` ↔ `spline` (enum). In `path` mode the
    /// pipe serializes its expanded waypoints as `path = …` (the default, unchanged
    /// behaviour). In `spline` mode the SAME points are emitted as `spline = …`
    /// smooth Catmull-Rom control points; exactly one of the two is ever written.
    /// The mode is editor-side metadata (the `re_level::Pipe` model has no flag),
    /// stored on [`EditorState::spline_pipes`] keyed by the `(from, to)` node pair.
    PipeMode,
    /// Signal state — `c` toggles Off ↔ On (enum).
    PipeState,
    /// Toggle-switch cell — `c` SETS `switch_at` to the cursor cell when it lies
    /// on the pipe's path, or CLEARS it when the cursor is already the switch cell
    /// / off-path (a toggle). `+`/`-` step the switch ALONG the path (selection).
    PipeSwitch,
    /// Flow density `0..1` — `+`/`-` step (numeric).
    PipeDensity,
    /// `color` RGB — `+`/`-` step the shared palette (colour).
    PipeColor,
    // ── ZoneInput (re_level::ZoneInput) ──
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    ZoneMinX,
    /// Region min Y — `+`/`-` step (numeric).
    ZoneMinY,
    /// Region min Z — `+`/`-` step (numeric).
    ZoneMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    ZoneMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    ZoneMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    ZoneMaxZ,
    /// Linked INPUT node — `c` cycles through (and `+`/`-` step) the level's INPUT
    /// nodes by name; the zone drives that node HIGH while occupied (selection).
    ZoneNode,
    // ── Note (re_level::LevelNote) ──
    /// Note kind — `c` cycles Researcher / SystemLog / ModelThought (enum).
    NoteKindField,
    /// Facing — `c` cycles the four cardinals (enum).
    NoteFacing,
    /// Note X — `+`/`-` step, clamped to the room (numeric).
    NotePosX,
    /// Note Y — `+`/`-` step (numeric).
    NotePosY,
    /// Note Z — `+`/`-` step (numeric).
    NotePosZ,
    /// Body text — typed printable chars append, Backspace pops (a single-line
    /// buffer). Neither numeric nor enum-cyclable; edited by direct key entry while
    /// focused (see `edit.rs`).
    NoteText,
    // ── WallDecal (re_level::WallDecal) ──
    /// Surface texture — `c` cycles trace / matrix / solid (enum).
    DecalSurface,
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    DecalMinX,
    /// Region min Y — `+`/`-` step (numeric).
    DecalMinY,
    /// Region min Z — `+`/`-` step (numeric).
    DecalMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    DecalMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    DecalMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    DecalMaxZ,
    /// Flow `density` `0..1` — `+`/`-` step, clamped `>= 0` (numeric).
    DecalDensity,
    /// `color head` RGB — `+`/`-` step the shared palette (colour).
    DecalColorHead,
    /// `color tail` RGB — `+`/`-` step the shared palette (colour).
    DecalColorTail,
    // ── Holo (re_level::LevelHolo) ──
    /// Interior fill shape — `c` cycles box / pyramid / sphere (enum).
    HoloShapeField,
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    HoloMinX,
    /// Region min Y — `+`/`-` step (numeric).
    HoloMinY,
    /// Region min Z — `+`/`-` step (numeric).
    HoloMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    HoloMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    HoloMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    HoloMaxZ,
    /// Colour/look variant tag `style` — `+`/`-` step the `u16` (numeric, min 0).
    HoloStyle,
    // ── Portal (re_level::LevelPortal) — REGION A ──
    /// Region A min X — `+`/`-` step, clamped to the room, kept ≤ A max (numeric).
    PortalAx0,
    /// Region A min Y — `+`/`-` step (numeric).
    PortalAy0,
    /// Region A min Z — `+`/`-` step (numeric).
    PortalAz0,
    /// Region A max X — `+`/`-` step, clamped to the room, kept ≥ A min (numeric).
    PortalAx1,
    /// Region A max Y — `+`/`-` step (numeric).
    PortalAy1,
    /// Region A max Z — `+`/`-` step (numeric).
    PortalAz1,
    // ── Portal — REGION B ──
    /// Region B min X — `+`/`-` step, clamped to the room, kept ≤ B max (numeric).
    PortalBx0,
    /// Region B min Y — `+`/`-` step (numeric).
    PortalBy0,
    /// Region B min Z — `+`/`-` step (numeric).
    PortalBz0,
    /// Region B max X — `+`/`-` step, clamped to the room, kept ≥ B min (numeric).
    PortalBx1,
    /// Region B max Y — `+`/`-` step (numeric).
    PortalBy1,
    /// Region B max Z — `+`/`-` step (numeric).
    PortalBz1,
    /// Extra yaw applied crossing A→B, in degrees — `+`/`-` step by 15° (numeric).
    PortalRotate,
    /// Optional skin material name — `c` cycles `(none)` + the level's material
    /// names (enum-cyclable).
    PortalSkin,
    // ── Zone (re_core::Zone via level.overlay.zones) ──
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    ZoneRegMinX,
    /// Region min Y — `+`/`-` step (numeric).
    ZoneRegMinY,
    /// Region min Z — `+`/`-` step (numeric).
    ZoneRegMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    ZoneRegMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    ZoneRegMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    ZoneRegMaxZ,
    /// Bound material name — `c` cycles `(none)` + the level's material names
    /// (enum-cyclable; empty ⇒ inherit / no binding).
    ZoneMaterial,
    /// Scale rule — `c` cycles None → Crisp → SelfSimilar (enum-cyclable).
    ZoneScale,
    /// SelfSimilar `arms` (u8) — `+`/`-` step, min 0; only meaningful under
    /// SelfSimilar (else shown as `-`) (numeric).
    ZoneArms,
    /// SelfSimilar `octave_warp` (f32) — `+`/`-` step by 0.05, clamped ≥ 0;
    /// SelfSimilar only (numeric).
    ZoneWarp,
    /// SelfSimilar `twist` (f32) — `+`/`-` step by 0.1; SelfSimilar only (numeric).
    ZoneTwist,
    /// Inset membership — `+`/`-` move a cursor through the level's inset NAMES;
    /// `c` toggles whether the cursor's inset is in this zone's `insets` list
    /// (enum-cyclable, mirrors `DoorTargetOn`).
    ZoneInsets,
    // ── Line (re_core::Line via level.overlay.lines) ──
    /// Bound line-material name — `c` cycles `(none)` + the level's material names
    /// (enum-cyclable; empty ⇒ no binding).
    LineMaterial,
    /// Scale rule — `c` cycles None → Crisp → SelfSimilar (enum-cyclable).
    LineScale,
    /// SelfSimilar `arms` (u8) — `+`/`-` step, min 0; SelfSimilar only (numeric).
    LineArms,
    /// SelfSimilar `octave_warp` (f32) — `+`/`-` step by 0.05, clamped ≥ 0;
    /// SelfSimilar only (numeric).
    LineWarp,
    /// SelfSimilar `twist` (f32) — `+`/`-` step by 0.1; SelfSimilar only (numeric).
    LineTwist,
    /// Waypoint count — `+`/`-` REMOVE the last waypoint / report (a no-op `+`;
    /// waypoints are APPENDED with `space` at the cursor like a pipe path)
    /// (numeric).
    LinePoints,
    // ── Inset (re_core::Inset via level.overlay.insets) ──
    /// Octave range LO (i32) — `+`/`-` step, kept ≤ hi (numeric).
    InsetOctaveLo,
    /// Octave range HI (i32) — `+`/`-` step, kept ≥ lo (numeric).
    InsetOctaveHi,
    /// Region min X — `+`/`-` step, clamped to the room, kept ≤ max (numeric).
    InsetRegMinX,
    /// Region min Y — `+`/`-` step (numeric).
    InsetRegMinY,
    /// Region min Z — `+`/`-` step (numeric).
    InsetRegMinZ,
    /// Region max X — `+`/`-` step, clamped to the room, kept ≥ min (numeric).
    InsetRegMaxX,
    /// Region max Y — `+`/`-` step (numeric).
    InsetRegMaxY,
    /// Region max Z — `+`/`-` step (numeric).
    InsetRegMaxZ,
    /// Child-scene path — typed printable chars append, Backspace pops (a single-
    /// line buffer like [`EntityField::NoteText`]). Neither numeric nor enum.
    InsetChild,
    // ── Field (re_level::fields::FieldDef, standalone only — see `crate::field`) ──
    /// `kind =` — the archetype (enum, 15 canonical names, `c` cycles).
    FieldArchetype,
    /// Shape — `c` cycles Sphere \u{2194} Room (enum). Room has no position; see
    /// [`EntityField::FieldPosX`]/Y/Z.
    FieldShapeKind,
    /// `at` X — `+`/`-` step, clamped to the room (numeric). Meaningless (shown
    /// `-`) while the shape is Room.
    FieldPosX,
    /// `at` Y — `+`/`-` step (numeric). Meaningless while Room.
    FieldPosY,
    /// `at` Z — `+`/`-` step (numeric). Meaningless while Room.
    FieldPosZ,
    /// `radius` — `+`/`-` step by 0.5 (numeric). Reads `room` and is inert while
    /// the shape is Room (radius is implied by the shape then, not an
    /// independent number).
    FieldRadius,
    /// `inner` — `+`/`-` step by 0.5, clamped `>= 0` (numeric).
    FieldInner,
    /// `falloff` — `c` cycles smooth/linear/inverse/hard (enum).
    FieldFalloff,
    /// `occlude` — `c` toggles yes/no (enum).
    FieldOcclude,
    /// `facing` — `c` cycles none/+x/-x/+y/-y/+z/-z (enum). An off-cycle authored
    /// vector (a non-cardinal `facing =`) is preserved as an extra cycle entry
    /// (`crate::field::safe_options`) rather than destroyed.
    FieldFacing,
    /// `cone` half-angle degrees — `+`/`-` step by 5, clamped 1..=180 (numeric).
    /// Meaningless (shown `-`) while `facing` is none.
    FieldCone,
    /// `bind` — `c` cycles none/door/`node:NAME`/`fed:NAME`/`pipe:A->B` by the
    /// level's own node/pipe names (enum; the "binding picker" — see
    /// `crate::field::bind_options`).
    FieldBind,
    /// `invert` — `c` toggles yes/no (enum). Only meaningful with a non-`none`
    /// `bind`, but always authorable/round-tripped.
    FieldInvert,
    /// `drive` — `c` cycles const/door_ready/solved/cycle_held/budget/seq/
    /// clarity/shards/`lit:NAME`/`fed:NAME`/`starve:NAME`/`live:A->B`/
    /// `dead_draw:A->B` (enum; see `crate::field::drive_options`).
    FieldDrive,
    /// `base` (the doc's "intensity" — the fixed/floor value while `drive =
    /// const`, or the resting level some archetypes add on top of their live
    /// drive) — `+`/`-` step by 0.05, clamped 0..=1 (numeric).
    FieldBase,
    /// `depth` (how strongly the recipe's DSP moves at full weight) — `+`/`-`
    /// step by 0.05, clamped 0..=1 (numeric).
    FieldDepth,
    /// `priority` — `+`/`-` step, clamped 0..=9 (numeric).
    FieldPriority,
}

impl EntityField {
    /// Short panel label for the focused field.
    pub fn label(self) -> &'static str {
        match self {
            EntityField::DoorPosX => "x",
            EntityField::DoorPosY => "y",
            EntityField::DoorPosZ => "z",
            EntityField::DoorTargetOn => "target(on)",
            EntityField::DoorTargetOff => "target(off)",
            EntityField::NodePosX => "x",
            EntityField::NodePosY => "y",
            EntityField::NodePosZ => "z",
            EntityField::NodeGlyph => "glyph",
            EntityField::NodeColor => "color",
            EntityField::NodeIo => "io",
            EntityField::NodeThreshold => "threshold",
            EntityField::NodeAnchor => "anchor",
            EntityField::NodePool => "pool",
            EntityField::NodePoolAlt => "pool_alt",
            EntityField::NodeEmits => "emits",
            EntityField::PedCenterX => "cx",
            EntityField::PedCenterY => "cy",
            EntityField::PedCenterZ => "cz",
            EntityField::PedVolW => "w",
            EntityField::PedVolH => "h",
            EntityField::PedVolD => "d",
            EntityField::PartSizeW => "w",
            EntityField::PartSizeH => "h",
            EntityField::PartSizeD => "d",
            EntityField::PartOffX => "offX",
            EntityField::PartOffY => "offY",
            EntityField::PartOffZ => "offZ",
            EntityField::PartColor => "color",
            EntityField::SpawnPosX => "x",
            EntityField::SpawnPosY => "y",
            EntityField::SpawnPosZ => "z",
            EntityField::SpawnFacing => "facing",
            EntityField::TrigMinX => "minX",
            EntityField::TrigMinY => "minY",
            EntityField::TrigMinZ => "minZ",
            EntityField::TrigMaxX => "maxX",
            EntityField::TrigMaxY => "maxY",
            EntityField::TrigMaxZ => "maxZ",
            EntityField::TrigActionType => "action",
            EntityField::TrigTarget => "target",
            EntityField::TrigOnce => "once",
            EntityField::PipeMode => "mode",
            EntityField::PipeState => "state",
            EntityField::PipeSwitch => "switch_at",
            EntityField::PipeDensity => "density",
            EntityField::PipeColor => "color",
            EntityField::ZoneMinX => "minX",
            EntityField::ZoneMinY => "minY",
            EntityField::ZoneMinZ => "minZ",
            EntityField::ZoneMaxX => "maxX",
            EntityField::ZoneMaxY => "maxY",
            EntityField::ZoneMaxZ => "maxZ",
            EntityField::ZoneNode => "input node",
            EntityField::NoteKindField => "kind",
            EntityField::NoteFacing => "facing",
            EntityField::NotePosX => "x",
            EntityField::NotePosY => "y",
            EntityField::NotePosZ => "z",
            EntityField::NoteText => "text",
            EntityField::DecalSurface => "surface",
            EntityField::DecalMinX => "minX",
            EntityField::DecalMinY => "minY",
            EntityField::DecalMinZ => "minZ",
            EntityField::DecalMaxX => "maxX",
            EntityField::DecalMaxY => "maxY",
            EntityField::DecalMaxZ => "maxZ",
            EntityField::DecalDensity => "density",
            EntityField::DecalColorHead => "color head",
            EntityField::DecalColorTail => "color tail",
            EntityField::HoloShapeField => "shape",
            EntityField::HoloMinX => "minX",
            EntityField::HoloMinY => "minY",
            EntityField::HoloMinZ => "minZ",
            EntityField::HoloMaxX => "maxX",
            EntityField::HoloMaxY => "maxY",
            EntityField::HoloMaxZ => "maxZ",
            EntityField::HoloStyle => "style",
            EntityField::PortalAx0 => "A minX",
            EntityField::PortalAy0 => "A minY",
            EntityField::PortalAz0 => "A minZ",
            EntityField::PortalAx1 => "A maxX",
            EntityField::PortalAy1 => "A maxY",
            EntityField::PortalAz1 => "A maxZ",
            EntityField::PortalBx0 => "B minX",
            EntityField::PortalBy0 => "B minY",
            EntityField::PortalBz0 => "B minZ",
            EntityField::PortalBx1 => "B maxX",
            EntityField::PortalBy1 => "B maxY",
            EntityField::PortalBz1 => "B maxZ",
            EntityField::PortalRotate => "rotate",
            EntityField::PortalSkin => "skin",
            EntityField::ZoneRegMinX => "minX",
            EntityField::ZoneRegMinY => "minY",
            EntityField::ZoneRegMinZ => "minZ",
            EntityField::ZoneRegMaxX => "maxX",
            EntityField::ZoneRegMaxY => "maxY",
            EntityField::ZoneRegMaxZ => "maxZ",
            EntityField::ZoneMaterial => "material",
            EntityField::ZoneScale => "scale",
            EntityField::ZoneArms => "arms",
            EntityField::ZoneWarp => "octave_warp",
            EntityField::ZoneTwist => "twist",
            EntityField::ZoneInsets => "insets",
            EntityField::LineMaterial => "material",
            EntityField::LineScale => "scale",
            EntityField::LineArms => "arms",
            EntityField::LineWarp => "octave_warp",
            EntityField::LineTwist => "twist",
            EntityField::LinePoints => "points",
            EntityField::InsetOctaveLo => "octave lo",
            EntityField::InsetOctaveHi => "octave hi",
            EntityField::InsetRegMinX => "minX",
            EntityField::InsetRegMinY => "minY",
            EntityField::InsetRegMinZ => "minZ",
            EntityField::InsetRegMaxX => "maxX",
            EntityField::InsetRegMaxY => "maxY",
            EntityField::InsetRegMaxZ => "maxZ",
            EntityField::InsetChild => "child",
            EntityField::FieldArchetype => "kind",
            EntityField::FieldShapeKind => "shape",
            EntityField::FieldPosX => "x",
            EntityField::FieldPosY => "y",
            EntityField::FieldPosZ => "z",
            EntityField::FieldRadius => "radius",
            EntityField::FieldInner => "inner",
            EntityField::FieldFalloff => "falloff",
            EntityField::FieldOcclude => "occlude",
            EntityField::FieldFacing => "facing",
            EntityField::FieldCone => "cone",
            EntityField::FieldBind => "bind",
            EntityField::FieldInvert => "invert",
            EntityField::FieldDrive => "drive",
            EntityField::FieldBase => "intensity",
            EntityField::FieldDepth => "depth",
            EntityField::FieldPriority => "priority",
        }
    }

    /// True when `+`/`-` adjust this field (numeric/colour/selection-step); false
    /// when `c` does (the pure enum/cycle fields: glyph, io, facing, action type,
    /// once). Door-target fields are adjusted with BOTH (`+`/`-` move the node
    /// cursor, `c` toggles membership), so they report numeric here; the trigger
    /// target also responds to BOTH (`+`/`-` and `c` step the node/label). Used by
    /// the help/route logic to hint the right key for the focused field.
    pub fn is_numeric(self) -> bool {
        !matches!(
            self,
            EntityField::NodeGlyph
                | EntityField::NodeIo
                | EntityField::SpawnFacing
                | EntityField::TrigActionType
                | EntityField::TrigOnce
                | EntityField::PipeMode
                | EntityField::PipeState
                | EntityField::NoteKindField
                | EntityField::NoteFacing
                | EntityField::NoteText
                | EntityField::DecalSurface
                | EntityField::HoloShapeField
                | EntityField::PortalSkin
                | EntityField::ZoneMaterial
                | EntityField::ZoneScale
                | EntityField::LineMaterial
                | EntityField::LineScale
                | EntityField::InsetChild
                | EntityField::FieldArchetype
                | EntityField::FieldShapeKind
                | EntityField::FieldFalloff
                | EntityField::FieldOcclude
                | EntityField::FieldFacing
                | EntityField::FieldBind
                | EntityField::FieldInvert
                | EntityField::FieldDrive
        )
    }

    /// True when `c` cycles/toggles this field (the enum fields plus the two
    /// door-target toggles and the trigger target, which respond to BOTH `c` and
    /// `+`/`-`).
    pub fn is_enum_cyclable(self) -> bool {
        matches!(
            self,
            EntityField::NodeGlyph
                | EntityField::NodeIo
                | EntityField::SpawnFacing
                | EntityField::DoorTargetOn
                | EntityField::DoorTargetOff
                | EntityField::NodeEmits
                | EntityField::TrigActionType
                | EntityField::TrigTarget
                | EntityField::TrigOnce
                | EntityField::PipeMode
                | EntityField::PipeState
                | EntityField::PipeSwitch
                | EntityField::ZoneNode
                | EntityField::NoteKindField
                | EntityField::NoteFacing
                | EntityField::DecalSurface
                | EntityField::HoloShapeField
                | EntityField::PortalSkin
                | EntityField::ZoneMaterial
                | EntityField::ZoneScale
                | EntityField::ZoneInsets
                | EntityField::LineMaterial
                | EntityField::LineScale
                | EntityField::FieldArchetype
                | EntityField::FieldShapeKind
                | EntityField::FieldFalloff
                | EntityField::FieldOcclude
                | EntityField::FieldFacing
                | EntityField::FieldBind
                | EntityField::FieldInvert
                | EntityField::FieldDrive
        )
    }
}

/// The ordered focused-field list for a non-emitter [`EntityKind`]. `;`/`'` page
/// through exactly these (wrapping). Emitter has its own [`EMITTER_FIELDS`] list.
pub fn entity_fields_for(kind: EntityKind) -> &'static [EntityField] {
    match kind {
        EntityKind::Door => &[
            EntityField::DoorPosX,
            EntityField::DoorPosY,
            EntityField::DoorPosZ,
            EntityField::DoorTargetOn,
            EntityField::DoorTargetOff,
        ],
        EntityKind::Node => &[
            EntityField::NodePosX,
            EntityField::NodePosY,
            EntityField::NodePosZ,
            EntityField::NodeGlyph,
            EntityField::NodeColor,
            EntityField::NodeIo,
            EntityField::NodeThreshold,
            EntityField::NodeAnchor,
            EntityField::NodePool,
            EntityField::NodePoolAlt,
            EntityField::NodeEmits,
        ],
        EntityKind::Pedestal => &[
            EntityField::PedCenterX,
            EntityField::PedCenterY,
            EntityField::PedCenterZ,
            EntityField::PedVolW,
            EntityField::PedVolH,
            EntityField::PedVolD,
        ],
        EntityKind::Part => &[
            EntityField::PartSizeW,
            EntityField::PartSizeH,
            EntityField::PartSizeD,
            EntityField::PartOffX,
            EntityField::PartOffY,
            EntityField::PartOffZ,
            EntityField::PartColor,
        ],
        EntityKind::Spawn => &[
            EntityField::SpawnPosX,
            EntityField::SpawnPosY,
            EntityField::SpawnPosZ,
            EntityField::SpawnFacing,
        ],
        EntityKind::Trigger => &[
            EntityField::TrigMinX,
            EntityField::TrigMinY,
            EntityField::TrigMinZ,
            EntityField::TrigMaxX,
            EntityField::TrigMaxY,
            EntityField::TrigMaxZ,
            EntityField::TrigActionType,
            EntityField::TrigTarget,
            EntityField::TrigOnce,
        ],
        EntityKind::Pipe => &[
            EntityField::PipeMode,
            EntityField::PipeState,
            EntityField::PipeSwitch,
            EntityField::PipeDensity,
            EntityField::PipeColor,
        ],
        EntityKind::ZoneInput => &[
            EntityField::ZoneMinX,
            EntityField::ZoneMinY,
            EntityField::ZoneMinZ,
            EntityField::ZoneMaxX,
            EntityField::ZoneMaxY,
            EntityField::ZoneMaxZ,
            EntityField::ZoneNode,
        ],
        EntityKind::Note => &[
            EntityField::NoteKindField,
            EntityField::NoteFacing,
            EntityField::NotePosX,
            EntityField::NotePosY,
            EntityField::NotePosZ,
            EntityField::NoteText,
        ],
        EntityKind::WallDecal => &[
            EntityField::DecalSurface,
            EntityField::DecalMinX,
            EntityField::DecalMinY,
            EntityField::DecalMinZ,
            EntityField::DecalMaxX,
            EntityField::DecalMaxY,
            EntityField::DecalMaxZ,
            EntityField::DecalDensity,
            EntityField::DecalColorHead,
            EntityField::DecalColorTail,
        ],
        EntityKind::Holo => &[
            EntityField::HoloShapeField,
            EntityField::HoloMinX,
            EntityField::HoloMinY,
            EntityField::HoloMinZ,
            EntityField::HoloMaxX,
            EntityField::HoloMaxY,
            EntityField::HoloMaxZ,
            EntityField::HoloStyle,
        ],
        // Portal: Region A (6 coords), then Region B (6 coords), then rotate + skin.
        EntityKind::Portal => &[
            EntityField::PortalAx0,
            EntityField::PortalAy0,
            EntityField::PortalAz0,
            EntityField::PortalAx1,
            EntityField::PortalAy1,
            EntityField::PortalAz1,
            EntityField::PortalBx0,
            EntityField::PortalBy0,
            EntityField::PortalBz0,
            EntityField::PortalBx1,
            EntityField::PortalBy1,
            EntityField::PortalBz1,
            EntityField::PortalRotate,
            EntityField::PortalSkin,
        ],
        // Zone: region (6), material, scale, the three SelfSimilar sub-fields, insets.
        EntityKind::Zone => &[
            EntityField::ZoneRegMinX,
            EntityField::ZoneRegMinY,
            EntityField::ZoneRegMinZ,
            EntityField::ZoneRegMaxX,
            EntityField::ZoneRegMaxY,
            EntityField::ZoneRegMaxZ,
            EntityField::ZoneMaterial,
            EntityField::ZoneScale,
            EntityField::ZoneArms,
            EntityField::ZoneWarp,
            EntityField::ZoneTwist,
            EntityField::ZoneInsets,
        ],
        // Line: material, scale, the three SelfSimilar sub-fields, the waypoint count.
        EntityKind::Line => &[
            EntityField::LineMaterial,
            EntityField::LineScale,
            EntityField::LineArms,
            EntityField::LineWarp,
            EntityField::LineTwist,
            EntityField::LinePoints,
        ],
        // Inset: octave lo/hi, region (6), child scene path.
        EntityKind::Inset => &[
            EntityField::InsetOctaveLo,
            EntityField::InsetOctaveHi,
            EntityField::InsetRegMinX,
            EntityField::InsetRegMinY,
            EntityField::InsetRegMinZ,
            EntityField::InsetRegMaxX,
            EntityField::InsetRegMaxY,
            EntityField::InsetRegMaxZ,
            EntityField::InsetChild,
        ],
        // Emitter uses the dedicated EMITTER_FIELDS focus list, not this one.
        EntityKind::Emitter => &[],
        // Field: archetype, shape (+ position while Sphere), radius/inner/
        // falloff/occlude, facing+cone, bind, invert, drive, intensity(base)/
        // depth, priority. See `crate::field`'s module doc for the full scope
        // note (v1 authors Sphere/Room only; Box/Capsule need a hand-edited
        // `[field]` section, still lossless to round-trip).
        EntityKind::Field => &[
            EntityField::FieldArchetype,
            EntityField::FieldShapeKind,
            EntityField::FieldPosX,
            EntityField::FieldPosY,
            EntityField::FieldPosZ,
            EntityField::FieldRadius,
            EntityField::FieldInner,
            EntityField::FieldFalloff,
            EntityField::FieldOcclude,
            EntityField::FieldFacing,
            EntityField::FieldCone,
            EntityField::FieldBind,
            EntityField::FieldInvert,
            EntityField::FieldDrive,
            EntityField::FieldBase,
            EntityField::FieldDepth,
            EntityField::FieldPriority,
        ],
    }
}

/// The first focused field of a kind (what focus resets to when the selector
/// switches kind). Defaults to a harmless position field; Emitter has none here.
pub fn entity_field_default(kind: EntityKind) -> Option<EntityField> {
    entity_fields_for(kind).first().copied()
}

/// Step the focused [`EntityField`] by `delta` (±1) within `kind`'s field list,
/// wrapping. A field not in `kind`'s list (a stale focus after a kind switch)
/// restarts at the list's first entry.
pub fn step_entity_field(kind: EntityKind, current: EntityField, delta: i32) -> EntityField {
    let list = entity_fields_for(kind);
    if list.is_empty() {
        return current;
    }
    let n = list.len() as i32;
    let i = list.iter().position(|&f| f == current).unwrap_or(0) as i32;
    list[(i + delta).rem_euclid(n) as usize]
}

/// The mutually-exclusive TOOL MODES the editor's `space` key dispatches on.
/// Exactly one tool mode is conceptually active at a time; [`EditorState::enter_tool_mode`]
/// is the single choke point that sets one mode's flags and clears every sibling's
/// flags + stale cross-mode selections.
///
/// `Brush` is the default geometry-paint mode (no special mode flag — all of
/// `shape_mode`/`stair_mode`/`vox_mode`/`material_mode` are false and `place` is
/// `None`). `Object` is entity-placement: it does NOT touch `place` (the caller
/// chooses the specific [`EntityKind`] separately), it only clears the OTHER tool
/// modes and stale selections.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolMode {
    /// Plain [`re_world::CellKind`] geometry paint (the default). Clears every
    /// special mode flag and `place`.
    Brush,
    /// 45° wedge/ramp shape brush (`shape_mode = true`).
    Shape,
    /// Staircase stamp tool (`stair_mode = true`).
    Stair,
    /// `.vox` box-model placement (`vox_mode = true`).
    Vox,
    /// Material authoring (`material_mode = true`).
    Material,
    /// Entity placement. Clears the other tool modes + stale selections but leaves
    /// `place` for the caller to set to the chosen [`EntityKind`].
    Object,
}

/// Which CENTER viewport currently carries the FOCUS cue (a purely visual
/// highlight + the [`EditorState::swap_viewports`] target). Click-to-focus sets
/// it; it does NOT change key routing — 2D keys edit and camera keys fly only
/// while RMB-captured (the existing modal separation in `mouse.rs`/`main.rs`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewportFocus {
    /// The top-down 2D grid viewport.
    Grid,
    /// The live 3D preview viewport.
    Preview,
}

/// Which panes are visible and how they are laid out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// Top-down editor on the left, live 3D preview on the right (default).
    Split,
    /// Top-down editor fills the width (more room to paint); panel still shown.
    Full2D,
    /// Live 3D preview fills the screen (walk-through); thin status line only.
    Full3D,
}

impl ViewMode {
    /// Cycle Split → Full2D → Full3D → Split.
    pub fn next(self) -> ViewMode {
        match self {
            ViewMode::Split => ViewMode::Full2D,
            ViewMode::Full2D => ViewMode::Full3D,
            ViewMode::Full3D => ViewMode::Split,
        }
    }
}

#[derive(Resource)]
pub struct EditorState {
    pub level: re_level::Level,
    pub map: Map,
    pub cam: CameraPose,
    pub cursor: (usize, usize), // (x, y) on the current z-slice
    pub z: usize,               // active z-layer
    /// Which CENTER viewport carries the FOCUS cue (header highlight + the swap
    /// target). A visual cue only — click-to-focus sets it; it does NOT change key
    /// routing (2D keys edit, camera keys fly only while RMB-captured). Default
    /// [`ViewportFocus::Grid`].
    pub focused_viewport: ViewportFocus,
    /// SWAP: when `true`, the Split-mode center is laid out PREVIEW (left) then GRID
    /// (right) instead of the default GRID (left) then PREVIEW (right). Toggled by
    /// the `[<>]` viewport-header control. Only Split is affected; Full2D/Full3D
    /// ignore it. Default `false`.
    pub swap_viewports: bool,
    /// Integer top-down zoom: 1×/2×/3× framebuffer cells per world cell.
    pub zoom: u8,
    /// Top-down pan origin: the world cell `(x, y)` shown at the grid pane's
    /// top-left corner. Auto-follows the cursor; see `view::follow_pan`.
    pub pan: (usize, usize),
    pub brush: re_world::CellKind,
    /// `true` when the **shape brush** is active: `space` paints a 45° wedge/ramp
    /// (a solid base cell + the selected [`Shape`]) instead of the plain
    /// [`re_world::CellKind`] brush. Toggled by `7`.
    pub shape_mode: bool,
    /// The currently-selected 45° orientation painted while `shape_mode` is on.
    /// Cycled by `[` / `]`. Never `Shape::Full` (that is just erase).
    pub shape_sel: Shape,
    /// `true` when the **staircase stamp tool** is active: `space` then stamps a
    /// whole climbable staircase (straight or spiral) at the cursor in one
    /// undoable action via the [`re_level::staircase`] prefab generator, instead
    /// of painting a single cell. Toggled by `8`; mutually exclusive with the
    /// other paint/place modes. See [`crate::staircase`].
    pub stair_mode: bool,
    /// Which staircase sub-mode `8`/`space` stamps: a straight run or a spiral.
    /// Toggled by `\` while in stair mode. See [`crate::staircase::StairSub`].
    pub stair_sub: crate::staircase::StairSub,
    /// The straight staircase's rise direction ([`RampDir`]). Cycled by `[` / `]`
    /// while in stair mode (mirrors the shape brush's orientation keys).
    pub stair_dir: RampDir,
    /// The spiral staircase's footprint radius (cells from the central column).
    /// Adjusted by `+` / `-` while in stair-spiral mode, clamped `>= 1`.
    pub stair_radius: i32,
    /// The spiral staircase's number of full turns. Adjusted by `;` / `'` while
    /// in stair-spiral mode (so it stays disjoint from the radius keys), min 1.
    pub stair_turns: u32,
    /// AUTO-CARVE STAIRWELL: when `true` (the default), stamping a staircase also
    /// punches a stairwell hole through the floor slab(s) above the run's XY
    /// footprint (one storey-boundary z per climbed storey) so the player can
    /// climb through without manually erasing the ceiling. Part of the SAME
    /// undoable stamp (the snapshot in `apply_tool_at_cursor` covers it). Toggled
    /// by `#` while in stair mode. See [`crate::staircase::carve_stairwell`].
    pub auto_carve_stairwell: bool,
    pub view_mode: ViewMode,
    /// OBJECT SELECTOR: `Some(kind)` when entity-placement mode is active —
    /// `space` then places/selects that entity at the cursor instead of painting
    /// geometry, and `o` cycles the kind. `None` ⇒ ordinary geometry painting.
    pub place: Option<EntityKind>,
    /// The puzzle node currently being EDITED (index into `level.nodes`), set
    /// when placing-onto or deleting selects an existing node so the io/threshold
    /// keys target it. `None` ⇒ no node selected.
    pub selected_node: Option<usize>,
    /// The particle emitter currently being EDITED (index into `level.emitters`),
    /// set when placing-onto or deleting selects an existing emitter so the
    /// focused-field keys target it. `None` ⇒ no emitter selected.
    pub selected_emitter: Option<usize>,
    /// The trigger FIELD currently being EDITED (index into `level.triggers`), set
    /// when placing/re-placing onto an existing trigger's region or deleting one so
    /// the focused-field keys (region/action/once) target it. `None` ⇒ none.
    pub selected_trigger: Option<usize>,
    /// The PIPE currently being EDITED (index into `level.pipes`), set when the
    /// pipe-create flow finishes, when a pipe is picked from the tree, or when a
    /// pipe-path cell is re-selected. `None` ⇒ no pipe selected. The focused-field
    /// keys (state/switch_at/density/color) target it.
    pub selected_pipe: Option<usize>,
    /// Pipes authored as a SPLINE rather than a `path`, keyed by their `(from, to)`
    /// node-name pair (pipes are unique by `A->B`, and the pair survives the
    /// save/reload re-resolution that re-indexes nodes). A pipe in this set
    /// serializes its points as `spline = …` (smooth Catmull-Rom control points);
    /// a pipe NOT in it serializes as `path = …` (the default). The set is the
    /// SSOT for the `path`/`spline` mode toggle — `re_level::Pipe` carries no such
    /// flag, so the editor owns it. Seeded on load by scanning the raw `.lvl` for
    /// `[pipe] spline =` keys; consumed on save by [`crate::io`] to rewrite the
    /// emitted `path =` line into `spline =` for member pipes.
    pub spline_pipes: std::collections::HashSet<(String, String)>,
    /// The ZONE INPUT currently being EDITED (index into `level.zone_inputs`), set
    /// when placing/re-placing onto an existing zone's region or deleting one so the
    /// focused-field keys (region/input-node) target it. `None` ⇒ none.
    pub selected_zone_input: Option<usize>,
    /// The NOTE currently being EDITED (index into `level.notes`), set when
    /// placing/re-placing onto an existing note's cell or deleting one so the
    /// focused-field keys (kind/facing/pos/text) target it. `None` ⇒ none.
    pub selected_note: Option<usize>,
    /// The WALL DECAL currently being EDITED (index into `level.decals`), set when
    /// placing/re-placing onto an existing decal's region or deleting one so the
    /// focused-field keys (surface/region/density/colours) target it. `None` ⇒ none.
    pub selected_decal: Option<usize>,
    /// The HOLOGRAM currently being EDITED (index into `level.holos`), set when
    /// placing/re-placing onto an existing hologram's region or deleting one so the
    /// focused-field keys (shape/region/style) target it. `None` ⇒ none.
    pub selected_holo: Option<usize>,
    /// The WARP PORTAL currently being EDITED (index into `level.portals`), set
    /// when placing/re-placing onto either of the portal's two regions or deleting
    /// one so the focused-field keys (region A / region B / rotate / skin) target
    /// it. `None` ⇒ none.
    pub selected_portal: Option<usize>,
    /// The overlay ZONE currently being EDITED (index into `level.overlay.zones`),
    /// set when placing/re-placing onto an existing zone's region or deleting one so
    /// the focused-field keys (region / material / scale / insets) target it.
    /// `None` ⇒ none.
    pub selected_zone: Option<usize>,
    /// The overlay LINE currently being EDITED (index into `level.overlay.lines`),
    /// set when placing/re-placing onto a line's waypoint or deleting one so the
    /// focused-field keys (material / scale / waypoints) target it. `None` ⇒ none.
    pub selected_line: Option<usize>,
    /// The overlay INSET currently being EDITED (index into `level.overlay.insets`),
    /// set when placing/re-placing onto an inset's region or deleting one so the
    /// focused-field keys (octave / region / child) target it. `None` ⇒ none.
    pub selected_inset: Option<usize>,
    /// The standalone SOUND FIELD currently being EDITED (index into
    /// `level.field_defs`, and — 1:1, by construction of
    /// `re_level::fields::resolve` — into `level.resolved_fields` whenever the
    /// last resolve succeeded), set when placing/re-placing onto an existing
    /// field's cell or deleting one so the focused-field keys target it.
    /// `None` ⇒ none. See `crate::field`.
    pub selected_field: Option<usize>,
    /// The inset the selected Zone's `ZoneInsets` cursor points at (index into
    /// `level.overlay.insets`) — the `+`/`-` keys move it through the inset list and
    /// `c` toggles its name in the zone's `insets` Vec. Clamped against the inset
    /// list; meaningless when there are no insets. Mirrors [`Self::door_target_cursor`].
    pub zone_inset_cursor: usize,
    /// The PART currently being EDITED (index into `level.parts`), set when
    /// placing/re-placing onto an existing part's anchor cell or deleting one so
    /// the focused-field keys (size/offset/color) + the mask-edit modal target it.
    /// `None` ⇒ none.
    pub selected_part: Option<usize>,
    /// The PART the Node `NodeEmits` cursor points at (index into `level.parts`) —
    /// the `+`/`-` keys move it and `c` toggles a `[emit]` clause linking the
    /// selected node to it. Clamped against the part list; meaningless when there
    /// are no parts. Mirrors [`Self::door_target_cursor`].
    pub emit_part_cursor: usize,
    /// MASK-EDIT modal: `true` while authoring the selected part's solid mask —
    /// `apply_key` is modal (arrows move the part-local cursor, `[`/`]` change the
    /// active h-layer, `space`/`x` set solid/empty, `q` exits). Entered by `q` on a
    /// selected Part. `false` ⇒ normal editing. See `edit.rs` + `part.rs`.
    pub mask_mode: bool,
    /// The active part-local h-LAYER (`ly`) shown in the mask-edit 2D footprint
    /// view. Stepped by `[`/`]` while `mask_mode`, clamped `0..part.h`. The on-screen
    /// grid axes are `lx` (horizontal) × `lz` (vertical); this is the layer index.
    pub mask_layer: usize,
    /// The part-local `(lx, lz)` cursor within the mask-edit footprint view. The
    /// h-layer is [`Self::mask_layer`]; the full voxel is `(lx, mask_layer, lz)`.
    /// Moved by the arrows while `mask_mode`. Default `(0, 0)`.
    pub mask_cursor: (usize, usize),
    /// The INPUT node the selected zone's `ZoneNode` field points at (index into
    /// `level.nodes`, restricted to INPUT nodes when cycled). The `c` / `+`/`-`
    /// keys move it; the zone's `node` name is set from it. Mirrors
    /// [`Self::door_target_cursor`]. Meaningless when there are no input nodes.
    pub zone_node_cursor: usize,
    /// PIPE-CREATE in flight: `Some(src_node)` once the SOURCE node is picked and
    /// the editor is waiting for the TARGET node (the second `space` on a node, or
    /// a tree Enter on a node). `None` ⇒ no pipe-create pending. Mirrors the
    /// reparent two-step so a pipe is wired without inventing a new modal.
    pub pipe_pending: Option<usize>,
    /// Which parameter of the selected emitter the `+`/`-`/`c` keys edit. Cycled
    /// by `;` (next) / `'` (prev). Only meaningful while an emitter is selected.
    pub emitter_field: EmitterField,
    /// Which FOCUSED FIELD of the active NON-emitter entity (Door/Node/Pedestal/
    /// Spawn) the `;`/`'` keys page through and the `c` / `+`/`-` keys edit. Reset
    /// to the kind's first field when the object selector switches kind. Only
    /// meaningful while `place` is a non-Emitter kind.
    pub entity_field: EntityField,
    /// The node the Door target-cursor points at (index into `level.nodes`) — the
    /// `DoorTargetOn` / `DoorTargetOff` fields move it with `+`/`-` and toggle that
    /// node's membership with `c`. Clamped against the node list; meaningless when
    /// there are no nodes.
    pub door_target_cursor: usize,
    /// MATERIAL mode (M5b): `true` when material authoring is active — `space`
    /// then PAINTS the selected material's palette id onto the cursor cell (and
    /// `x` clears it) instead of geometry, and `+`/`-` edit the focused field.
    /// Entered/cycled by `m`. `false` ⇒ ordinary geometry painting.
    pub material_mode: bool,
    /// Index of the material being selected/edited (into `level.materials`). Its
    /// 1-based palette id is `material_sel + 1`. Cycled by `m`/`M`.
    pub material_sel: usize,
    /// Which appearance field of the selected material `+`/`-` edit. Cycled by `b`.
    pub material_field: crate::material::MaterialField,
    /// OFF-LIST CYCLE VOCABULARY (see [`crate::vocab`]) — the values this session
    /// has seen that the hand-written option lists do not offer (a node glyph the
    /// author typed, a renderer preset the catalog carries no name for, a
    /// non-cardinal emission direction, …). Seeded by `Vocab::harvest` when a
    /// level is opened and extended by every cycle that steps off an off-list
    /// value, so `c` can always come back to where it started. Session state, not
    /// content: undo/redo (`history::Snapshot`) deliberately does not restore it.
    pub vocab: crate::vocab::Vocab,
    /// VOX / BOX-MODEL mode (M5c): `true` when vox-placement is active — `space`
    /// then STAMPS a `.vox` model (the selected file) at the cursor as a
    /// `LevelVox`, `x` deletes the nearest stamp, `c` cycles the surface
    /// material, and `+`/`-` adjust the integer scale. Entered/cycled by `v`.
    /// `false` ⇒ ordinary geometry painting.
    pub vox_mode: bool,
    /// The scanned list of selectable `.vox` file paths (from `levels/`,
    /// relative to the process cwd so a stamped `file =` resolves the same way
    /// the shipped levels' `[vox]` paths do). Empty ⇒ no models found; the user
    /// can still place by a typed path (none here, see `vox.rs`).
    pub vox_files: Vec<String>,
    /// Index into [`vox_files`](Self::vox_files) of the currently-selected model.
    /// Cycled by `v`. Clamped against the list length.
    pub vox_file_sel: usize,
    /// Surface material a freshly-placed (or re-placed/selected) vox stamp uses.
    /// Cycled by `c` in vox mode.
    pub vox_material: re_level::VoxMaterial,
    /// Integer downscale a freshly-placed vox stamp uses (`1` = full res).
    /// Adjusted by `+`/`-` in vox mode, clamped `>= 1`.
    pub vox_scale: i32,
    /// The vox stamp currently SELECTED for editing (index into `level.voxes`),
    /// set when re-placing on an existing stamp's anchor so `c`/`+`/`-` retarget
    /// it. `None` ⇒ edits apply to the NEXT placement's defaults.
    pub selected_vox: Option<usize>,
    /// Free-fly noclip: when `true` the preview camera ignores collision.
    pub noclip: bool,
    /// OBJECT MARKER VISIBILITY: when `true` (the default) the top-down grid paints
    /// the entity + pipe markers (emitters/nodes/door/spawn/pedestal/triggers/vox/
    /// pipes) over the geometry. When `false` only the raw geometry (walls/floors/
    /// shapes) is drawn, so the author can read the map underneath the objects. The
    /// camera arrow + edit cursor are NEVER gated by this. Toggled by `H` and the
    /// `[x] Show objects` checkbox in the side panel's OBJECTS section.
    pub show_objects: bool,
    /// ONION SKIN (view aid): ghost the occupied cells of the layer(s) BELOW and
    /// ABOVE the edit layer into the 2D pane, dimmed and direction-tinted, so a
    /// stairwell or shaft lines up without flipping z. [`crate::viewaids::OnionSkin::Off`] by
    /// default, and while off `chrome::resolve_cell` runs its original path
    /// untouched. Cycled by `G`. A ghost is PAINT ONLY — the hit-test never sees
    /// it, so a click always edits the live layer. See [`crate::viewaids`].
    pub onion: crate::viewaids::OnionSkin,
    /// SOUND-FIELD view aid: when
    /// `true`, the 2D grid tints each field's footprint (ring at `radius`,
    /// brighter ring at `inner`) by archetype family, and the 3D preview marks
    /// each field's position. `false` by default, and while off neither
    /// painter is touched — `viewaids::paint_tests`'s off-path byte-identity
    /// guarantee extends to this aid the same way it does to onion skin.
    /// Toggled by `F`. See [`crate::field`].
    pub fields_aid: bool,
    /// The opt-in LIVE AUDITION:
    /// `Some` while `Shift+U` has armed it — the level's ambience + drone play
    /// through a real (or, off TS_AUDIO_SILENT/no-device, silently degraded)
    /// `re_audio::Player`, with field weights re-evaluated from the preview
    /// camera every frame. `None` (the default) opens no device at all — see
    /// [`crate::field::Audition::start`]'s gating.
    pub audition: Option<crate::field::Audition>,
    /// `true` when the map has unsaved edits since the last load/save.
    pub dirty: bool,
    /// Comments + `[map]`/`[win_when]` raw text preserved from the loaded file.
    /// Captured by [`crate::open::load_path`]; re-emitted by the
    /// `crate::io` save post-pass. Empty for a fresh/wizard document.
    /// NOTE(wizard): `crate::wizard::apply` builds a new document by copying
    /// fields from `new_empty` — it resets this (and `had_cells_at_load`) via
    /// [`EditorState::reset_load_provenance`]; keep that call when touching it.
    pub preserved: PreservedText,
    /// `true` when the loaded `.lvl` carried an explicit `[cells]` section,
    /// for the conditional-baking decision on save: a level that was ALREADY
    /// baked at load may always re-bake on save; a procedural level must not be
    /// silently flattened. `false` for a fresh/wizard document and for a loaded
    /// level whose geometry is purely procedural (`[room]`/`[part]`/`[vox]`).
    /// Set only by [`crate::open::load_path`] / [`EditorState::reset_load_provenance`].
    pub had_cells_at_load: bool,
    pub path: Option<String>,
    pub status: String,
    /// Bounded snapshot UNDO/REDO history (`u` / `y`). Captures `(map, level)`
    /// before each mutating edit; see `crate::history`.
    pub history: crate::history::History,
    /// Geometry FILL anchor: corner A of a rect/line fill, set by `k` at the
    /// cursor. `Some((x, y, z))` while a fill corner is pending; the second key
    /// (`K` rect / `L` line) consumes it. `None` ⇒ no anchor set.
    pub fill_anchor: Option<(usize, usize, usize)>,
    /// Active NEW-LEVEL wizard form (`w` opens it). `Some` ⇒ the wizard panel is
    /// shown and captures `+`/`-`/Tab/Enter/Esc; `None` ⇒ normal editing.
    pub wizard: Option<crate::wizard::Wizard>,
    /// Active LEVEL SETTINGS form (`[Level]` toolbar button opens it). `Some` ⇒
    /// the settings modal is shown and captures keys; `None` ⇒ normal editing.
    /// See [`crate::settings`].
    pub settings: Option<crate::settings::SettingsState>,
    /// OBJECT TREE / OUTLINER navigation state (`t` toggles it). When
    /// `tree.open`, the panel is shown + focused: Up/Dn navigate, Enter selects +
    /// jumps to an object, `h` folds a group, `z` duplicates, `d` deletes. The
    /// row list itself is rebuilt each frame from `level.*` (a read view), so only
    /// the cursor + fold flags persist here. See `crate::tree`.
    pub tree: crate::tree::TreeState,
    /// Active object-TYPE SELECTOR palette (`i` opens it). `Some(idx)` ⇒ the
    /// create menu is shown with item `idx` highlighted; Up/Dn move, Enter picks
    /// a type → placement mode. `None` ⇒ palette closed. Mutually exclusive with
    /// the focused tree panel.
    pub palette: Option<usize>,
    /// REPARENT-PENDING: `Some(child)` when `p` was pressed on a tree object and
    /// the editor is now waiting for the user to navigate to the intended PARENT
    /// row and press `p` again to confirm. `None` ⇒ no reparent in progress. The
    /// nesting hierarchy itself lives in `level.hierarchy` (so it round-trips +
    /// undoes with the level); this only tracks the in-flight pick.
    pub reparent_pending: Option<re_level::HierRef>,
    /// TOGGLEABLE HELP popup: `true` ⇒ the centered cheat-sheet overlay is shown
    /// and `apply_key` is modal (the sheet's own scroll / filter / close keys
    /// act, nothing else). Toggled by `?`; Esc also closes it. Default `false`
    /// (the permanent help wall was removed). Open it through [`Self::open_help`]
    /// and close it through [`Self::close_help`] so the scroll offset and the
    /// filter can never be left stale from the previous visit.
    pub help_open: bool,
    /// HELP SHEET SCROLL: the index of the first `keymap::help_lines_filtered`
    /// line painted in the popup body.
    ///
    /// The sheet is ~180 lines and the body window on a 40-row terminal is ~35,
    /// so without this roughly four fifths of the table was unreachable — every
    /// row after "view / files" existed in the generated sheet and nowhere a
    /// user could read it. Always CLAMPED through
    /// [`crate::chrome::clamp_help_scroll`] (never stored past the last page),
    /// and reset to 0 by [`Self::open_help`] and by any filter edit.
    pub help_scroll: usize,
    /// HELP SHEET FILTER: the live query typed into the open sheet; empty ⇒ the
    /// whole sheet. A row survives when the query is a case-insensitive
    /// substring of its chord, its label, or its group
    /// (`keymap::help_lines_filtered`). Mirrors the outliner's `/` filter:
    /// printable characters append, Backspace pops, Esc clears it before Esc
    /// closes the sheet (`main::quit_on_esc`).
    pub help_filter: String,
    /// The BODY HEIGHT the help popup was last PAINTED with, published by
    /// `draw_system` exactly as [`Self::view_cells`] publishes the grid
    /// viewport. `PageUp`/`PageDown`/`End` and the wheel need the real window
    /// height to step and to clamp, and the key handlers do not know the
    /// framebuffer size; deriving it from the painter is what keeps "one page"
    /// meaning the page the user is actually looking at. `0` until the sheet has
    /// been painted once (the steppers then fall back to a single line).
    pub help_page: usize,
    /// LOAD / OPEN overlay: `Some` ⇒ the in-editor file-list picker is shown and
    /// `apply_key` is modal (Up/Dn move, Enter loads, Esc/`O` close). Opened by
    /// the toolbar `[Open]` button or `O`. `None` ⇒ closed.
    pub open: Option<OpenState>,
    /// SAVE-AS text-entry overlay: `Some` ⇒ the in-editor filename prompt is shown
    /// and `apply_key` is modal (type chars, Enter confirm, Esc cancel). Opened by
    /// `A`, the `[SaveAs]` toolbar button, or automatically when a never-saved
    /// document is saved (`path == None`). `None` ⇒ closed. See `crate::saveas`.
    pub saveas: Option<SaveAsState>,
    /// DIRTY-OPEN GUARD: `true` once a first load attempt was made (Enter / row
    /// click) while the document is dirty — a second attempt then discards and
    /// loads, mirroring the `quit_armed` two-step. Reset on a successful load,
    /// overlay close, or successful save. Prevents losing unsaved work on Open.
    pub open_armed: bool,
    /// OVERWRITE GUARD: `true` once a Save-As confirm named a file that
    /// ALREADY EXISTS on disk — a second Enter then overwrites it, mirroring the
    /// `quit_armed`/`open_armed` two-step. Cleared by opening/closing the prompt
    /// and by every buffer edit (the armed name must be the one confirmed), so a
    /// mistyped name can never silently clobber a shipped level.
    pub saveas_armed: bool,
    /// DIRTY-QUIT GUARD: `true` once a first Esc was pressed while the document
    /// is dirty — a second Esc then quits, any other key disarms it. Prevents
    /// losing unsaved work to a stray Esc.
    pub quit_armed: bool,
    /// PREVIEW MOUSE-CAPTURE: `true` while the preview pane has "launched" the
    /// cursor into continuous free-look. The OS pointer grab itself lives in the
    /// `re_present::OsCursorGrab` resource; this is the editor-side mirror kept in
    /// sync by `preview_capture_look_system`. While captured the pure `apply_mouse`
    /// is modal over the grid (no paint) and the camera rotates from
    /// `OsCursorGrab::poll_delta` each frame instead of from drag deltas.
    pub preview_captured: bool,
    /// Bridge between the pure `apply_mouse` and the Bevy system that owns the
    /// `OsCursorGrab` resource: `apply_mouse` cannot call `activate`/`deactivate`
    /// (it has no resource access), so it records the desire here and a system
    /// drains it. `Some(true)` ⇒ request activate, `Some(false)` ⇒ request
    /// deactivate, `None` ⇒ no pending request.
    pub capture_request: Option<bool>,
    /// DIRECTION PICKER overlay: `Some` ⇒ the 3×3×3 cube aim picker is shown and
    /// `apply_key` is modal (cursor keys move within the layer, PageUp/Dn switch
    /// layer, Enter/space picks the cell → sets the emitter's `direction`, Esc
    /// cancels). Opened by `D` on a selected emitter. `None` ⇒ closed. See
    /// [`crate::picker`].
    pub dir_picker: Option<crate::picker::DirPickerState>,
    /// PAINT-STROKE in flight: `true` between the first paint of a mouse drag and
    /// its release, so the whole drag collapses into ONE undo entry (the snapshot
    /// is pushed once when the stroke starts, not per painted cell). Set/cleared by
    /// the Stage-3 mouse paint path. Default `false`.
    pub stroke_active: bool,
    /// The last grid cell `(x, y, z)` painted in the current stroke, so a drag that
    /// re-enters the same cell does not re-paint it. `None` ⇒ no stroke cell yet (or
    /// no stroke in flight). Default `None`.
    pub stroke_last_cell: Option<(usize, usize, usize)>,
    /// MIDDLE-DRAG PAN anchor: the framebuffer pointer cell `(col, row)` where the
    /// current middle-button pan started, so a drag computes the world-cell delta
    /// from it and subtracts it from `pan` (dragging the map moves the view).
    /// `Some` while a middle-drag is in flight (set on Middle Press, updated on each
    /// Middle Drag, cleared on Release). Default `None`.
    pub last_pan_ptr: Option<(u16, u16)>,
    /// Last-known VISIBLE grid cell area `(cols, rows)` of the top-down pane,
    /// EXCLUDING the rulers — written by the draw system each frame. Keyboard/mouse
    /// zoom reads it to anchor the zoom around the cursor. Default `(0, 0)` until the
    /// first frame measures the layout.
    pub view_cells: (usize, usize),
    /// The cursor position `(x, y)` as of the PREVIOUS frame, so the draw system can
    /// run follow-pan ONLY when the cursor actually moved (manual panning then
    /// persists while the cursor is stationary). Initialised to the starting cursor.
    pub last_cursor: (usize, usize),
    /// DEBOUNCED PREVIEW REBUILD: `Some(frames_left)` while a map rebuild
    /// is pending. Entity/material edit paths call
    /// [`schedule_preview_rebuild`](Self::schedule_preview_rebuild); the 60 Hz
    /// `preview_rebuild_system` decrements it each frame via
    /// [`crate::vox::tick_preview_rebuild`] and rebuilds when it hits zero —
    /// ~150 ms after the LAST edit, never per keystroke. `None` ⇒ nothing pending.
    pub preview_rebuild_delay: Option<u8>,
    /// The PROCEDURAL BASELINE map for [`crate::vox::rebuild_map`]'s
    /// baseline-diff merge: the last `level.build_map()` result BEFORE
    /// user edits were carried onto it. A live-map cell that diverges from this
    /// baseline is a hand edit (paint / erase / shape / painted material id)
    /// and wins over the fresh stamp on rebuild; everything else takes the
    /// fresh value so entity placements/deletions show up. Kept in step by
    /// `new_empty`, `open::load_path`, `wizard::apply`, `resize::resize_level`,
    /// and every `rebuild_map`. `None` only defensively (legacy fallback).
    pub preview_baseline: Option<Map>,
    /// The `[cells]` grid AS LOADED (a clone of `level.cells` at
    /// `open::load_path` time; `None` for procedural levels and fresh/wizard
    /// documents). [`crate::vox::rebuild_map`] builds its fresh map against
    /// THIS provenance rather than the live `level.cells` — the save path
    /// (`io::capture_and_serialize`) overwrites `level.cells` with a full map
    /// capture, and rebuilding against that stale capture would freeze every
    /// procedural stamp into geometry (the exact wipe bug the baseline-diff
    /// merge removes). Reset by resize (which invalidates the authored grid)
    /// and the wizard.
    pub loaded_cells: Option<re_level::CellGrid>,
    /// F2 RENAME overlay: `Some` ⇒ the rename prompt is shown and `apply_key` is
    /// modal (type chars, Enter commit + cascade, Esc cancel). Opened by `F2` on
    /// a selected/outliner object. `None` ⇒ closed. See [`crate::rename`].
    pub rename: Option<crate::rename::RenameState>,
    /// F5 PLAYTEST warn-and-confirm guard: `true` once an F5 on a level
    /// lvlcheck FAILs was pressed — a second F5 then launches anyway; any other
    /// key disarms it (cleared at the top of `apply_key`, mirroring `quit_armed`).
    pub playtest_armed: bool,
    /// F5 PLAYTEST request flag: set by the pure `apply_key` F5 arm; drained by
    /// the exclusive `playtest_system` in `main.rs` (it needs `&mut World` for
    /// `re_present::suspend_terminal`/`resume_terminal`).
    pub playtest_request: bool,
    /// AUTOSAVE timer: seconds accumulated while the document is dirty.
    /// Ticked by `autosave_system` in `main.rs` from `Res<Time>`; when it reaches
    /// [`crate::io::AUTOSAVE_INTERVAL_SECS`] a reparse-guarded sidecar autosave is
    /// written to `levels/.autosave/<stem>.lvl`. Reset to 0 whenever an autosave
    /// is written, a real save succeeds, or the document becomes clean.
    pub autosave_elapsed: f32,
    /// CRASH-RECOVERY OFFER: `Some(sidecar_path)` when
    /// [`crate::open::load_path`] found an autosave sidecar with a strictly NEWER
    /// mtime than the file just opened. A transient pre-chain prompt (NOT a
    /// modal) in `edit::apply_key` answers it: `y` restores the sidecar into the
    /// editor (path stays the original, document marked dirty), `n` discards the
    /// offer; any other key clears it and acts normally. `None` ⇒ no offer.
    pub autosave_offer: Option<String>,
    /// TEXT-EDIT CARET: the caret for the currently focused text surface
    /// (note body / inset child path / pool words / trigger event label). Only
    /// meaningful while [`EditorState::text_owner`] matches the focused target;
    /// every widget op clamps it, so a stale value can never splice wrongly.
    pub text_caret: crate::widget::Caret,
    /// Which `(kind, object index, field)` the caret/buffer currently belong
    /// to. When the focused text target changes (other object, other field),
    /// `entity::text_entry_key` re-seeds the caret at end-of-text and rebuilds
    /// the buffer, so edits never leak across objects.
    pub text_owner: Option<(EntityKind, usize, EntityField)>,
    /// FREE-TEXT EDIT BUFFER for the BUFFERED text fields — node
    /// `pool`/`pool_alt` (space-separated words; committed to the word list on
    /// every keystroke via `split_whitespace`) and the trigger EVENT label
    /// (committed trimmed). The buffer preserves in-progress whitespace the
    /// committed model would drop (e.g. the trailing space while typing the
    /// next word). `None` for the direct (unbuffered) note/inset targets.
    pub text_buf: Option<String>,
    /// Per-panel collapse preferences + the adaptive-shedding
    /// switch. SESSION-ONLY state — no config file is persisted.
    /// The live editor runs with `adaptive: true`; the effective (width-shed)
    /// states are derived per frame via [`crate::layout::effective_panels`].
    pub panels: crate::layout::PanelPrefs,
    /// The `z` zoom-to-pane toggle — `Some(pane)` fullscreens that center
    /// viewport (toolbar/status kept); the same key restores `None`.
    pub zoomed: Option<crate::layout::PanelId>,
    /// The `:` COMMAND PALETTE — a one-line fuzzy finder over
    /// every keymap action + goto-object + open-level entries. `Some` while
    /// open (a topmost focus-stack frame); Enter executes, Esc closes.
    pub cmd_palette: Option<CmdPaletteState>,
    /// The read-only VALIDATION DRAWER toggle (`V`) — lists the last
    /// save's full lvlcheck report ([`EditorState::last_check`]), scrollable.
    pub validation_open: bool,
    /// Scroll offset (first visible line) of the validation drawer.
    pub validation_scroll: usize,
    /// The FULL lvlcheck report of the last save (one formatted line per
    /// check, worst-first header included), captured by `io::save_lvl_text`.
    /// Empty until a save has run. Read-only — no jump-to (the Check type
    /// carries no coordinates to jump to).
    pub last_check: Vec<String>,
    /// The INLINE NUMERIC EDITOR over a typed `Int`/`Float` inspector row —
    /// `Some` while open (a focus-stack frame). Nothing is written to the level
    /// until its `Enter` commits, so `Esc` needs no snapshot to restore.
    pub num_edit: Option<crate::rowedit::NumEdit>,
    /// The ENUM OPTION POPUP over a typed `Enum` inspector row — `Some` while
    /// open. Replaces blind `c`-cycling with a list you can see.
    pub enum_pick: Option<crate::rowedit::EnumPick>,
    /// The COLOUR OVERLAY over a typed `Color` inspector row — `Some` while
    /// open. Hex entry + the palette grid + the level's own used colours; the
    /// recovery path for a value the old snap-to-white stepper destroyed.
    pub color_edit: Option<crate::rowedit::ColorEdit>,
}

/// The in-editor OPEN/LOAD overlay: a scanned list of `levels/*.lvl` paths plus a
/// highlighted cursor. Built when the overlay opens; navigated with Up/Dn; Enter
/// loads the selected file in-place (no restart).
#[derive(Clone, Debug, Default)]
pub struct OpenState {
    /// Scanned `.lvl` paths (sorted, `levels/<name>.lvl` form).
    pub files: Vec<String>,
    /// Highlighted row index into `files`.
    pub cursor: usize,
}

/// The `:` COMMAND PALETTE: a one-line fuzzy query over the
/// keymap's actions, the outliner's objects (`goto <name>`), and `levels/*.lvl`
/// (`open level <file>`). The candidate list is derived per keystroke by
/// [`crate::keymap::palette_matches`] — nothing here caches rows, so the
/// painter, the key handler, and the mouse hit-test always agree.
#[derive(Clone, Debug, Default)]
pub struct CmdPaletteState {
    /// The live fuzzy query the user is typing.
    pub query: String,
    /// Highlighted row index into the CURRENT match list (clamped on use).
    pub sel: usize,
    /// `levels/*.lvl` scanned ONCE when the palette opened (the `open level`
    /// candidates) — no per-frame disk I/O while the palette redraws.
    pub files: Vec<String>,
}

/// The in-editor SAVE-AS prompt: an editable filename buffer the user types into.
/// On confirm the buffer is normalized to a `levels/<name>.lvl` path (see
/// [`crate::saveas::normalize_path`]) and set as `state.path`. Held by
/// [`EditorState::saveas`] while the prompt is open.
#[derive(Clone, Debug, Default)]
pub struct SaveAsState {
    /// The editable filename buffer. Holds the bare name the user types (no
    /// `levels/` prefix, no `.lvl` suffix — those are added on confirm).
    pub buf: String,
}

/// TEXT PRESERVED from the loaded `.lvl` file that the typed `Level` model (and
/// therefore `re_voximport::to_lvl_text`) cannot carry: `#` comments, and the
/// `[map]`/`[win_when]` sections the parser skips (`re_level` `Section::Skip`).
/// Captured by [`crate::open::capture_preserved`] at load; re-emitted by the
/// `crate::io` save post-pass (the same seam as `rewrite_spline_pipes`), so an
/// open+save never silently drops a comment or a skipped-but-parseable section.
///
/// CAPTURE IS SECTION-AWARE: the bodies of `[cells]`, `[part]`, and
/// `[note]` are never comment-scanned — `#` there is GEOMETRY (`[cells]` rows map
/// `'#'`→`WallSolid`; `[part]` mask rows accept `#`) or verbatim note text. A naive
/// "lines starting with #" capture would destroy geometry while the file still
/// parses cleanly.
///
/// An empty `PreservedText` (fresh document, level authored in-editor) makes the
/// save post-pass a byte-identical no-op.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PreservedText {
    /// Full-line comment blocks (with their interleaved/trailing blank lines,
    /// verbatim) anchored to a position in the file. Consumed IN ORDER on
    /// re-emit, so duplicate headers (e.g. several `[trigger]` sections) resolve
    /// by file order.
    pub anchors: Vec<CommentAnchor>,
    /// Trailing inline comments (` # …` after code, per `re_level`'s
    /// `strip_comment` rules — a `#RRGGBB` color value is NOT a comment),
    /// re-attached to the matching emitted line. See `crate::io`'s module docs
    /// for the re-emit pass.
    pub trailing: Vec<TrailingComment>,
    /// Comment lines pending at EOF (after the last section's body / in a file
    /// with no sections). Re-emitted at the end of the saved file.
    pub tail: Vec<String>,
    /// Verbatim `[map]` / `[win_when]` blocks — header line, body, and any
    /// comment block immediately above the header — exactly as loaded. The
    /// parser skips these sections and the writer never emits them, so without
    /// this they would be silently lost on open+save. Re-emitted at the end
    /// of the saved file (position is irrelevant to the parser).
    pub raw_blocks: Vec<String>,
}

impl PreservedText {
    /// `true` when there is nothing to re-emit (the save post-pass fast-path).
    pub const fn is_empty(&self) -> bool {
        self.anchors.is_empty()
            && self.trailing.is_empty()
            && self.tail.is_empty()
            && self.raw_blocks.is_empty()
    }
}

/// One captured full-line comment block and where to re-emit it.
///
/// NOTE(rename): anchors are keyed by the section-header STRING (e.g.
/// `"[node in_a]"`). [`crate::rename::rename_object`] must rewrite the `header` of every
/// anchor (and [`TrailingComment`]) whose section header it renames, or the
/// comments detach to the file tail on the next save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommentAnchor {
    /// The TRIMMED header line of the section this block belongs to, e.g.
    /// `"[meta]"` or `"[node in_a]"`. The block re-emits when the serialized
    /// text emits this exact header string.
    pub header: String,
    /// `None` ⇒ the block sat immediately ABOVE the section header (the common
    /// "leading block" — also the file-header block, anchored to the first
    /// section). `Some(key)` ⇒ an interior block that sat immediately above the
    /// `key = …` line (or the full trimmed line for key-less lines) inside the
    /// section; re-emitted before the first matching emitted line.
    pub key: Option<String>,
    /// The comment (and interleaved blank) lines, verbatim.
    pub lines: Vec<String>,
}

/// One captured trailing inline comment: `key = value   # like this`.
///
/// NOTE(rename): keyed by section-header string like [`CommentAnchor`] — a
/// header rename must rewrite `header` here too (see [`crate::rename::rename_object`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrailingComment {
    /// The TRIMMED header line of the owning section.
    pub header: String,
    /// Which line inside the section carried the comment.
    pub key: TrailKey,
    /// The comment suffix, verbatim, INCLUDING the whitespace run before `#`
    /// (so `code + suffix` reproduces the original line byte-for-byte when the
    /// emitted code text matches).
    pub suffix: String,
}

/// Addressing for a [`TrailingComment`] within its section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrailKey {
    /// The comment sat on the section-header line itself (`[node a] # hub`).
    Header,
    /// The comment sat on a body line; the key is the `key` of `key = value`,
    /// or the full trimmed code text for key-less lines (mask/hierarchy rows).
    Key(String),
}

/// A minimal, empty `re_level::Level`. `Level` does NOT implement `Default`
/// (see `crates/re_level/src/lib.rs`), so we construct one explicitly: a named
/// room of the given size with default sub-structs and empty collections.
fn empty_level(sx: usize, sy: usize, sz: usize) -> re_level::Level {
    // `re_level::Level` derives `Default`, so we only spell the three fields a
    // fresh room actually overrides and let `..Default::default()` supply the
    // rest. This keeps `empty_level` immune to additive `Level` fields (it broke
    // three times when each was hand-spelled: zone_inputs, color_cell_ids /
    // render_pipes, then sequence / cycle_keep).
    re_level::Level {
        name: "untitled".into(),
        title: "Untitled".into(),
        size: (sx, sy, sz),
        ..Default::default()
    }
}

impl EditorState {
    /// A fresh empty room of the given size with a default camera.
    pub fn new_empty(sx: usize, sy: usize, sz: usize) -> Self {
        let level = empty_level(sx, sy, sz);
        let map = Map::new(sx, sy, sz);
        // The procedural baseline for rebuild_map's diff merge. The map
        // starts EMPTY (a blank canvas) while the level's build stamps a shell —
        // that divergence counts as "user erased", which is exactly right: the
        // canvas the user sees stays authoritative across rebuilds.
        let preview_baseline = Some(level.build_map());
        EditorState {
            level,
            map,
            cam: CameraPose::spawn_at(1, 1, 1, 0),
            cursor: (0, 0),
            z: 0,
            focused_viewport: ViewportFocus::Grid,
            swap_viewports: false,
            zoom: 1,
            pan: (0, 0),
            brush: re_world::CellKind::WallSolid,
            shape_mode: false,
            shape_sel: SHAPE_CYCLE[0],
            stair_mode: false,
            stair_sub: crate::staircase::StairSub::Straight,
            stair_dir: RampDir::PosX,
            stair_radius: 1,
            stair_turns: 1,
            auto_carve_stairwell: true,
            view_mode: ViewMode::Split,
            place: None,
            selected_node: None,
            selected_emitter: None,
            selected_trigger: None,
            selected_pipe: None,
            spline_pipes: std::collections::HashSet::new(),
            selected_zone_input: None,
            selected_note: None,
            selected_decal: None,
            selected_holo: None,
            selected_portal: None,
            selected_zone: None,
            selected_line: None,
            selected_inset: None,
            selected_field: None,
            zone_inset_cursor: 0,
            selected_part: None,
            emit_part_cursor: 0,
            mask_mode: false,
            mask_layer: 0,
            mask_cursor: (0, 0),
            zone_node_cursor: 0,
            pipe_pending: None,
            emitter_field: EMITTER_FIELDS[0],
            entity_field: EntityField::DoorPosX,
            door_target_cursor: 0,
            material_mode: false,
            material_sel: 0,
            material_field: crate::material::MaterialField::Fg,
            vocab: crate::vocab::Vocab::default(),
            vox_mode: false,
            vox_files: crate::vox::scan_vox_files(),
            vox_file_sel: 0,
            vox_material: re_level::VoxMaterial::Solid,
            vox_scale: 1,
            selected_vox: None,
            noclip: true,
            show_objects: true,
            onion: crate::viewaids::OnionSkin::Off,
            fields_aid: false,
            audition: None,
            dirty: false,
            preserved: PreservedText::default(),
            had_cells_at_load: false,
            path: None,
            status: "new".into(),
            history: crate::history::History::default(),
            fill_anchor: None,
            wizard: None,
            settings: None,
            tree: crate::tree::TreeState::default(),
            palette: None,
            reparent_pending: None,
            help_open: false,
            help_scroll: 0,
            help_filter: String::new(),
            help_page: 0,
            open: None,
            saveas: None,
            open_armed: false,
            saveas_armed: false,
            quit_armed: false,
            preview_captured: false,
            capture_request: None,
            dir_picker: None,
            stroke_active: false,
            stroke_last_cell: None,
            last_pan_ptr: None,
            view_cells: (0, 0),
            last_cursor: (0, 0),
            preview_rebuild_delay: None,
            preview_baseline,
            loaded_cells: None,
            rename: None,
            playtest_armed: false,
            playtest_request: false,
            autosave_elapsed: 0.0,
            autosave_offer: None,
            text_caret: crate::widget::Caret::default(),
            text_owner: None,
            text_buf: None,
            // Adaptive shedding is ON in the live editor; the legacy layout
            // entry points (and their tests) run with `PanelPrefs::default()`
            // (adaptive off) via the `compute_swapped` delegation instead.
            panels: crate::layout::PanelPrefs {
                adaptive: true,
                ..Default::default()
            },
            zoomed: None,
            cmd_palette: None,
            validation_open: false,
            validation_scroll: 0,
            last_check: Vec::new(),
            num_edit: None,
            enum_pick: None,
            color_edit: None,
        }
    }

    /// The LIVE layout for a `w`×`h` framebuffer: the panel-aware
    /// [`crate::layout::compute_with_panels`] with this state's view mode,
    /// tree focus, swap flag, panel prefs, and zoom. THE single entry point
    /// `draw_system` (painters) and `apply_mouse` (hit-tests) share, so what
    /// is drawn and what is clickable can never disagree.
    pub fn layout_for(&self, w: usize, h: usize) -> crate::layout::Layout {
        crate::layout::compute_with_panels(
            self.view_mode,
            w,
            h,
            self.tree.open,
            self.swap_viewports,
            self.panels,
            self.zoomed,
        )
    }

    /// The EFFECTIVE (width-shed) panel states at framebuffer width `w` — the
    /// shared derivation for organ sub-rects, the rail painter, and the rail
    /// hit-test.
    pub fn panels_eff(&self, w: usize) -> crate::layout::EffPanels {
        crate::layout::effective_panels(self.view_mode, w, self.panels)
    }

    /// Schedule the DEBOUNCED preview map rebuild: (re)arms the ~150 ms
    /// countdown consumed by [`crate::vox::tick_preview_rebuild`]. Called from
    /// the entity/material edit choke points; a burst of edits keeps re-arming,
    /// so exactly one rebuild runs after the burst goes quiet — never one per
    /// keystroke. Pure state; the rebuild itself happens in the 60 Hz loop.
    pub const fn schedule_preview_rebuild(&mut self) {
        self.preview_rebuild_delay = Some(crate::vox::REBUILD_DEBOUNCE_FRAMES);
    }

    /// Mark the map edited (un-saved). Centralised so every mutating edit path
    /// flips the same flag.
    pub const fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Reset the per-document LOAD PROVENANCE — the preserved comment/raw text
    /// and the `had_cells_at_load` flag — to the fresh-document state. Called
    /// when the current document stops being "the file we loaded": the
    /// new-level wizard (`crate::wizard::apply`). [`crate::open::load_path`]
    /// instead overwrites both fields from the newly-loaded file. Without this,
    /// a wizard-fresh level would inherit the PREVIOUS file's comments (they
    /// would re-emit into the new file on save) and its `[cells]` provenance.
    pub fn reset_load_provenance(&mut self) {
        self.preserved = PreservedText::default();
        self.had_cells_at_load = false;
    }

    /// Open the `?` HELP SHEET at the TOP, unfiltered.
    ///
    /// THE one way in. Every entry point (the `?` key, the toolbar `[Help]`
    /// button, the `:` palette synthesizing `?`) routes here, because a sheet
    /// reopened with the previous visit's scroll offset still applied can open
    /// on a blank window — the offset was legal for the old filter and runs past
    /// the end of the new one — and a sheet reopened with the previous visit's
    /// FILTER applied looks like a sheet that lost most of its rows.
    pub fn open_help(&mut self) {
        self.help_open = true;
        self.help_scroll = 0;
        self.help_filter.clear();
        self.status = "help: type to filter, Up/Dn PgUp/PgDn scroll, ? or Esc to close".into();
    }

    /// Close the `?` HELP SHEET, dropping its scroll offset and filter.
    pub fn close_help(&mut self) {
        self.help_open = false;
        self.help_scroll = 0;
        self.help_filter.clear();
        self.status = "help closed".into();
    }

    /// `true` when ANY modal / overlay / focused outliner currently owns the
    /// keyboard: the crash-recovery autosave offer, the part MASK editor, the
    /// help popup, the directory picker, the open/save-as overlays, the
    /// new-level wizard, the level settings form, the create palette, the
    /// focused object tree, a pending reparent pick, the F2 rename prompt, the
    /// `:` command palette, or one of the three typed inspector-row editors.
    /// This is the predicate that suppresses the fly-camera and mouse-look while
    /// a modal is up — mirrors the `quit_on_esc` Esc chain in `main.rs`.
    ///
    /// THE ONE PREDICATE. `camera_input_system` gates on this and nothing else,
    /// and `apply_camera_key` binds `Enter` to jump-to-cursor unconditionally —
    /// so a modal missing from this list has its confirm key ALSO teleport the
    /// preview camera, because the two systems hold separate `MessageReader`s
    /// and both see the same keypress. The row editors and `cmd_palette` were
    /// each missing exactly that way. Anything that consumes Enter belongs here.
    ///
    /// The CRASH-RECOVERY OFFER and the MASK editor were the same hole one layer
    /// earlier: `edit::apply_key` dispatches both AHEAD of every other modal and
    /// swallows every key (the offer's `_` arm re-states the question; the mask
    /// editor's `_ => {}` eats the rest, Enter included), yet neither was listed
    /// here — so while a crash-recovery prompt was pending, WASD still flew the
    /// preview camera and Enter still teleported it.
    pub const fn any_modal_active(&self) -> bool {
        self.autosave_offer.is_some()
            || self.mask_mode
            || self.help_open
            || self.dir_picker.is_some()
            || self.open.is_some()
            || self.saveas.is_some()
            || self.wizard.is_some()
            || self.settings.is_some()
            || self.palette.is_some()
            || self.tree.open
            || self.reparent_pending.is_some()
            || self.rename.is_some()
            || self.cmd_palette.is_some()
            || self.num_edit.is_some()
            || self.enum_pick.is_some()
            || self.color_edit.is_some()
    }

    /// THE single choke point for switching tool modes, making them truly mutually
    /// exclusive. Sets EXACTLY the flags for `mode` and clears ALL siblings, plus
    /// every stale cross-mode selection.
    ///
    /// INVARIANT: after `enter_tool_mode`, at most one tool-mode flag
    /// (`shape_mode` / `stair_mode` / `vox_mode` / `material_mode`) is true and no
    /// selection belonging to a different mode remains
    /// (`selected_emitter` / `selected_pipe` / `selected_trigger` /
    /// `selected_node` / `selected_vox` / `pipe_pending` are all cleared).
    ///
    /// - [`ToolMode::Brush`] / `Shape` / `Stair` / `Vox` / `Material`: set their own
    ///   flag (none for `Brush`), clear the other three mode flags, and clear
    ///   `place` (geometry/material modes are not entity placement).
    /// - [`ToolMode::Object`]: clears all four mode flags but leaves `place`
    ///   untouched — the caller sets `place` to the chosen [`EntityKind`].
    ///
    /// Does NOT touch modal Options (`open` / `saveas` / `wizard` / …) and does NOT
    /// touch `fill_anchor`. The brush-hotkey arms can call `enter_tool_mode(Brush)`
    /// and then set `brush` themselves.
    pub fn enter_tool_mode(&mut self, mode: ToolMode) {
        // Set exactly this mode's flag; clear the rest.
        self.shape_mode = matches!(mode, ToolMode::Shape);
        self.stair_mode = matches!(mode, ToolMode::Stair);
        self.vox_mode = matches!(mode, ToolMode::Vox);
        self.material_mode = matches!(mode, ToolMode::Material);
        // Non-Object modes are not entity placement, so they drop `place`. Object
        // leaves `place` for the caller to set.
        if !matches!(mode, ToolMode::Object) {
            self.place = None;
        }
        // Clear stale cross-mode selections so switching tools never leaves a
        // dangling edit target from the previous mode.
        self.selected_emitter = None;
        self.selected_pipe = None;
        self.selected_trigger = None;
        self.selected_node = None;
        self.selected_vox = None;
        self.selected_zone_input = None;
        self.selected_note = None;
        self.selected_decal = None;
        self.selected_holo = None;
        self.selected_portal = None;
        self.selected_zone = None;
        self.selected_line = None;
        self.selected_inset = None;
        self.selected_field = None;
        self.selected_part = None;
        // Leaving the entity tool also closes any mask-edit modal so a stale mask
        // session can't outlive the part selection that opened it.
        self.mask_mode = false;
        self.pipe_pending = None;
    }

    /// Capture the current `(map, level)` onto the undo stack BEFORE a mutating
    /// edit, clearing the redo branch. THE single snapshot choke point — call it
    /// once at the start of every undoable edit (paint, erase, fill, entity /
    /// material / emitter / trigger edits). Pure clone; no I/O.
    pub fn push_undo(&mut self) {
        self.history.push(crate::history::Snapshot {
            map: self.map.clone(),
            level: self.level.clone(),
        });
    }

    /// COMMIT-IF-CHANGED, part 1: take an undo CHECKPOINT before running an
    /// operation whose handler may turn out to be a no-op (e.g. `c`/`+`/`-` on a
    /// mode with no selection or on an enum field under `+`/`-`). Snapshots the
    /// current `(map, level)` exactly like [`push_undo`](Self::push_undo) and
    /// returns the resulting undo depth. After running the handler, call
    /// [`drop_undo_if_unchanged`](Self::drop_undo_if_unchanged) to discard the
    /// snapshot when nothing actually changed — so exploring fields never stacks
    /// dead undo steps.
    ///
    /// Unlike [`push_undo`](Self::push_undo) this does NOT clear the redo branch:
    /// the checkpoint might be a no-op, and a no-op must leave redo intact. Redo is
    /// invalidated only when the checkpoint is KEPT (a real change), inside
    /// `drop_undo_if_unchanged`.
    pub fn push_undo_checkpoint(&mut self) -> usize {
        self.history.push_checkpoint(crate::history::Snapshot {
            map: self.map.clone(),
            level: self.level.clone(),
        });
        self.history.undo_len()
    }

    /// COMMIT-IF-CHANGED, part 2: if the document is byte-identical to the
    /// most-recent undo snapshot (the checkpoint just pushed), discard that
    /// snapshot — the operation was a no-op, so it must not leave a dead undo step.
    /// Returns `true` when a snapshot was dropped (no change), `false` when the
    /// edit really changed the document (the snapshot is kept). Mirrors the spirit
    /// of `pipe::pick_node`'s rollback but without polluting the redo stack.
    ///
    /// Redo handling: a NO-OP leaves both stacks untouched (redo survives, so an
    /// innocent `c`/`+`/`-` while idle can't wipe a redoable branch — fixes B1). A
    /// REAL change keeps the checkpoint and clears redo, matching `push_undo`.
    pub fn drop_undo_if_unchanged(&mut self) -> bool {
        let current = crate::history::Snapshot {
            map: self.map.clone(),
            level: self.level.clone(),
        };
        let unchanged = self
            .history
            .top_undo()
            .map(|snap| snap.same_document(&current))
            .unwrap_or(false);
        if unchanged {
            self.history.pop_undo();
        } else {
            // A real edit invalidates the redo branch (deferred from the checkpoint
            // push so a no-op never reaches here).
            self.history.clear_redo();
        }
        unchanged
    }

    /// UNDO: restore the most-recent pre-edit `(map, level)`, pushing the current
    /// state onto the redo stack. No-op (returns `false`) when the undo stack is
    /// empty. Leaves `dirty` set (the document still differs from the saved file
    /// in the general case); a precise saved-state comparison is a future nicety.
    pub fn undo(&mut self) -> bool {
        let current = crate::history::Snapshot {
            map: self.map.clone(),
            level: self.level.clone(),
        };
        match self.history.undo(current) {
            Some(prev) => {
                self.map = prev.map;
                self.level = prev.level;
                self.clamp_cursor();
                self.dirty = true;
                true
            }
            None => false,
        }
    }

    /// REDO: re-apply the most-recently undone `(map, level)`, pushing the current
    /// state back onto the undo stack. No-op (returns `false`) when nothing to redo.
    pub fn redo(&mut self) -> bool {
        let current = crate::history::Snapshot {
            map: self.map.clone(),
            level: self.level.clone(),
        };
        match self.history.redo(current) {
            Some(next) => {
                self.map = next.map;
                self.level = next.level;
                self.clamp_cursor();
                self.dirty = true;
                true
            }
            None => false,
        }
    }

    /// Re-clamp the cursor + active z into the current map bounds (a restored or
    /// freshly-created map may be smaller than the one the cursor was sized for).
    pub fn clamp_cursor(&mut self) {
        let (sx, sy, sz) = self.map.size();
        self.cursor.0 = self.cursor.0.min(sx.saturating_sub(1));
        self.cursor.1 = self.cursor.1.min(sy.saturating_sub(1));
        self.z = self.z.min(sz.saturating_sub(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_state_has_requested_size() {
        let s = EditorState::new_empty(8, 8, 4);
        assert_eq!(s.map.size(), (8, 8, 4));
        assert_eq!(s.brush, re_world::CellKind::WallSolid);
        assert_eq!(s.view_mode, ViewMode::Split);
        assert_eq!(s.zoom, 1);
        assert_eq!(s.pan, (0, 0));
        assert!(!s.dirty);
    }

    #[test]
    fn view_mode_cycles() {
        assert_eq!(ViewMode::Split.next(), ViewMode::Full2D);
        assert_eq!(ViewMode::Full2D.next(), ViewMode::Full3D);
        assert_eq!(ViewMode::Full3D.next(), ViewMode::Split);
    }

    #[test]
    fn shape_cycle_covers_all_eight_and_wraps() {
        // Forward through all 8 orientations and back to the start.
        let mut s = SHAPE_CYCLE[0];
        for expected in SHAPE_CYCLE.iter().skip(1) {
            s = cycle_shape(s, 1);
            assert_eq!(s, *expected);
        }
        // One more step wraps to index 0.
        s = cycle_shape(s, 1);
        assert_eq!(s, SHAPE_CYCLE[0]);
        // Backward from index 0 wraps to the last entry.
        assert_eq!(cycle_shape(SHAPE_CYCLE[0], -1), SHAPE_CYCLE[7]);
    }

    #[test]
    fn shape_cycle_includes_4_wedges_and_4_ramps() {
        let wedges = SHAPE_CYCLE
            .iter()
            .filter(|s| matches!(s, Shape::WedgeXY(_)))
            .count();
        let ramps = SHAPE_CYCLE
            .iter()
            .filter(|s| matches!(s, Shape::RampZ(_)))
            .count();
        assert_eq!(wedges, 4);
        assert_eq!(ramps, 4);
    }

    #[test]
    fn full_shape_restarts_cycle() {
        assert_eq!(cycle_shape(Shape::Full, 1), SHAPE_CYCLE[0]);
    }
}

#[cfg(test)]
mod decal_surface_tests {
    use super::{cycle_decal_surface, decal_surface_label, DECAL_SURFACES};
    use re_world::CellKind as CK;

    /// Every surface kind `re_voximport::to_lvl_text` can emit for a `[wall_decal]`
    /// (mirrors its `match d.surface_kind` arms + the `trace` default), which is
    /// exactly the set a level on disk can legitimately carry.
    const WRITABLE_SURFACES: [CK; 6] = [
        CK::WallTrace,
        CK::WallMatrix,
        CK::WallSolid,
        CK::Hologram,
        CK::Portal,
        CK::Object,
    ];

    /// A minimal `[wall_decal]` document carrying one `surface = <label>` line —
    /// the smallest input that exercises the parser arm this list must match.
    fn decal_lvl(label: &str) -> String {
        format!(
            "[meta]\nname = surface-pin\n\n\
             [room]\nsize = 4 4 2\n\n\
             [pedestal]\ncenter = 2 2 0\nvolume = 1 1 1\n\n\
             [door]\npos = 0 2 0\n\n\
             [player]\nspawn = 2 2 0\n\n\
             [wall_decal d0]\nsurface = {label}\nregion = (1,1,0)-(1,1,0)\n"
        )
    }

    /// THE ANTI-DRIFT PIN. Every `(CellKind, label)` the editor cycles through must
    /// be a spelling `re_level`'s `[wall_decal] surface` parser accepts, AND must
    /// parse back to exactly the `CellKind` the editor paired it with. This is the
    /// real defect the widening fixes: two hand-maintained lists that could drift.
    /// If `re_level` renames or drops a spelling, this fails here rather than
    /// silently rewriting a level's decal on the author's next `c` press.
    #[test]
    fn every_editor_decal_surface_is_a_spelling_the_parser_accepts() {
        for &(kind, label) in DECAL_SURFACES.iter() {
            let text = decal_lvl(label);
            let level = re_level::load_string(&text, "surface-pin.lvl")
                .unwrap_or_else(|e| panic!("parser rejects editor label '{label}': {e:?}"));
            assert_eq!(level.decals.len(), 1, "label '{label}' yields one decal");
            assert_eq!(
                level.decals[0].surface_kind, kind,
                "label '{label}' must parse to the kind the editor pairs it with"
            );
        }
    }

    /// The other direction: every surface kind `re_voximport::to_lvl_text` can EMIT
    /// (and therefore every kind a `.vox`-imported level can carry on disk) is on
    /// the editor's cycle. A kind the writer emits but the cycle omits is a kind
    /// the editor destroys on the first `c` press.
    #[test]
    fn every_writable_decal_surface_is_on_the_editor_cycle() {
        for kind in WRITABLE_SURFACES {
            assert!(
                DECAL_SURFACES.iter().any(|&(k, _)| k == kind),
                "{kind:?} is emitted by to_lvl_text but missing from DECAL_SURFACES — \
                 cycling it would destroy it"
            );
        }
    }

    /// THE BUG. `c` on an imported hologram/portal/object decal used to fall through
    /// `cycle_decal_surface`'s `None` arm and SNAP the surface to `trace`,
    /// permanently rewriting authored data. Cycling any valid surface must move it
    /// to another valid surface and, after a full lap, return it unchanged.
    #[test]
    fn cycling_an_imported_surface_never_destroys_it() {
        let n = DECAL_SURFACES.len() as i32;
        for kind in WRITABLE_SURFACES {
            let once = cycle_decal_surface(kind, 1);
            assert_ne!(once, kind, "{kind:?} must step to a DIFFERENT surface");
            // A full lap forward is the identity — no kind is absorbed by another.
            let mut k = kind;
            for _ in 0..n {
                k = cycle_decal_surface(k, 1);
            }
            assert_eq!(
                k, kind,
                "a full forward lap from {kind:?} returns it unchanged"
            );
            // …and one step back undoes one step forward.
            assert_eq!(cycle_decal_surface(once, -1), kind, "{kind:?} steps back");
        }
    }

    /// Every cycle entry has a distinct, non-debug label — a `Hologram` decal must
    /// read `hologram` in the tree/inspector, not the `unknown`-shaped debug
    /// fallback that a 3-entry list produced for it.
    #[test]
    fn every_cycle_entry_has_its_parser_label() {
        for &(kind, label) in DECAL_SURFACES.iter() {
            assert_eq!(decal_surface_label(kind), label, "{kind:?} label");
        }
        // …and no writable surface falls through to the debug-name fallback.
        for kind in WRITABLE_SURFACES {
            let label = decal_surface_label(kind);
            let text = decal_lvl(&label);
            let level = re_level::load_string(&text, "surface-pin.lvl").unwrap_or_else(|e| {
                panic!("{kind:?} labelled '{label}', which the parser rejects: {e:?}")
            });
            assert_eq!(
                level.decals[0].surface_kind, kind,
                "the label shown for {kind:?} must be the one that reloads as {kind:?}"
            );
        }
    }
}
