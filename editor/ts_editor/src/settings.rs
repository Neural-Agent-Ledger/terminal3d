//! LEVEL / PUZZLE SETTINGS modal — a pure form over [`EditorState`] for editing
//! the level's META, ROOM, INPUT-node, and SEQUENCE fields without re-typing the
//! `.lvl` by hand. Mirrors `wizard.rs` / `saveas.rs`: NO terminal, NO I/O — every
//! function only reads/writes `EditorState` data, so the whole module is unit
//! testable with no terminal present.
//!
//! ## Field / focus scheme
//!
//! The form spans four sections (Meta / Room / Inputs / Sequence). The scalar
//! fields each get their own [`SettingsField`] variant; the two list sections and
//! the input list use a single "section" focus variant plus a separate *cursor*
//! index in [`SettingsState`] that selects the active row/name within that
//! section. The flat focus order is:
//!
//! The order lives in exactly ONE place — [`SettingsField::ORDER`] — which both
//! `next` and `prev` index into (it used to be two hand-written mirrored matches):
//!
//! ```text
//! Name → Title → Version → SpecRev                   ([meta])
//!      → Ambient → StoreyHeight → SizeW/H/D
//!      → PowerBudget → RoomFloor → RoomCeiling       ([room])
//!      → CycleKeep → RenderPipes
//!      → Snow → Steam → Matrix → Glitch              ([effects])
//!      → FogColor → FogNear → FogFar                 ([palette] distance-fog)
//!      → AtmosAmbient → KeyColor
//!      → FogHeightBase → FogHeightFalloff → Postfx   ([atmosphere] lighting mood)
//!      → SkyCylinder → SkyCapTop → SkyCapBottom      ([sky] image cubemap)
//!      → Textures → DoorTargetModel
//!      → Inputs → Observe → Steps → (wraps to Name)
//! ```
//!
//! ## Inert authoring metadata (`version` / `spec_rev` / `floor` / `ceiling` /
//! ## `target_model`)
//!
//! Five `.lvl` keys are parsed and re-emitted VERBATIM but read by nothing in the
//! renderer. They were previously unauthorable — preserved through a save but
//! impossible to create or change from the editor. All five are free TEXT here:
//!
//! * `[meta] version`, `[meta] spec_rev` — file/spec revision tags (`spec_rev` is
//!   read off disk by `re_game`'s canon guard).
//! * `[room] floor`, `[room] ceiling` — floor/ceiling STYLE tags. NOT material
//!   names: the surface's actual look comes from a `[material]` section carrying
//!   `surface = floor` / `surface = ceiling`, which is authored on the MATERIAL
//!   (`material::MaterialField::Surface`), not here. There is therefore no
//!   vocabulary to validate a value against — the parser accepts any string.
//! * `[door] target_model` — the model tag the door leads to. Per-LEVEL in
//!   practice: `re_level::Level` owns ONE `door: Door` field (not a `Vec`), so a
//!   `.lvl` carries exactly one `[door]` section and one `target_model`.
//!
//! `[room] power_budget` is NOT inert — it is the pipe-power cap the game HUD
//! shows and enforces. It is staged as a typed DIGIT buffer (see
//! [`SettingsState::power_budget`]) so a value like `12` can be typed, not
//! stepped twelve times.
//!
//! ## Atmosphere round-trip (effects / palette / sky)
//!
//! The three atmosphere sections are EMIT-ON-CHANGE so a level with no atmosphere
//! stays byte-identical after a no-edit save:
//! * `[effects]` — `snow`/`matrix`/`glitch` are `f32` (clamped `0..=1`), `steam` a
//!   bool. The serializer emits `[effects]` only when one is non-default
//!   (`snow>0` / `steam` / `matrix>0` / `glitch>0`), so an all-zero block is silent.
//! * `[palette]` — `fog_color` is an `#RRGGBB` hex TEXT buffer; an empty buffer
//!   leaves `palette.has_fog` false (no `fog_color` emitted). `fog_near`/`fog_far`
//!   set `palette.has_fog_dist` only when the user adjusts them. The writer gates
//!   `[palette]` on `has_fog || has_fog_dist`.
//! * `[sky]` — `cylinder`/`cap_top`/`cap_bottom` are `Option<String>` paths; an
//!   empty buffer ⇒ `None`. The writer emits `[sky]` only when at least one is set.
//! * `[atmosphere]` — the per-level LIGHTING mood: `ambient` (an `Option<f32>` in
//!   `[0,1]`, stepping below `0` clears it back to unset), `key_color` (an
//!   `#RRGGBB` hex TEXT buffer, empty ⇒ unset), `fog_height` (a PAIR staged as
//!   `fog_height_base`/`fog_height_falloff` + the `has_fog_height` authored flag,
//!   exactly like the fog-distance pair, so touching either half emits a COMPLETE
//!   `HeightFog`), and `postfx` (a cycle over [`POSTFX_CYCLE_LEN`] — index `0` is
//!   the unset `(default)` state, the rest are [`re_world::POSTFX_PRESETS`]).
//!   Every one is omit-when-unset, so an atmosphere-less level stays
//!   byte-identical. `[atmosphere] key_dir` (the key-light DIRECTION) is NOT
//!   authorable here — lights are v-next for the editor — but [`apply`] never
//!   touches `atmos.key_dir`, so a hand-authored one survives a load→save.
//!
//! * **Name / Title** — free text. `Char(c)` types, `Backspace` pops.
//! * **Ambient** — cycles [`AMBIENT_CYCLE`] with `-`/`+` (or Left/Right); the
//!   empty string is shown as `(default)`.
//! * **StoreyHeight** — `-`/`+` adjusts, clamped `>= 1`.
//! * **CycleKeep / RenderPipes** — `Space` (or `-`/`+`) toggles the bool.
//! * **Inputs** (section) — Up/Down move `input_cursor` over the per-input rows;
//!   `Space` toggles the highlighted input's initial-on bit.
//! * **Observe** (section) — Up/Down move `observe_cursor`; `a` adds the next
//!   candidate node name (cycled from the node-name snapshot), `x` removes the
//!   row under the cursor.
//! * **Steps** (section) — Up/Down move `step_cursor` over the step rows; `a`
//!   appends a new (empty) step, `x` removes the step under the cursor; `-`/`+`
//!   (or Left/Right) move `name_cursor` within the selected step; `n` adds the
//!   next candidate observe-name into the selected step, `r` removes the name
//!   under `name_cursor`.
//!
//! `Tab`/Down advances focus (`focus_next`), Up retreats (`focus_prev`). `Enter`
//! always confirms + applies the whole form (see [`apply`]); `Esc` discards (the
//! discard is handled by `main.rs::quit_on_esc`).
//!
//! ## Round-trip note (sequence patterns)
//!
//! `re_voximport::to_lvl_text` emits the `[sequence]` block ONLY from
//! `observe_names` + `step_names`; on reload `re_level` finalize REBUILDS
//! `patterns` from those authored names. So [`apply`] writes only the AUTHORED
//! fields for a clean save→load round-trip. To keep `re_puzzle` working *in the
//! same session* (without a reload), [`apply`] ALSO rebuilds `sequence.patterns`
//! in place: for each step it builds `bits = (observe_node_index, step_has_name)`
//! for every observe node, resolving names via `Level::node_index`. If ANY
//! observe/step name fails to resolve, `patterns` is left empty (a later load
//! re-resolves it from the authored names).
use crate::state::EditorState;

/// Minimum authorable grid dimension per axis (matches the new-level wizard clamp).
pub const MIN_DIM: usize = crate::wizard::MIN_DIM;
/// Maximum authorable grid dimension per axis (matches the new-level wizard clamp).
pub const MAX_DIM: usize = crate::wizard::MAX_DIM;

/// The ambient values that map to a wall kind in `re_world::ambient_to_wall_kind`:
/// `""`/`trace*` → WallTrace, `"matrix"` → WallMatrix, else → WallSolid. The empty
/// string is the level default; [`ambient_label`] renders it `(default)`.
pub const AMBIENT_CYCLE: [&str; 4] = ["", "trace", "matrix", "solid"];

/// Digit cap on the `power_budget` typed buffer — `u32::MAX` is 10 digits, so a
/// longer run could never parse and would only look like it had been accepted.
pub const POWER_BUDGET_MAX_DIGITS: usize = 10;

/// Step for `[atmosphere] ambient` (`-`/`+`). The parser clamps the value to
/// `[0,1]`, so 50 presses cover the whole range.
pub const ATMOS_AMBIENT_STEP: f32 = 0.02;

/// Step for the `[atmosphere] fog_height` BASE height (world units).
pub const FOG_HEIGHT_BASE_STEP: f32 = 0.5;

/// Step for the `[atmosphere] fog_height` FALLOFF (extinction per world unit).
pub const FOG_HEIGHT_FALLOFF_STEP: f32 = 0.1;

/// Length of the `postfx` cycle: the UNSET state plus every
/// [`re_world::POSTFX_PRESETS`] entry.
pub const POSTFX_CYCLE_LEN: usize = re_world::POSTFX_PRESETS.len() + 1;

/// The postfx preset at cycle index `idx`: `0` is the UNSET state (the empty
/// string, rendered `(default)` by [`ambient_label`]), `1..` index
/// [`re_world::POSTFX_PRESETS`]. Out-of-range indices read as unset rather than
/// panicking — [`SettingsState::postfx_cycle`] keeps the index in range.
#[must_use]
pub fn postfx_name(idx: usize) -> &'static str {
    if idx == 0 {
        ""
    } else {
        re_world::POSTFX_PRESETS.get(idx - 1).copied().unwrap_or("")
    }
}

/// Cycle index of an authored `postfx` name — `0` (unset) for `None` or an
/// unknown name, else its [`re_world::POSTFX_PRESETS`] position `+ 1`.
fn postfx_idx_for(name: Option<&str>) -> usize {
    let Some(n) = name else { return 0 };
    let n = n.trim().to_ascii_lowercase();
    re_world::POSTFX_PRESETS
        .iter()
        .position(|p| *p == n)
        .map_or(0, |i| i + 1)
}

/// Human-readable label for an [`AMBIENT_CYCLE`] entry — `(default)` for the empty
/// string, the value itself otherwise.
pub fn ambient_label(s: &str) -> &str {
    if s.is_empty() {
        "(default)"
    } else {
        s
    }
}

/// Which field the form's focus currently rests on. `Inputs`/`Observe`/`Steps`
/// are SECTION foci — the active row within them is tracked by the matching cursor
/// field on [`SettingsState`]. See the module doc for the full scheme.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsField {
    Name,
    Title,
    /// `[meta] version` — the authored file-format revision tag (free text, kept
    /// VERBATIM by `re_level`). Empty ⇒ the writer emits no `version` line.
    Version,
    /// `[meta] spec_rev` — the design-document revision this level was authored
    /// against, read by the `re_game` canon guard (free text). Empty ⇒ no line.
    SpecRev,
    Ambient,
    StoreyHeight,
    /// Live grid WIDTH (sx). `-`/`+` adjust the staged value, clamped
    /// `MIN_DIM..=MAX_DIM`; `[Resize]` (`R`) applies all three to the map.
    SizeW,
    /// Live grid HEIGHT (sy).
    SizeH,
    /// Live grid DEPTH (sz).
    SizeD,
    /// `[room] power_budget` — the per-level cap on simultaneously-ON pipes.
    /// A TYPED DIGIT buffer (`0-9` type, Backspace pops) that ALSO steps with
    /// `-`/`+`; an empty buffer means `0` ⇒ the writer emits no `power_budget`.
    PowerBudget,
    /// `[room] floor` — the authored floor-STYLE tag (free text, e.g. `grid`).
    /// INERT: the floor's look comes from `[material] surface = floor`, which is
    /// authored on the MATERIAL, not here. Kept verbatim so a save cannot drop it.
    RoomFloor,
    /// `[room] ceiling` — the authored ceiling-STYLE tag. Inert, see
    /// [`RoomFloor`](SettingsField::RoomFloor).
    RoomCeiling,
    CycleKeep,
    RenderPipes,
    // ── EFFECTS ([effects]) ──
    /// `[effects] snow` density (f32, step 0.1, clamped `0.0..=1.0`).
    Snow,
    /// `[effects] steam` (bool, Space toggles).
    Steam,
    /// `[effects] matrix` density (f32, step 0.1, clamped `0.0..=1.0`).
    Matrix,
    /// `[effects] glitch` density (f32, step 0.1, clamped `0.0..=1.0`).
    Glitch,
    // ── FOG ([palette]) ──
    /// `[palette] fog_color` (`#RRGGBB` hex text buffer).
    FogColor,
    /// `[palette] fog_near` (f32, step 0.5, clamped `>= 0`).
    FogNear,
    /// `[palette] fog_far` (f32, step 0.5, clamped `>= 0`).
    FogFar,
    // ── ATMOSPHERE ([atmosphere]) ──
    /// `[atmosphere] ambient` — the ambient light floor, an OPTIONAL f32 stepped
    /// by [`ATMOS_AMBIENT_STEP`] and clamped `0.0..=1.0` (the parser's range).
    /// Stepping below `0` clears it back to unset ⇒ no `ambient` line emitted.
    /// Distinct from [`Ambient`](SettingsField::Ambient), which is the `[room]`
    /// wall-KIND tag.
    AtmosAmbient,
    /// `[atmosphere] key_color` — the key light's tint (`#RRGGBB` hex text
    /// buffer, exactly like [`FogColor`](SettingsField::FogColor)). Empty ⇒ unset.
    KeyColor,
    /// `[atmosphere] fog_height` BASE height (f32, step
    /// [`FOG_HEIGHT_BASE_STEP`]). Touching it authors the whole PAIR.
    FogHeightBase,
    /// `[atmosphere] fog_height` FALLOFF (f32, step
    /// [`FOG_HEIGHT_FALLOFF_STEP`], clamped `>= 0` — the parser's constraint).
    /// Touching it authors the whole PAIR.
    FogHeightFalloff,
    /// `[atmosphere] postfx` — cycles the UNSET state plus every
    /// [`re_world::POSTFX_PRESETS`] entry (see [`postfx_name`]).
    Postfx,
    // ── SKY ([sky]) ──
    /// `[sky] cylinder` image path (free text).
    SkyCylinder,
    /// `[sky] cap_top` image path (free text).
    SkyCapTop,
    /// `[sky] cap_bottom` image path (free text).
    SkyCapBottom,
    /// `[texture NAME]` list section. Up/Down move `texture_cursor`; `Left`/`Right`
    /// move `texture_col` (0 = name, 1 = file); printable chars type into the focused
    /// row's focused column; `a` adds a row, `x` removes the row under the cursor.
    Textures,
    /// `[door] target_model` — the authored tag naming the model the door leads
    /// to (free text, INERT). Per-LEVEL in practice: `re_level::Level` owns ONE
    /// `door: Door` (not a `Vec`), so a `.lvl` has exactly one `[door]` section
    /// and exactly one `target_model`. See the module doc for why it lives here
    /// rather than on the door's inspector rows.
    DoorTargetModel,
    Inputs,
    Observe,
    Steps,
}

impl SettingsField {
    /// THE flat focus order — the SINGLE table both [`next`](SettingsField::next)
    /// and [`prev`](SettingsField::prev) index into.
    ///
    /// This used to be two hand-written matches (forwards and backwards). Adding
    /// a field meant editing both, and nothing caught a half-edit; the sibling
    /// SSOT tables in this crate (`KEYMAP`, `inspector_rows`, `settings_rows`)
    /// all avoid exactly that shape. `order_covers_every_variant` enumerates the
    /// variants explicitly and fails if a new one is missing here.
    pub const ORDER: &'static [SettingsField] = {
        use SettingsField as F;
        &[
            F::Name,
            F::Title,
            F::Version,
            F::SpecRev,
            F::Ambient,
            F::StoreyHeight,
            F::SizeW,
            F::SizeH,
            F::SizeD,
            F::PowerBudget,
            F::RoomFloor,
            F::RoomCeiling,
            F::CycleKeep,
            F::RenderPipes,
            F::Snow,
            F::Steam,
            F::Matrix,
            F::Glitch,
            F::FogColor,
            F::FogNear,
            F::FogFar,
            F::AtmosAmbient,
            F::KeyColor,
            F::FogHeightBase,
            F::FogHeightFalloff,
            F::Postfx,
            F::SkyCylinder,
            F::SkyCapTop,
            F::SkyCapBottom,
            F::Textures,
            F::DoorTargetModel,
            F::Inputs,
            F::Observe,
            F::Steps,
        ]
    };

    /// This field's index in [`ORDER`](SettingsField::ORDER). A variant missing
    /// from the table would be unreachable by Tab, so it defaults to `0` rather
    /// than panicking — `order_covers_every_variant` is what makes that
    /// impossible in the first place.
    fn order_index(self) -> usize {
        Self::ORDER.iter().position(|f| *f == self).unwrap_or(0)
    }

    /// Advance focus one step along [`ORDER`](SettingsField::ORDER), wrapping.
    #[must_use]
    pub fn next(self) -> SettingsField {
        let n = Self::ORDER.len();
        Self::ORDER[(self.order_index() + 1) % n]
    }

    /// Retreat focus (the inverse of [`next`](SettingsField::next)).
    #[must_use]
    pub fn prev(self) -> SettingsField {
        let n = Self::ORDER.len();
        Self::ORDER[(self.order_index() + n - 1) % n]
    }
}

/// Live state of the open LEVEL SETTINGS form, seeded from the level by [`open`]
/// and written back by [`apply`]. Pure data — no terminal, no I/O.
#[derive(Clone, Debug)]
pub struct SettingsState {
    /// `[meta] name` buffer (free text).
    pub name: String,
    /// `[meta] title` buffer (free text).
    pub title: String,
    /// `[meta] version` buffer (free text). Empty ⇒ no `version` line emitted.
    pub version: String,
    /// `[meta] spec_rev` buffer (free text). Empty ⇒ no `spec_rev` line emitted.
    pub spec_rev: String,
    /// `[room] power_budget` as a TYPED DIGIT buffer. Held as text (not a `u32`)
    /// so the field accepts direct numeric entry — pressing `+` N times is not an
    /// authoring method for a value like 12. Read back through
    /// [`power_budget_value`](SettingsState::power_budget_value); an empty or
    /// unparseable buffer reads `0`, and the writer emits `power_budget` only
    /// when it is `> 0`, so an untouched level stays byte-identical.
    pub power_budget: String,
    /// `[room] floor` style tag buffer (free text, INERT — see
    /// [`SettingsField::RoomFloor`]).
    pub room_floor: String,
    /// `[room] ceiling` style tag buffer (free text, INERT).
    pub room_ceiling: String,
    /// `[door] target_model` buffer (free text, INERT).
    pub door_target_model: String,
    /// Index into [`AMBIENT_CYCLE`] for the room ambient.
    pub ambient_idx: usize,
    /// `[room] storey_height` (clamped `>= 1`).
    pub storey_height: u32,
    /// Staged grid WIDTH (sx). Seeded from the live `level.size` on [`open`];
    /// `[Resize]` applies `(size_w, size_h, size_d)` to the map. Clamped
    /// `MIN_DIM..=MAX_DIM`.
    pub size_w: usize,
    /// Staged grid HEIGHT (sy). See [`size_w`](Self::size_w).
    pub size_h: usize,
    /// Staged grid DEPTH (sz). See [`size_w`](Self::size_w).
    pub size_d: usize,
    /// `[room] cycle_keep`.
    pub cycle_keep: bool,
    /// `[room] render_pipes`.
    pub render_pipes: bool,
    /// `[effects] snow` density (`0.0..=1.0`). Default `0.0` ⇒ no `[effects] snow`.
    pub snow: f32,
    /// `[effects] steam`.
    pub steam: bool,
    /// `[effects] matrix` density (`0.0..=1.0`).
    pub matrix: f32,
    /// `[effects] glitch` density (`0.0..=1.0`).
    pub glitch: f32,
    /// `[palette] fog_color` as an `#RRGGBB` hex text buffer. EMPTY means "no fog
    /// tint" — [`apply`] leaves `palette.has_fog` unset so a fog-less level stays
    /// byte-identical. A non-empty buffer that PARSES as a hex colour enables fog.
    pub fog_color: String,
    /// `[palette] fog_near` distance.
    pub fog_near: f32,
    /// `[palette] fog_far` distance.
    pub fog_far: f32,
    /// Whether the fog-distance (`fog_near`/`fog_far`) is AUTHORED. Seeded from the
    /// level's `palette.has_fog_dist`; set true when the user adjusts near/far. Keeps
    /// a level with no `fog_near`/`fog_far` byte-identical until the user edits them.
    pub has_fog_dist: bool,
    /// `[atmosphere] ambient` in `[0,1]`. `None` = UNSET ⇒ no `ambient` line, which
    /// is what an atmosphere-less level must stay. Reachable both ways: stepping
    /// down past `0` clears it, stepping up from unset sets it.
    pub atmos_ambient: Option<f32>,
    /// `[atmosphere] key_color` as an `#RRGGBB` hex text buffer. EMPTY (or
    /// unparseable) means UNSET ⇒ no `key_color` line, mirroring
    /// [`fog_color`](Self::fog_color).
    pub key_color: String,
    /// `[atmosphere] fog_height` BASE height (meaningful only with
    /// [`has_fog_height`](Self::has_fog_height)). May be negative — the parser
    /// only requires it finite.
    pub fog_height_base: f32,
    /// `[atmosphere] fog_height` FALLOFF, clamped `>= 0` (the parser rejects a
    /// negative one).
    pub fog_height_falloff: f32,
    /// Whether `fog_height` is AUTHORED. Seeded from `atmos.fog_height.is_some()`
    /// and set by adjusting EITHER half, so a half-set pair is impossible; unset
    /// ⇒ `atmos.fog_height` stays `None`. Same shape as
    /// [`has_fog_dist`](Self::has_fog_dist).
    pub has_fog_height: bool,
    /// Cycle index of `[atmosphere] postfx`: `0` = unset `(default)`, else the
    /// entry at `idx - 1` in [`re_world::POSTFX_PRESETS`] (see [`postfx_name`]).
    pub postfx_idx: usize,
    /// `[sky] cylinder` image path (empty ⇒ `None`, no `[sky] cylinder`).
    pub sky_cylinder: String,
    /// `[sky] cap_top` image path (empty ⇒ `None`).
    pub sky_cap_top: String,
    /// `[sky] cap_bottom` image path (empty ⇒ `None`).
    pub sky_cap_bottom: String,
    /// One entry per `[texture NAME]`: `(name, file)` text buffers. Seeded from
    /// `level.textures` on [`open`] and written back verbatim on [`apply`]. An entry
    /// with a blank name OR blank file is DROPPED on apply (so an empty added row
    /// never emits a malformed `[texture]` section, keeping byte-identity).
    pub textures: Vec<(String, String)>,
    /// Which texture row the Textures section's cursor sits on.
    pub texture_cursor: usize,
    /// Which COLUMN of the selected texture row typing edits: `0` = name, `1` = file.
    pub texture_col: usize,
    /// One entry per INPUT node: `(node_name, initial_on)`.
    pub inputs: Vec<(String, bool)>,
    /// `[sequence] observe = ...` node names, in order.
    pub observe: Vec<String>,
    /// `[sequence] step = ...` rows — each a list of observe-names ON at that step.
    pub steps: Vec<Vec<String>>,
    /// Which input row the Inputs section's cursor sits on.
    pub input_cursor: usize,
    /// Which observe row the Observe section's cursor sits on.
    pub observe_cursor: usize,
    /// Which step row the Steps section's cursor sits on.
    pub step_cursor: usize,
    /// Which name within the selected step the `-`/`+` cursor sits on.
    pub name_cursor: usize,
    /// Snapshot of ALL node names at open time — the candidate pool that adding an
    /// observe name cycles through (so the form needn't borrow `state.level`).
    pub node_names: Vec<String>,
    /// Snapshot of the OUTPUT node names at open time. `add_observe` prefers these
    /// (the observe window watches outputs); it falls back to `node_names` only when
    /// every output is already observed (or there are none).
    pub output_names: Vec<String>,
    /// The focused field/section.
    pub focus: SettingsField,
}

impl SettingsState {
    // ── text fields ──────────────────────────────────────────────────────────

    /// Append a character to the name buffer.
    pub fn name_push(&mut self, c: char) {
        self.name.push(c);
    }
    /// Pop the last character of the name buffer (no-op when empty).
    pub fn name_pop(&mut self) {
        self.name.pop();
    }
    /// Append a character to the title buffer.
    pub fn title_push(&mut self, c: char) {
        self.title.push(c);
    }
    /// Pop the last character of the title buffer (no-op when empty).
    pub fn title_pop(&mut self) {
        self.title.pop();
    }

    /// The mutable buffer behind one of the FREE-TEXT metadata fields
    /// (`version` / `spec_rev` / `[room] floor` / `[room] ceiling` /
    /// `[door] target_model`), or `None` for any other field. ONE routing table
    /// shared by [`text_push`](SettingsState::text_push),
    /// [`text_pop`](SettingsState::text_pop) and
    /// [`text_of`](SettingsState::text_of) so the three can never disagree about
    /// which field owns which buffer.
    fn text_slot(&mut self, field: SettingsField) -> Option<&mut String> {
        match field {
            SettingsField::Version => Some(&mut self.version),
            SettingsField::SpecRev => Some(&mut self.spec_rev),
            SettingsField::RoomFloor => Some(&mut self.room_floor),
            SettingsField::RoomCeiling => Some(&mut self.room_ceiling),
            SettingsField::DoorTargetModel => Some(&mut self.door_target_model),
            _ => None,
        }
    }

    /// True when `field` is one of the free-text metadata fields routed by
    /// [`text_slot`](SettingsState::text_slot) — the predicate the key dispatch
    /// uses to decide that a printable char TYPES rather than steps a value.
    #[must_use]
    pub fn is_text_field(field: SettingsField) -> bool {
        matches!(
            field,
            SettingsField::Version
                | SettingsField::SpecRev
                | SettingsField::RoomFloor
                | SettingsField::RoomCeiling
                | SettingsField::DoorTargetModel
        )
    }

    /// Read one of the free-text metadata buffers (empty string for a non-text
    /// field) — the painter's accessor.
    #[must_use]
    pub fn text_of(&self, field: SettingsField) -> &str {
        match field {
            SettingsField::Version => &self.version,
            SettingsField::SpecRev => &self.spec_rev,
            SettingsField::RoomFloor => &self.room_floor,
            SettingsField::RoomCeiling => &self.room_ceiling,
            SettingsField::DoorTargetModel => &self.door_target_model,
            _ => "",
        }
    }

    /// Append a character to the focused free-text metadata buffer (no-op when
    /// `field` is not one).
    pub fn text_push(&mut self, field: SettingsField, c: char) {
        if let Some(buf) = self.text_slot(field) {
            buf.push(c);
        }
    }

    /// Pop the last character of the focused free-text metadata buffer (no-op
    /// when empty / not a text field).
    pub fn text_pop(&mut self, field: SettingsField) {
        if let Some(buf) = self.text_slot(field) {
            buf.pop();
        }
    }

    // ── power budget ([room] power_budget) ──────────────────────────────────────

    /// The staged `power_budget` as the `u32` [`apply`] will write. An empty or
    /// unparseable buffer reads `0` (⇒ the writer emits nothing), and an
    /// over-long digit run saturates rather than wrapping.
    #[must_use]
    pub fn power_budget_value(&self) -> u32 {
        let t = self.power_budget.trim();
        if t.is_empty() {
            return 0;
        }
        t.parse::<u32>().unwrap_or(u32::MAX)
    }

    /// Type one DIGIT into the power-budget buffer. Non-digits are rejected (the
    /// field is an integer, so a stray `q` must not become part of it) and the
    /// buffer is capped at [`POWER_BUDGET_MAX_DIGITS`] so it can never grow past
    /// what a `u32` can hold.
    pub fn power_budget_push(&mut self, c: char) {
        if c.is_ascii_digit() && self.power_budget.len() < POWER_BUDGET_MAX_DIGITS {
            self.power_budget.push(c);
        }
    }

    /// Pop the last digit of the power-budget buffer (no-op when empty).
    pub fn power_budget_pop(&mut self) {
        self.power_budget.pop();
    }

    /// Step the power budget by `delta`, clamped `>= 0` (the parser's own
    /// constraint) and re-rendered into the digit buffer. Stepping DOWN to 0 from
    /// a level that had no `power_budget` leaves the buffer showing `0`, which
    /// still emits nothing — `0` and "absent" are the same state in the writer.
    pub fn power_budget_adjust(&mut self, delta: i32) {
        let cur = i64::from(self.power_budget_value());
        let next = (cur + i64::from(delta)).clamp(0, i64::from(u32::MAX));
        self.power_budget = next.to_string();
    }

    // ── ambient / storey / bools ───────────────────────────────────────────────

    /// Cycle the ambient by `delta` (wraps over [`AMBIENT_CYCLE`]).
    pub fn ambient_cycle(&mut self, delta: i32) {
        let n = AMBIENT_CYCLE.len() as i32;
        self.ambient_idx = (self.ambient_idx as i32 + delta).rem_euclid(n) as usize;
    }
    /// The currently-selected ambient string.
    pub fn ambient(&self) -> &'static str {
        AMBIENT_CYCLE[self.ambient_idx]
    }
    /// Adjust the storey height by `delta`, clamped to `>= 1`.
    pub fn storey_adjust(&mut self, delta: i32) {
        self.storey_height = (self.storey_height as i32 + delta).max(1) as u32;
    }
    /// Adjust the staged grid dimension for `field` (`SizeW`/`SizeH`/`SizeD`) by
    /// `delta`, clamped to `MIN_DIM..=MAX_DIM`. A non-size field is a no-op.
    pub fn size_adjust(&mut self, field: SettingsField, delta: i32) {
        let slot = match field {
            SettingsField::SizeW => &mut self.size_w,
            SettingsField::SizeH => &mut self.size_h,
            SettingsField::SizeD => &mut self.size_d,
            _ => return,
        };
        *slot = (*slot as i32 + delta).clamp(MIN_DIM as i32, MAX_DIM as i32) as usize;
    }
    /// Toggle `cycle_keep`.
    pub fn toggle_cycle_keep(&mut self) {
        self.cycle_keep = !self.cycle_keep;
    }
    /// Toggle `render_pipes`.
    pub fn toggle_render_pipes(&mut self) {
        self.render_pipes = !self.render_pipes;
    }

    // ── effects ([effects]) ─────────────────────────────────────────────────────

    /// Adjust the `snow` density by `delta * 0.1`, clamped `0.0..=1.0`.
    pub fn snow_adjust(&mut self, delta: i32) {
        self.snow = (self.snow + delta as f32 * 0.1).clamp(0.0, 1.0);
    }
    /// Toggle `steam`.
    pub fn toggle_steam(&mut self) {
        self.steam = !self.steam;
    }
    /// Adjust the `matrix` density by `delta * 0.1`, clamped `0.0..=1.0`.
    pub fn matrix_adjust(&mut self, delta: i32) {
        self.matrix = (self.matrix + delta as f32 * 0.1).clamp(0.0, 1.0);
    }
    /// Adjust the `glitch` density by `delta * 0.1`, clamped `0.0..=1.0`.
    pub fn glitch_adjust(&mut self, delta: i32) {
        self.glitch = (self.glitch + delta as f32 * 0.1).clamp(0.0, 1.0);
    }

    // ── fog ([palette]) ─────────────────────────────────────────────────────────

    /// Append a character to the fog-colour hex buffer.
    pub fn fog_color_push(&mut self, c: char) {
        self.fog_color.push(c);
    }
    /// Pop the last character of the fog-colour hex buffer (no-op when empty).
    pub fn fog_color_pop(&mut self) {
        self.fog_color.pop();
    }
    /// Adjust `fog_near` by `delta * 0.5`, clamped `>= 0`; marks the fog distance
    /// as AUTHORED (so it is emitted).
    pub fn fog_near_adjust(&mut self, delta: i32) {
        self.fog_near = (self.fog_near + delta as f32 * 0.5).max(0.0);
        self.has_fog_dist = true;
    }
    /// Adjust `fog_far` by `delta * 0.5`, clamped `>= 0`; marks the fog distance as
    /// AUTHORED.
    pub fn fog_far_adjust(&mut self, delta: i32) {
        self.fog_far = (self.fog_far + delta as f32 * 0.5).max(0.0);
        self.has_fog_dist = true;
    }

    // ── atmosphere ([atmosphere]) ───────────────────────────────────────────────

    /// Step `[atmosphere] ambient` by `delta` x [`ATMOS_AMBIENT_STEP`], clamped to
    /// the parser's `[0,1]`. Stepping DOWN past `0` clears the field back to UNSET
    /// (`None` ⇒ no line emitted), which is the only way back to "no ambient
    /// authored"; stepping UP from unset starts at one step above `0`.
    pub fn atmos_ambient_adjust(&mut self, delta: i32) {
        let next = self.atmos_ambient.unwrap_or(0.0) + delta as f32 * ATMOS_AMBIENT_STEP;
        self.atmos_ambient = if next < 0.0 {
            None
        } else {
            Some(next.clamp(0.0, 1.0))
        };
    }
    /// Append a character to the key-colour hex buffer.
    pub fn key_color_push(&mut self, c: char) {
        self.key_color.push(c);
    }
    /// Pop the last character of the key-colour hex buffer (no-op when empty).
    pub fn key_color_pop(&mut self) {
        self.key_color.pop();
    }
    /// Adjust the height-fog BASE by `delta` x [`FOG_HEIGHT_BASE_STEP`] and mark the
    /// PAIR as authored (so a complete `HeightFog` is emitted, never a half of one).
    pub fn fog_height_base_adjust(&mut self, delta: i32) {
        self.fog_height_base += delta as f32 * FOG_HEIGHT_BASE_STEP;
        self.has_fog_height = true;
    }
    /// Adjust the height-fog FALLOFF by `delta` x [`FOG_HEIGHT_FALLOFF_STEP`],
    /// clamped `>= 0` (the parser rejects a negative falloff), and mark the PAIR as
    /// authored.
    pub fn fog_height_falloff_adjust(&mut self, delta: i32) {
        self.fog_height_falloff =
            (self.fog_height_falloff + delta as f32 * FOG_HEIGHT_FALLOFF_STEP).max(0.0);
        self.has_fog_height = true;
    }
    /// Cycle `[atmosphere] postfx` by `delta` over the unset state plus every
    /// [`re_world::POSTFX_PRESETS`] entry (wraps).
    pub fn postfx_cycle(&mut self, delta: i32) {
        let n = POSTFX_CYCLE_LEN as i32;
        self.postfx_idx = (self.postfx_idx as i32 + delta).rem_euclid(n) as usize;
    }
    /// The currently-selected postfx preset (`""` when unset).
    #[must_use]
    pub fn postfx(&self) -> &'static str {
        postfx_name(self.postfx_idx)
    }

    // ── sky ([sky]) ─────────────────────────────────────────────────────────────

    /// Append a character to the focused sky-path buffer (`field` selects which).
    /// A non-sky field is a no-op.
    pub fn sky_push(&mut self, field: SettingsField, c: char) {
        match field {
            SettingsField::SkyCylinder => self.sky_cylinder.push(c),
            SettingsField::SkyCapTop => self.sky_cap_top.push(c),
            SettingsField::SkyCapBottom => self.sky_cap_bottom.push(c),
            _ => {}
        }
    }
    /// Pop the last character of the focused sky-path buffer (no-op when empty / not
    /// a sky field).
    pub fn sky_pop(&mut self, field: SettingsField) {
        match field {
            SettingsField::SkyCylinder => self.sky_cylinder.pop(),
            SettingsField::SkyCapTop => self.sky_cap_top.pop(),
            SettingsField::SkyCapBottom => self.sky_cap_bottom.pop(),
            _ => None,
        };
    }

    // ── textures ([texture NAME]) ────────────────────────────────────────────────

    /// Move the texture-row cursor by `delta` (clamped to the texture list); resets
    /// the in-row column cursor to the NAME column.
    pub fn move_texture_cursor(&mut self, delta: i32) {
        self.texture_cursor = clamp_cursor(self.texture_cursor, delta, self.textures.len());
        self.texture_col = 0;
    }
    /// Move the in-row column cursor between NAME (0) and FILE (1), clamped.
    pub fn move_texture_col(&mut self, delta: i32) {
        self.texture_col = (self.texture_col as i32 + delta).clamp(0, 1) as usize;
    }
    /// Append a new empty texture row and select it (cursor on the NAME column).
    pub fn add_texture(&mut self) {
        self.textures.push((String::new(), String::new()));
        self.texture_cursor = self.textures.len() - 1;
        self.texture_col = 0;
    }
    /// Remove the texture row under the cursor (no-op if empty).
    pub fn remove_texture(&mut self) {
        if self.texture_cursor < self.textures.len() {
            self.textures.remove(self.texture_cursor);
            self.texture_cursor = self
                .texture_cursor
                .min(self.textures.len().saturating_sub(1));
            self.texture_col = 0;
        }
    }
    /// Append a character to the focused texture row's focused column (name/file).
    /// No-op if there is no selected row.
    pub fn texture_push(&mut self, c: char) {
        if let Some(row) = self.textures.get_mut(self.texture_cursor) {
            if self.texture_col == 0 {
                row.0.push(c);
            } else {
                row.1.push(c);
            }
        }
    }
    /// Pop the last character of the focused texture row's focused column (no-op when
    /// empty / no selected row).
    pub fn texture_pop(&mut self) {
        if let Some(row) = self.textures.get_mut(self.texture_cursor) {
            if self.texture_col == 0 {
                row.0.pop();
            } else {
                row.1.pop();
            }
        }
    }

    // ── inputs ─────────────────────────────────────────────────────────────────

    /// Move the input cursor by `delta` (clamped to the input list).
    pub fn move_input_cursor(&mut self, delta: i32) {
        self.input_cursor = clamp_cursor(self.input_cursor, delta, self.inputs.len());
    }
    /// Toggle the initial-on bit of the input under the cursor (no-op if empty).
    pub fn toggle_input(&mut self) {
        if let Some(e) = self.inputs.get_mut(self.input_cursor) {
            e.1 = !e.1;
        }
    }

    // ── observe list ───────────────────────────────────────────────────────────

    /// Move the observe cursor by `delta` (clamped to the observe list).
    pub fn move_observe_cursor(&mut self, delta: i32) {
        self.observe_cursor = clamp_cursor(self.observe_cursor, delta, self.observe.len());
    }
    /// Add the next sensible candidate to the observe window. The observe window
    /// watches OUTPUT nodes, so this PREFERS the first un-observed
    /// [`output_names`](Self::output_names) entry; only when every output is already
    /// observed (or there are none) does it fall back to any un-observed
    /// [`node_names`](Self::node_names) entry. No-op if every node is already
    /// observed (or there are none). The new row becomes the cursor.
    pub fn add_observe(&mut self) {
        let pick = self
            .output_names
            .iter()
            .find(|n| !self.observe.contains(n))
            .or_else(|| self.node_names.iter().find(|n| !self.observe.contains(n)))
            .cloned();
        if let Some(name) = pick {
            self.observe.push(name);
            self.observe_cursor = self.observe.len() - 1;
        }
    }
    /// Remove the observe row under the cursor (no-op if empty). Also drops that
    /// name from every step so steps stay a subset of `observe`.
    pub fn remove_observe(&mut self) {
        if self.observe_cursor < self.observe.len() {
            let removed = self.observe.remove(self.observe_cursor);
            for step in &mut self.steps {
                step.retain(|n| n != &removed);
            }
            self.observe_cursor = self
                .observe_cursor
                .min(self.observe.len().saturating_sub(1));
        }
    }

    // ── steps list ─────────────────────────────────────────────────────────────

    /// Move the step cursor by `delta` (clamped to the step list); resets the
    /// in-step name cursor.
    pub fn move_step_cursor(&mut self, delta: i32) {
        self.step_cursor = clamp_cursor(self.step_cursor, delta, self.steps.len());
        self.name_cursor = 0;
    }
    /// Append a new empty step (the "all observed OFF" step) and select it.
    pub fn add_step(&mut self) {
        self.steps.push(Vec::new());
        self.step_cursor = self.steps.len() - 1;
        self.name_cursor = 0;
    }
    /// Remove the step under the cursor (no-op if empty).
    pub fn remove_step(&mut self) {
        if self.step_cursor < self.steps.len() {
            self.steps.remove(self.step_cursor);
            self.step_cursor = self.step_cursor.min(self.steps.len().saturating_sub(1));
            self.name_cursor = 0;
        }
    }
    /// Move the in-step name cursor by `delta` (clamped to the selected step).
    pub fn move_name_cursor(&mut self, delta: i32) {
        let len = self
            .steps
            .get(self.step_cursor)
            .map(|s| s.len())
            .unwrap_or(0);
        self.name_cursor = clamp_cursor(self.name_cursor, delta, len);
    }
    /// Add the next candidate observe-name into the SELECTED step (the first
    /// observe name not already in that step). No-op if there is no selected step
    /// or every observe name is already present. The added name becomes the
    /// in-step cursor.
    pub fn add_step_name(&mut self) {
        let observe = self.observe.clone();
        if let Some(step) = self.steps.get_mut(self.step_cursor) {
            if let Some(name) = observe.iter().find(|n| !step.contains(n)).cloned() {
                step.push(name);
                self.name_cursor = step.len() - 1;
            }
        }
    }
    /// Remove the name under `name_cursor` from the selected step (no-op if empty).
    pub fn remove_step_name(&mut self) {
        if let Some(step) = self.steps.get_mut(self.step_cursor) {
            if self.name_cursor < step.len() {
                step.remove(self.name_cursor);
                self.name_cursor = self.name_cursor.min(step.len().saturating_sub(1));
            }
        }
    }

    // ── focus ──────────────────────────────────────────────────────────────────

    /// Advance focus to the next field.
    pub fn focus_next(&mut self) {
        self.focus = self.focus.next();
    }
    /// Retreat focus to the previous field.
    pub fn focus_prev(&mut self) {
        self.focus = self.focus.prev();
    }
}

/// Clamp-step a cursor `cur` by `delta` over a list of `len` items. Returns `0`
/// for an empty list; otherwise stays in `0..len`.
fn clamp_cursor(cur: usize, delta: i32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let max = (len - 1) as i32;
    (cur as i32 + delta).clamp(0, max) as usize
}

/// Map a level ambient string to its [`AMBIENT_CYCLE`] index, mirroring
/// `re_world::ambient_to_wall_kind`'s grouping: `""` → default index, `"matrix"`
/// → matrix, `trace*` → trace, anything else → solid.
fn ambient_idx_for(ambient: &str) -> usize {
    let a = ambient.trim();
    if a.is_empty() {
        0 // ""  → "(default)"
    } else if a == "matrix" {
        2
    } else if a.starts_with("trace") {
        1
    } else {
        3 // solid
    }
}

/// Parse a `#RRGGBB` / `0xRRGGBB` / bare `RRGGBB` fog-colour hex buffer into a
/// packed `0x00RRGGBB`, mirroring `re_level`'s `parse_hex_color` (the inverse of
/// the writer's `#{:06X}`). Returns `None` for an empty or malformed buffer so a
/// blank field never enables fog. Trailing chars beyond the first 6 hex digits are
/// ignored (matching the parser).
fn parse_hex_color_buf(buf: &str) -> Option<u32> {
    let t = buf.trim();
    if t.is_empty() {
        return None;
    }
    let hex = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .or_else(|| t.strip_prefix('#'))
        .unwrap_or(t);
    if hex.len() < 6 || !hex.chars().take(6).all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(&hex[..6], 16).ok()
}

/// `Some(trimmed.to_string())` for a non-empty (after-trim) buffer, else `None` —
/// the editor's empty-path ⇒ `None` rule for the optional `[sky]` image paths.
fn non_empty(buf: &str) -> Option<String> {
    let t = buf.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// Open the LEVEL SETTINGS modal, seeding [`SettingsState`] from `state.level`.
/// Pure: only reads the level and sets `state.settings` + the status line.
pub fn open(state: &mut EditorState) {
    let level = &state.level;
    // Inputs: one row per INPUT node; initial-on = the level has a nonzero
    // `[inputs]` entry for that node name.
    let inputs: Vec<(String, bool)> = level
        .nodes
        .iter()
        .filter(|n| n.is_input)
        .map(|n| {
            let on = level
                .inputs
                .iter()
                .any(|(name, v)| name == &n.name && *v != 0);
            (n.name.clone(), on)
        })
        .collect();
    let node_names: Vec<String> = level.nodes.iter().map(|n| n.name.clone()).collect();
    let output_names: Vec<String> = level
        .nodes
        .iter()
        .filter(|n| n.is_output)
        .map(|n| n.name.clone())
        .collect();
    let (sx, sy, sz) = level.size;

    // Seed the fog-colour hex buffer ONLY when the level actually carries a fog
    // tint, so a fog-less level shows an empty buffer and stays byte-identical on a
    // no-edit apply (the writer gates `[palette]` on `has_fog`/`has_fog_dist`).
    let fog_color = if level.palette.has_fog {
        format!("#{:06X}", level.palette.fog_color & 0xFF_FFFF)
    } else {
        String::new()
    };
    // `power_budget` is seeded as a DIGIT BUFFER. A level with no budget (raw 0)
    // seeds an EMPTY buffer, not "0", so an untouched apply writes 0 back and the
    // writer still emits nothing — byte-identity preserved either way.
    // `[atmosphere] key_color` seeds its hex buffer only when authored, so an
    // atmosphere-less level shows `(none)` and a no-edit apply writes `None` back.
    let key_color = level
        .atmos
        .key_color
        .map_or_else(String::new, |c| format!("#{:06X}", c & 0xFF_FFFF));
    let power_budget = if level.power_budget > 0 {
        level.power_budget.to_string()
    } else {
        String::new()
    };
    let st = SettingsState {
        name: level.name.clone(),
        title: level.title.clone(),
        version: level.version.clone(),
        spec_rev: level.spec_rev.clone(),
        power_budget,
        room_floor: level.room_floor.clone(),
        room_ceiling: level.room_ceiling.clone(),
        door_target_model: level.door.target_model.clone(),
        ambient_idx: ambient_idx_for(&level.ambient),
        storey_height: level.storey_height(), // accessor maps raw 0 → 4
        size_w: sx.clamp(MIN_DIM, MAX_DIM),
        size_h: sy.clamp(MIN_DIM, MAX_DIM),
        size_d: sz.clamp(MIN_DIM, MAX_DIM),
        cycle_keep: level.cycle_keep,
        render_pipes: level.render_pipes,
        snow: level.effects.snow,
        steam: level.effects.steam,
        matrix: level.effects.matrix,
        glitch: level.effects.glitch,
        fog_color,
        fog_near: level.palette.fog_near,
        fog_far: level.palette.fog_far,
        has_fog_dist: level.palette.has_fog_dist,
        atmos_ambient: level.atmos.ambient,
        key_color,
        fog_height_base: level.atmos.fog_height.map_or(0.0, |h| h.base),
        fog_height_falloff: level.atmos.fog_height.map_or(0.0, |h| h.falloff),
        has_fog_height: level.atmos.fog_height.is_some(),
        postfx_idx: postfx_idx_for(level.atmos.postfx.as_deref()),
        sky_cylinder: level.sky.cylinder.clone().unwrap_or_default(),
        sky_cap_top: level.sky.cap_top.clone().unwrap_or_default(),
        sky_cap_bottom: level.sky.cap_bottom.clone().unwrap_or_default(),
        textures: level
            .textures
            .iter()
            .map(|t| (t.name.clone(), t.file.clone()))
            .collect(),
        texture_cursor: 0,
        texture_col: 0,
        inputs,
        observe: level.sequence.observe_names.clone(),
        steps: level.sequence.step_names.clone(),
        input_cursor: 0,
        observe_cursor: 0,
        step_cursor: 0,
        name_cursor: 0,
        node_names,
        output_names,
        focus: SettingsField::Name,
    };
    state.settings = Some(st);
    state.status =
        "level settings: Tab/Up-Dn focus  -/+ edit (W/H/D = live size)  R resize  Enter apply  Esc cancel"
            .into();
}

/// RESIZE the live level to the form's staged `(size_w, size_h, size_d)` as ONE
/// undoable edit, WITHOUT closing the modal (resize is a structural remap distinct
/// from the meta/room [`apply`]). Captures an undo snapshot, calls
/// [`crate::resize::resize_level`] (which copies the overlapping cells, updates
/// `level.size`, clamps entities + view state, and clears any out-of-bounds pipe
/// path), re-syncs the form's staged size to the clamped result, and marks dirty.
/// No-op if no form is open or the staged size already matches the live map.
pub fn resize(state: &mut EditorState) {
    let Some(st) = state.settings.as_ref() else {
        return;
    };
    let (want_w, want_h, want_d) = (st.size_w, st.size_h, st.size_d);
    // Skip a no-op resize so it never stacks a dead undo step.
    if state.map.size() == (want_w, want_h, want_d) {
        state.status = "resize: size unchanged".into();
        return;
    }
    state.push_undo();
    let (nx, ny, nz) = crate::resize::resize_level(state, want_w, want_h, want_d);
    // Re-sync the staged size to the clamped result so the form reflects reality.
    if let Some(s) = state.settings.as_mut() {
        s.size_w = nx;
        s.size_h = ny;
        s.size_d = nz;
    }
    state.mark_dirty();
    // A structural remap changes the geometry the 3D preview raycasts against.
    state.schedule_preview_rebuild();
    state.status = format!("level resized to {nx}x{ny}x{nz}");
}

/// APPLY the open form into `state.level` as ONE undoable edit. Captures an undo
/// snapshot first, writes the META/ROOM/INPUT/SEQUENCE fields, rebuilds the
/// in-session `sequence.patterns` (see the module doc round-trip note), closes the
/// modal, and marks dirty. No-op if no form is open.
pub fn apply(state: &mut EditorState) {
    let Some(st) = state.settings.take() else {
        return;
    };
    // One undoable edit for the whole form.
    state.push_undo();

    let level = &mut state.level;
    level.name = st.name.clone();
    level.title = st.title.clone();
    level.ambient = AMBIENT_CYCLE[st.ambient_idx].to_string();
    level.storey_height = st.storey_height.max(1);
    level.cycle_keep = st.cycle_keep;
    level.render_pipes = st.render_pipes;

    // INERT AUTHORING METADATA. Every one of these is kept VERBATIM by `re_level`
    // and emitted by `to_lvl_text` only when non-empty (`power_budget` only when
    // `> 0`), so writing a trimmed buffer back is byte-stable: the parser already
    // hands us trimmed values, and trimming is idempotent.
    level.version = st.version.trim().to_string();
    level.spec_rev = st.spec_rev.trim().to_string();
    level.room_floor = st.room_floor.trim().to_string();
    level.room_ceiling = st.room_ceiling.trim().to_string();
    level.door.target_model = st.door_target_model.trim().to_string();
    // `[room] power_budget` — the pipe-power cap the game HUD reads and enforces.
    level.power_budget = st.power_budget_value();

    // EFFECTS ([effects]): write the four knobs verbatim. The writer emits the
    // section ONLY when a value is non-default (snow>0 / steam / matrix>0 /
    // glitch>0), so an all-default effects block stays byte-identical.
    level.effects.snow = st.snow.clamp(0.0, 1.0);
    level.effects.steam = st.steam;
    level.effects.matrix = st.matrix.clamp(0.0, 1.0);
    level.effects.glitch = st.glitch.clamp(0.0, 1.0);

    // FOG ([palette]): the fog TINT is authored iff the hex buffer is non-empty AND
    // parses; otherwise `has_fog` stays false so a fog-less level emits no
    // `fog_color`. The fog DISTANCE is authored iff the user adjusted near/far
    // (`has_fog_dist`). The writer gates the whole `[palette]` block on these flags,
    // so an untouched fog-less level stays byte-identical.
    match parse_hex_color_buf(&st.fog_color) {
        Some(c) => {
            level.palette.fog_color = c;
            level.palette.has_fog = true;
        }
        None => {
            level.palette.has_fog = false;
        }
    }
    level.palette.has_fog_dist = st.has_fog_dist;
    if st.has_fog_dist {
        level.palette.fog_near = st.fog_near.max(0.0);
        level.palette.fog_far = st.fog_far.max(0.0);
    }

    // ATMOSPHERE ([atmosphere]): the per-level LIGHTING mood. Every key is
    // OPTIONAL and written back individually, so an untouched atmosphere-less
    // level keeps `atmos` all-`None` and stays byte-identical. The values are
    // re-clamped to the PARSER's ranges here (ambient `[0,1]`, falloff `>= 0`) so
    // the editor can never emit a `.lvl` its own parser would reject.
    //
    // `atmos.key_dir` is deliberately NOT touched: the key-light DIRECTION has no
    // row (lights are v-next for the editor), and leaving the field alone is what
    // lets a hand-authored `key_dir` survive a load → apply → save round-trip.
    level.atmos.ambient = st.atmos_ambient.map(|a| a.clamp(0.0, 1.0));
    level.atmos.key_color = parse_hex_color_buf(&st.key_color);
    level.atmos.fog_height = if st.has_fog_height {
        Some(re_world::HeightFog {
            base: st.fog_height_base,
            falloff: st.fog_height_falloff.max(0.0),
        })
    } else {
        None
    };
    level.atmos.postfx = match st.postfx_idx {
        0 => None,
        i => Some(postfx_name(i).to_string()),
    };

    // SKY ([sky]): a path is authored iff its buffer is non-empty; an empty buffer
    // clears it to `None`. The writer emits `[sky]` only when at least one path is
    // set, so a sky-less level stays byte-identical.
    level.sky.cylinder = non_empty(&st.sky_cylinder);
    level.sky.cap_top = non_empty(&st.sky_cap_top);
    level.sky.cap_bottom = non_empty(&st.sky_cap_bottom);

    // TEXTURES ([texture NAME]): rebuild `level.textures` from the form rows. A row
    // with a blank name OR blank file is DROPPED so an empty added row never emits a
    // malformed section; the writer gates `[texture]` on a non-empty list, so a
    // texture-less level that gains no textures stays byte-identical.
    //
    // `max_size`, `height`, `frames` and `fps` have no form row (a memory lever, a
    // relief channel and an animation set — not authored looks), so they are CARRIED
    // OVER from the level's existing entry with the same name: rebuilding the list
    // must not silently drop them. Carrying the WHOLE prior entry (rather than a
    // tuple of the fields we remember today) means a future `[texture]` key survives
    // this rebuild without anyone having to remember to widen a tuple. A renamed row
    // loses them, which is the same rule the rest of the form follows for name-keyed
    // data.
    let old_keyed: Vec<re_level::LevelTexture> = level.textures.clone();
    level.textures = st
        .textures
        .iter()
        .filter_map(|(name, file)| {
            let n = name.trim();
            let f = file.trim();
            if n.is_empty() || f.is_empty() {
                None
            } else {
                let mut out = old_keyed
                    .iter()
                    .find(|t| t.name == n)
                    .cloned()
                    .unwrap_or_default();
                out.name = n.to_string();
                out.file = f.to_string();
                Some(out)
            }
        })
        .collect();

    // INPUTS: preserve any `[inputs]` entries that are NOT for our input-node set
    // (e.g. zone-driven names), then re-push one entry per form row. This keeps
    // unrelated entries intact while overwriting the bits the form owns.
    let owned: std::collections::HashSet<&String> = st.inputs.iter().map(|(n, _)| n).collect();
    level.inputs.retain(|(name, _)| !owned.contains(name));
    for (name, on) in &st.inputs {
        level.inputs.push((name.clone(), if *on { 1 } else { 0 }));
    }

    // SEQUENCE: write the AUTHORED fields verbatim (these are what the writer
    // round-trips), then rebuild `patterns` in-session so re_puzzle works without
    // a reload. For each step, build bits = (observe_node_index, step_has_name)
    // over EVERY observe node. If ANY observe/step name fails to resolve to a node
    // index, leave patterns empty — a later load re-resolves from the names.
    level.sequence.observe_names = st.observe.clone();
    level.sequence.step_names = st.steps.clone();
    level.sequence.patterns.clear();
    let observe_indices: Option<Vec<usize>> =
        st.observe.iter().map(|n| level.node_index(n)).collect();
    if let Some(indices) = observe_indices {
        let mut patterns = Vec::with_capacity(st.steps.len());
        let mut all_resolved = true;
        for step in &st.steps {
            // Every step name must be a known observe name to resolve cleanly.
            if step.iter().any(|n| !st.observe.contains(n)) {
                all_resolved = false;
                break;
            }
            let bits: Vec<(usize, bool)> = st
                .observe
                .iter()
                .zip(indices.iter())
                .map(|(name, &idx)| (idx, step.contains(name)))
                .collect();
            patterns.push(re_level::SequencePattern { bits });
        }
        if all_resolved {
            level.sequence.patterns = patterns;
        }
    }

    state.mark_dirty();
    // The form can change what the 3D preview shows (ambient, fog, sky, textures,
    // materials bound by name), so the debounced preview rebuild must be armed —
    // the same rule every other level-mutating path in the editor follows.
    state.schedule_preview_rebuild();
    state.status = "level settings applied".into();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `SettingsState` with the given lists for cursor/list tests.
    fn fixture() -> SettingsState {
        SettingsState {
            name: "vault".into(),
            title: "The Vault".into(),
            version: String::new(),
            spec_rev: String::new(),
            power_budget: String::new(),
            room_floor: String::new(),
            room_ceiling: String::new(),
            door_target_model: String::new(),
            ambient_idx: 0,
            storey_height: 4,
            size_w: 8,
            size_h: 8,
            size_d: 4,
            cycle_keep: false,
            render_pipes: false,
            snow: 0.0,
            steam: false,
            matrix: 0.0,
            glitch: 0.0,
            fog_color: String::new(),
            fog_near: 0.0,
            fog_far: 0.0,
            has_fog_dist: false,
            atmos_ambient: None,
            key_color: String::new(),
            fog_height_base: 0.0,
            fog_height_falloff: 0.0,
            has_fog_height: false,
            postfx_idx: 0,
            sky_cylinder: String::new(),
            sky_cap_top: String::new(),
            sky_cap_bottom: String::new(),
            textures: Vec::new(),
            texture_cursor: 0,
            texture_col: 0,
            inputs: vec![("in0".into(), false), ("in1".into(), true)],
            observe: vec!["outA".into(), "outB".into()],
            steps: vec![vec!["outA".into()], vec!["outB".into()]],
            input_cursor: 0,
            observe_cursor: 0,
            step_cursor: 0,
            name_cursor: 0,
            node_names: vec![
                "in0".into(),
                "in1".into(),
                "outA".into(),
                "outB".into(),
                "extra".into(),
            ],
            output_names: vec!["outA".into(), "outB".into()],
            focus: SettingsField::Name,
        }
    }

    /// SSOT: [`SettingsField::ORDER`] contains EVERY variant EXACTLY ONCE.
    ///
    /// The variants are enumerated here by hand ON PURPOSE — an exhaustive
    /// `match` over a locally-built list is what makes a newly-added variant that
    /// someone forgot to put in `ORDER` a COMPILE error here (the match stops
    /// compiling) or an assertion failure (the counts diverge). Before this, the
    /// order lived in two mirrored hand-written matches and nothing caught a
    /// half-edit.
    #[test]
    fn order_covers_every_variant() {
        use SettingsField as F;
        let all = [
            F::Name,
            F::Title,
            F::Version,
            F::SpecRev,
            F::Ambient,
            F::StoreyHeight,
            F::SizeW,
            F::SizeH,
            F::SizeD,
            F::PowerBudget,
            F::RoomFloor,
            F::RoomCeiling,
            F::CycleKeep,
            F::RenderPipes,
            F::Snow,
            F::Steam,
            F::Matrix,
            F::Glitch,
            F::FogColor,
            F::FogNear,
            F::FogFar,
            F::AtmosAmbient,
            F::KeyColor,
            F::FogHeightBase,
            F::FogHeightFalloff,
            F::Postfx,
            F::SkyCylinder,
            F::SkyCapTop,
            F::SkyCapBottom,
            F::Textures,
            F::DoorTargetModel,
            F::Inputs,
            F::Observe,
            F::Steps,
        ];
        // The exhaustive match: adding a variant without extending `all` above
        // fails to compile, which is the point — the list cannot silently rot.
        for f in all {
            match f {
                F::Name
                | F::Title
                | F::Version
                | F::SpecRev
                | F::Ambient
                | F::StoreyHeight
                | F::SizeW
                | F::SizeH
                | F::SizeD
                | F::PowerBudget
                | F::RoomFloor
                | F::RoomCeiling
                | F::CycleKeep
                | F::RenderPipes
                | F::Snow
                | F::Steam
                | F::Matrix
                | F::Glitch
                | F::FogColor
                | F::FogNear
                | F::FogFar
                | F::AtmosAmbient
                | F::KeyColor
                | F::FogHeightBase
                | F::FogHeightFalloff
                | F::Postfx
                | F::SkyCylinder
                | F::SkyCapTop
                | F::SkyCapBottom
                | F::Textures
                | F::DoorTargetModel
                | F::Inputs
                | F::Observe
                | F::Steps => {}
            }
        }
        assert_eq!(
            SettingsField::ORDER.len(),
            all.len(),
            "ORDER must hold every variant exactly once (no duplicates, no omissions)"
        );
        for f in all {
            let n = SettingsField::ORDER.iter().filter(|o| **o == f).count();
            assert_eq!(n, 1, "{f:?} appears {n} times in ORDER, want exactly 1");
        }
    }

    #[test]
    fn focus_next_walks_order_and_wraps() {
        let order = SettingsField::ORDER;
        let mut f = order[0];
        for expect in order.iter().skip(1).chain(std::iter::once(&order[0])) {
            f = f.next();
            assert_eq!(f, *expect);
        }
        assert_eq!(f, order[0], "next() wraps back to the first field");
    }

    #[test]
    fn focus_prev_is_inverse_of_next() {
        for &f in SettingsField::ORDER {
            assert_eq!(f.next().prev(), f);
            assert_eq!(f.prev().next(), f);
        }
    }

    #[test]
    fn power_budget_types_steps_and_reads_back() {
        let mut s = fixture();
        assert_eq!(s.power_budget_value(), 0, "an empty buffer reads 0");
        // TYPED entry: digits land, non-digits do not.
        for c in "1q2".chars() {
            s.power_budget_push(c);
        }
        assert_eq!(s.power_budget, "12", "only digits are accepted");
        assert_eq!(s.power_budget_value(), 12);
        s.power_budget_pop();
        assert_eq!(s.power_budget, "1");
        // Stepping shares the same buffer and clamps at 0.
        s.power_budget_adjust(4);
        assert_eq!(s.power_budget, "5");
        s.power_budget_adjust(-99);
        assert_eq!(
            s.power_budget, "0",
            "clamps at >= 0 (the parser's constraint)"
        );
        assert_eq!(s.power_budget_value(), 0);
        // The buffer cannot outgrow what a u32 can hold.
        s.power_budget.clear();
        for _ in 0..(POWER_BUDGET_MAX_DIGITS + 5) {
            s.power_budget_push('9');
        }
        assert_eq!(
            s.power_budget.len(),
            POWER_BUDGET_MAX_DIGITS,
            "digit cap holds"
        );
        // 9999999999 > u32::MAX → saturates rather than wrapping to a small number.
        assert_eq!(s.power_budget_value(), u32::MAX);
    }

    #[test]
    fn free_text_metadata_routes_by_field() {
        use SettingsField as F;
        let mut s = fixture();
        for f in [
            F::Version,
            F::SpecRev,
            F::RoomFloor,
            F::RoomCeiling,
            F::DoorTargetModel,
        ] {
            assert!(SettingsState::is_text_field(f), "{f:?} is a text field");
            s.text_push(f, 'a');
            s.text_push(f, 'b');
            assert_eq!(s.text_of(f), "ab", "{f:?} buffer");
            s.text_pop(f);
            assert_eq!(s.text_of(f), "a", "{f:?} pop");
        }
        // Every buffer is DISTINCT (a routing bug would alias two of them).
        assert_eq!(
            (
                s.version.as_str(),
                s.spec_rev.as_str(),
                s.room_floor.as_str(),
                s.room_ceiling.as_str(),
                s.door_target_model.as_str()
            ),
            ("a", "a", "a", "a", "a")
        );
        s.text_push(F::Version, 'Z');
        assert_eq!(s.version, "aZ");
        assert_eq!(s.spec_rev, "a", "the other buffers are untouched");
        // A non-text field is inert on every entry point.
        assert!(!SettingsState::is_text_field(F::Snow));
        s.text_push(F::Snow, 'x');
        s.text_pop(F::Snow);
        assert_eq!(s.text_of(F::Snow), "", "a non-text field reads empty");
    }

    #[test]
    fn open_seeds_the_new_metadata_and_apply_writes_it_back() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.version = "1".into();
        state.level.spec_rev = "2026-06-15".into();
        state.level.power_budget = 7;
        state.level.room_floor = "grid".into();
        state.level.room_ceiling = "dim".into();
        state.level.door.target_model = "test".into();

        open(&mut state);
        {
            let st = state.settings.as_ref().unwrap();
            assert_eq!(st.version, "1");
            assert_eq!(st.spec_rev, "2026-06-15");
            assert_eq!(st.power_budget, "7", "seeded as a digit buffer");
            assert_eq!(st.room_floor, "grid");
            assert_eq!(st.room_ceiling, "dim");
            assert_eq!(st.door_target_model, "test");
        }
        {
            let st = state.settings.as_mut().unwrap();
            st.version = "2".into();
            st.spec_rev = " 2026-08-22 ".into(); // whitespace must be trimmed
            st.power_budget = "12".into();
            st.room_floor = "plate".into();
            st.room_ceiling = "vault".into();
            st.door_target_model = "atrium".into();
        }
        apply(&mut state);

        let lv = &state.level;
        assert_eq!(lv.version, "2");
        assert_eq!(lv.spec_rev, "2026-08-22", "apply trims the buffer");
        assert_eq!(lv.power_budget, 12);
        assert_eq!(lv.room_floor, "plate");
        assert_eq!(lv.room_ceiling, "vault");
        assert_eq!(lv.door.target_model, "atrium");
    }

    #[test]
    fn open_seeds_an_empty_power_budget_when_the_level_has_none() {
        // A level with no `power_budget` must seed EMPTY, not "0" — the writer
        // emits nothing for 0, so an untouched apply stays byte-identical.
        let mut state = EditorState::new_empty(8, 8, 4);
        assert_eq!(state.level.power_budget, 0);
        open(&mut state);
        assert_eq!(state.settings.as_ref().unwrap().power_budget, "");
        apply(&mut state);
        assert_eq!(state.level.power_budget, 0, "untouched apply writes 0 back");
    }

    // -- ATMOSPHERE ([atmosphere]) ----------------------------------------------

    /// `ambient` steps by [`ATMOS_AMBIENT_STEP`], clamps at the parser's `1.0`, and
    /// is reachable BOTH ways: stepping below `0` clears it back to unset.
    #[test]
    fn atmos_ambient_steps_clamps_and_unsets() {
        let mut s = fixture();
        assert_eq!(
            s.atmos_ambient, None,
            "a fresh form has no ambient authored"
        );
        // Stepping DOWN from unset stays unset (it would land below 0).
        s.atmos_ambient_adjust(-1);
        assert_eq!(
            s.atmos_ambient, None,
            "stepping down from unset stays unset"
        );
        // Stepping UP from unset starts one step above 0.
        s.atmos_ambient_adjust(1);
        assert!((s.atmos_ambient.unwrap() - ATMOS_AMBIENT_STEP).abs() < 1e-6);
        // Clamps at the parser's upper bound.
        for _ in 0..200 {
            s.atmos_ambient_adjust(1);
        }
        assert!(
            (s.atmos_ambient.unwrap() - 1.0).abs() < 1e-6,
            "clamps at 1.0"
        );
        assert!(
            (0.0..=1.0).contains(&s.atmos_ambient.unwrap()),
            "never leaves the parser's [0,1]"
        );
        // Walking back down reaches exactly 0.0 ...
        for _ in 0..50 {
            s.atmos_ambient_adjust(-1);
        }
        let at_zero = s.atmos_ambient.expect("0.0 is still an AUTHORED value");
        assert!(at_zero.abs() < 1e-6, "lands on 0.0, got {at_zero}");
        // ... and one more step CLEARS it.
        s.atmos_ambient_adjust(-1);
        assert_eq!(s.atmos_ambient, None, "stepping below 0 clears the field");
    }

    /// `fog_height` is a PAIR: touching EITHER half authors the whole thing, and
    /// the falloff never goes negative (the parser rejects that).
    #[test]
    fn fog_height_pair_is_authored_by_either_half() {
        let mut s = fixture();
        assert!(!s.has_fog_height, "untouched => not authored");
        s.fog_height_base_adjust(3); // +1.5
        assert!(s.has_fog_height, "the base half authors the pair");
        assert!((s.fog_height_base - 1.5).abs() < 1e-6);
        assert!(
            s.fog_height_falloff.abs() < 1e-6,
            "the other half keeps its default"
        );

        let mut s2 = fixture();
        s2.fog_height_falloff_adjust(2); // +0.2
        assert!(s2.has_fog_height, "the falloff half authors the pair too");
        assert!((s2.fog_height_falloff - 0.2).abs() < 1e-6);
        s2.fog_height_falloff_adjust(-99);
        assert!(s2.fog_height_falloff >= 0.0, "falloff clamps at >= 0");
        assert!(s2.fog_height_falloff.abs() < 1e-6);
        // The BASE may legitimately go negative - the parser only wants it finite.
        s2.fog_height_base_adjust(-4);
        assert!(
            (s2.fog_height_base + 2.0).abs() < 1e-6,
            "base may be negative"
        );
        assert!(s2.fog_height_base.is_finite());
    }

    /// `postfx` cycles the unset state plus EVERY known preset, and every value it
    /// can produce is one the parser accepts.
    #[test]
    fn postfx_cycles_every_preset_plus_unset() {
        let mut s = fixture();
        assert_eq!(s.postfx(), "", "unset by default");
        assert_eq!(
            ambient_label(s.postfx()),
            "(default)",
            "unset reads (default)"
        );
        let mut seen: Vec<&str> = Vec::new();
        for _ in 0..POSTFX_CYCLE_LEN {
            s.postfx_cycle(1);
            seen.push(s.postfx());
        }
        assert_eq!(s.postfx_idx, 0, "one full lap wraps back to unset");
        for p in re_world::POSTFX_PRESETS {
            assert!(seen.contains(&p), "the cycle reaches '{p}'");
        }
        for name in seen {
            assert!(
                name.is_empty() || re_world::is_postfx_preset(name),
                "'{name}' must be a preset the parser accepts"
            );
        }
        // Cycling backwards is the inverse.
        s.postfx_cycle(-1);
        assert_eq!(s.postfx(), re_world::POSTFX_PRESETS[POSTFX_CYCLE_LEN - 2]);
    }

    /// The five new rows are SEEDED from the level on [`open`] and WRITTEN BACK on
    /// [`apply`], unchanged.
    #[test]
    fn open_seeds_the_atmosphere_and_apply_writes_it_back() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.atmos.ambient = Some(0.4);
        state.level.atmos.key_color = Some(0x00_FF_88_11);
        state.level.atmos.fog_height = Some(re_world::HeightFog {
            base: 2.5,
            falloff: 0.3,
        });
        state.level.atmos.postfx = Some("cinematic".into());

        open(&mut state);
        {
            let st = state.settings.as_ref().unwrap();
            assert_eq!(st.atmos_ambient, Some(0.4));
            assert_eq!(st.key_color, "#FF8811", "seeded as the writer's hex form");
            assert!(st.has_fog_height);
            assert!((st.fog_height_base - 2.5).abs() < 1e-6);
            assert!((st.fog_height_falloff - 0.3).abs() < 1e-6);
            assert_eq!(st.postfx(), "cinematic");
        }
        // A no-edit apply writes the SAME atmosphere back (byte-identity).
        let before = state.level.atmos.clone();
        apply(&mut state);
        assert_eq!(state.level.atmos, before, "a no-edit apply changes nothing");

        // Now EDIT every row and prove the edit lands on the level.
        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            st.atmos_ambient_adjust(5); // 0.4 + 5*0.02 = 0.5
            st.key_color.clear();
            for c in "#102030".chars() {
                st.key_color_push(c);
            }
            st.fog_height_base_adjust(1); // 2.5 + 0.5 = 3.0
            st.fog_height_falloff_adjust(2); // 0.3 + 0.2 = 0.5
            st.postfx_cycle(1);
        }
        apply(&mut state);
        let a = &state.level.atmos;
        assert!((a.ambient.unwrap() - 0.5).abs() < 1e-5);
        assert_eq!(a.key_color, Some(0x10_2030));
        let hf = a.fog_height.expect("fog_height authored");
        assert!((hf.base - 3.0).abs() < 1e-5);
        assert!((hf.falloff - 0.5).abs() < 1e-5);
        assert_eq!(
            a.postfx.as_deref(),
            Some("toon"),
            "the cycle stepped one preset on"
        );
    }

    /// An UNSET row leaves its `Level.atmos` field `None` - an atmosphere-less
    /// level must survive open -> apply untouched (the writer then emits no
    /// `[atmosphere]` block at all). `key_dir` has no row, and apply must not
    /// clobber a hand-authored one.
    #[test]
    fn untouched_atmosphere_stays_unset_and_key_dir_survives() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.atmos.key_dir = Some(re_core::Vec3::new(0.3, -0.4, 1.0));
        open(&mut state);
        apply(&mut state);
        let a = &state.level.atmos;
        assert_eq!(a.ambient, None, "unset ambient stays None");
        assert_eq!(a.key_color, None, "unset key_color stays None");
        assert_eq!(a.fog_height, None, "unset fog_height stays None");
        assert_eq!(a.postfx, None, "unset postfx stays None");
        assert_eq!(
            a.key_dir,
            Some(re_core::Vec3::new(0.3, -0.4, 1.0)),
            "the editor has no key_dir row, so it must leave one alone"
        );
        // ...and the serializer therefore emits no ambient/postfx line.
        let text = re_voximport::to_lvl_text(&state.level);
        assert!(
            text.contains("key_dir"),
            "the preserved key_dir is still written"
        );
        assert!(
            !text.contains("ambient ="),
            "no ambient line for an unset ambient:\n{text}"
        );
        assert!(
            !text.contains("postfx"),
            "no postfx line for an unset postfx:\n{text}"
        );
    }

    /// The CLAMP EDGES the form can reach are values the REAL parser accepts:
    /// serialize through `re_voximport::to_lvl_text` and reload through
    /// `re_level::load_string`, asserting nothing is rejected or drifts.
    #[test]
    fn atmosphere_clamp_edges_round_trip_through_the_real_parser() {
        let mut state = EditorState::new_empty(8, 8, 4);
        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            // ambient at the UPPER clamp edge (1.0 - the parser's inclusive max).
            for _ in 0..200 {
                st.atmos_ambient_adjust(1);
            }
            // fog_height with a NEGATIVE base (legal: only finiteness is required)
            // and the falloff at its LOWER clamp edge (0.0 - `>= 0`).
            st.fog_height_base_adjust(-3); // -1.5
            st.fog_height_falloff_adjust(-9); // clamps to 0.0
            for c in "#0AF0C1".chars() {
                st.key_color_push(c);
            }
            // The LAST preset in the cycle.
            st.postfx_idx = POSTFX_CYCLE_LEN - 1;
        }
        apply(&mut state);

        let text = re_voximport::to_lvl_text(&state.level);
        assert!(
            text.contains("[atmosphere]"),
            "save emits [atmosphere]:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "atmos-edges.lvl")
            .expect("the editor must never emit an [atmosphere] the parser rejects");
        let a = &reloaded.atmos;
        assert!(
            (a.ambient.unwrap() - 1.0).abs() < 1e-6,
            "ambient 1.0 survives"
        );
        assert_eq!(a.key_color, Some(0x0A_F0C1), "key_color survives");
        let hf = a.fog_height.expect("fog_height survives");
        assert!((hf.base + 1.5).abs() < 1e-6, "negative base survives");
        assert!(hf.falloff.abs() < 1e-6, "falloff 0.0 survives");
        assert_eq!(
            a.postfx.as_deref(),
            Some(re_world::POSTFX_PRESETS[POSTFX_CYCLE_LEN - 2]),
            "the last preset survives"
        );

        // The LOWER ambient edge (0.0) is authored, not unset, and also round-trips.
        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            for _ in 0..50 {
                st.atmos_ambient_adjust(-1);
            }
        }
        apply(&mut state);
        assert_eq!(
            state.level.atmos.ambient.map(|a| a.abs() < 1e-6),
            Some(true)
        );
        let text = re_voximport::to_lvl_text(&state.level);
        let reloaded = re_level::load_string(&text, "atmos-zero.lvl").expect("ambient 0 parses");
        assert!(
            reloaded.atmos.ambient.unwrap().abs() < 1e-6,
            "ambient 0.0 survives"
        );
    }

    #[test]
    fn apply_schedules_the_preview_rebuild() {
        // The form can change ambient / fog / sky / textures, all of which the 3D
        // preview raycasts against — so applying it must arm the debounced rebuild.
        let mut state = EditorState::new_empty(8, 8, 4);
        state.preview_rebuild_delay = None;
        open(&mut state);
        apply(&mut state);
        assert!(
            state.preview_rebuild_delay.is_some(),
            "apply arms the debounced preview rebuild"
        );
    }

    #[test]
    fn ambient_cycles_and_wraps() {
        let mut s = fixture();
        assert_eq!(s.ambient(), "");
        s.ambient_cycle(1);
        assert_eq!(s.ambient(), "trace");
        s.ambient_cycle(1);
        assert_eq!(s.ambient(), "matrix");
        s.ambient_cycle(1);
        assert_eq!(s.ambient(), "solid");
        s.ambient_cycle(1);
        assert_eq!(s.ambient(), "", "wraps back to default");
        s.ambient_cycle(-1);
        assert_eq!(s.ambient(), "solid", "negative wraps too");
    }

    #[test]
    fn storey_clamps_at_one() {
        let mut s = fixture();
        s.storey_height = 1;
        s.storey_adjust(-1);
        assert_eq!(s.storey_height, 1, "clamps at >= 1");
        s.storey_adjust(5);
        assert_eq!(s.storey_height, 6);
    }

    #[test]
    fn bools_toggle() {
        let mut s = fixture();
        assert!(!s.cycle_keep);
        s.toggle_cycle_keep();
        assert!(s.cycle_keep);
        assert!(!s.render_pipes);
        s.toggle_render_pipes();
        assert!(s.render_pipes);
    }

    #[test]
    fn name_and_title_push_pop() {
        let mut s = fixture();
        s.name = String::new();
        s.name_push('h');
        s.name_push('i');
        assert_eq!(s.name, "hi");
        s.name_pop();
        assert_eq!(s.name, "h");
        s.title = String::new();
        s.title_push('x');
        assert_eq!(s.title, "x");
        s.title_pop();
        s.title_pop(); // no panic when empty
        assert_eq!(s.title, "");
    }

    #[test]
    fn input_toggle_by_cursor() {
        let mut s = fixture();
        assert!(!s.inputs[0].1);
        s.toggle_input();
        assert!(s.inputs[0].1, "cursor 0 toggled");
        s.move_input_cursor(1);
        assert_eq!(s.input_cursor, 1);
        assert!(s.inputs[1].1);
        s.toggle_input();
        assert!(!s.inputs[1].1, "cursor 1 toggled");
        // Cursor clamps at the end.
        s.move_input_cursor(5);
        assert_eq!(s.input_cursor, 1);
    }

    #[test]
    fn observe_add_and_remove() {
        let mut s = fixture();
        // Both outA/outB already observed; the next candidate is in0.
        let before = s.observe.len();
        s.add_observe();
        assert_eq!(s.observe.len(), before + 1);
        assert_eq!(s.observe.last().unwrap(), "in0");
        // Remove the cursor row.
        s.observe_cursor = 0;
        s.remove_observe();
        assert!(!s.observe.contains(&"outA".to_string()), "outA removed");
        // Removing an observe name also drops it from steps.
        assert!(s.steps.iter().all(|st| !st.contains(&"outA".to_string())));
    }

    #[test]
    fn step_add_and_remove() {
        let mut s = fixture();
        let before = s.steps.len();
        s.add_step();
        assert_eq!(s.steps.len(), before + 1);
        assert!(s.steps.last().unwrap().is_empty(), "new step is empty");
        assert_eq!(s.step_cursor, s.steps.len() - 1);
        s.remove_step();
        assert_eq!(s.steps.len(), before);
    }

    #[test]
    fn step_name_add_and_remove() {
        let mut s = fixture();
        s.add_step(); // empty step, selected
        s.add_step_name(); // first observe not present → outA
        assert_eq!(s.steps.last().unwrap(), &vec!["outA".to_string()]);
        s.add_step_name(); // next → outB
        assert_eq!(
            s.steps.last().unwrap(),
            &vec!["outA".to_string(), "outB".to_string()]
        );
        s.add_step_name(); // nothing left → no-op
        assert_eq!(s.steps.last().unwrap().len(), 2);
        s.name_cursor = 0;
        s.remove_step_name();
        assert_eq!(s.steps.last().unwrap(), &vec!["outB".to_string()]);
    }

    #[test]
    fn add_observe_prefers_outputs_then_falls_back() {
        let mut s = fixture();
        s.observe.clear();
        s.observe_cursor = 0;
        // First two adds take the OUTPUT nodes, in order (outA, outB).
        s.add_observe();
        assert_eq!(s.observe, vec!["outA".to_string()]);
        s.add_observe();
        assert_eq!(s.observe, vec!["outA".to_string(), "outB".to_string()]);
        // Outputs exhausted → fall back to the first un-observed node (in0).
        s.add_observe();
        assert_eq!(s.observe.last().unwrap(), "in0", "falls back to any node");
        // No outputs at all → fall back path is the only one.
        let mut s2 = fixture();
        s2.observe.clear();
        s2.output_names.clear();
        s2.add_observe();
        assert_eq!(s2.observe.last().unwrap(), "in0", "no outputs → any node");
    }

    /// Build observe + ≥2 ordered steps PURELY through the keyboard helpers
    /// (`add_observe` / `add_step` / `add_step_name`), never by direct field
    /// assignment, and assert the resulting order. This proves the authoring
    /// keystrokes produce the rows the io round-trip test relies on.
    #[test]
    fn sequence_built_via_helpers_has_correct_order() {
        let mut s = fixture();
        s.observe.clear();
        s.steps.clear();
        s.observe_cursor = 0;
        s.step_cursor = 0;
        s.name_cursor = 0;

        // Observe window: two adds → the two outputs in order.
        s.add_observe();
        s.add_observe();
        assert_eq!(
            s.observe,
            vec!["outA".to_string(), "outB".to_string()],
            "observe order from helpers"
        );

        // Step 1: just outA. add_step appends + selects an empty step; add_step_name
        // adds the first un-present observe name (outA).
        s.add_step();
        s.add_step_name();
        // Step 2: just outB. New empty step, then add twice and drop outA so the row
        // is exactly [outB] — exercises add + remove ordering through helpers.
        s.add_step();
        s.add_step_name(); // outA
        s.add_step_name(); // outB
        s.name_cursor = 0;
        s.remove_step_name(); // drop outA → [outB]

        assert_eq!(
            s.steps,
            vec![vec!["outA".to_string()], vec!["outB".to_string()]],
            "step rows + order from helpers only"
        );
    }

    #[test]
    fn open_seeds_from_level_and_apply_round_trips_fields() {
        let mut state = EditorState::new_empty(8, 8, 4);
        // One input node + two output nodes for the observe window.
        let i0 = state.level.push_node(1, 1, 0);
        state.level.nodes[i0].is_input = true;
        state.level.nodes[i0].name = "drive".into();
        let a = state.level.push_node(2, 2, 0);
        state.level.nodes[a].is_output = true;
        state.level.nodes[a].name = "outA".into();
        let b = state.level.push_node(3, 3, 0);
        state.level.nodes[b].is_output = true;
        state.level.nodes[b].name = "outB".into();

        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            st.name = "vault".into();
            st.title = "The Vault".into();
            st.ambient_idx = ambient_idx_for("matrix");
            st.storey_height = 6;
            st.cycle_keep = true;
            st.render_pipes = true;
            // Toggle the (single) input ON.
            st.toggle_input();
            st.observe = vec!["outA".into(), "outB".into()];
            st.steps = vec![vec!["outA".into()], vec!["outB".into()]];
        }
        apply(&mut state);

        let lv = &state.level;
        assert_eq!(lv.name, "vault");
        assert_eq!(lv.title, "The Vault");
        assert_eq!(lv.ambient, "matrix");
        assert_eq!(lv.storey_height, 6);
        assert!(lv.cycle_keep);
        assert!(lv.render_pipes);
        assert!(lv.inputs.iter().any(|(n, v)| n == "drive" && *v == 1));
        assert_eq!(
            lv.sequence.observe_names,
            vec!["outA".to_string(), "outB".to_string()]
        );
        assert_eq!(
            lv.sequence.step_names,
            vec![vec!["outA".to_string()], vec!["outB".to_string()]]
        );
        // In-session patterns were rebuilt (both names resolve to nodes).
        assert_eq!(lv.sequence.patterns.len(), 2);
        assert!(state.dirty);
        assert!(state.settings.is_none(), "apply closes the modal");
    }

    #[test]
    fn open_seeds_staged_size_from_level() {
        let mut state = EditorState::new_empty(7, 9, 3);
        open(&mut state);
        let s = state.settings.as_ref().unwrap();
        assert_eq!(
            (s.size_w, s.size_h, s.size_d),
            (7, 9, 3),
            "staged size = live size"
        );
    }

    #[test]
    fn size_adjust_clamps_to_dim_range() {
        let mut s = fixture();
        s.size_w = MAX_DIM;
        s.size_adjust(SettingsField::SizeW, 10);
        assert_eq!(s.size_w, MAX_DIM, "clamps at MAX_DIM");
        s.size_h = MIN_DIM;
        s.size_adjust(SettingsField::SizeH, -10);
        assert_eq!(s.size_h, MIN_DIM, "clamps at MIN_DIM");
        // A non-size focus is a no-op.
        let before = s.size_d;
        s.size_adjust(SettingsField::Ambient, 5);
        assert_eq!(s.size_d, before, "non-size field is a no-op");
    }

    #[test]
    fn resize_applies_staged_size_as_one_undo_and_keeps_modal_open() {
        let mut state = EditorState::new_empty(4, 4, 2);
        state.map.set(1, 1, 0, re_world::CellKind::WallSolid);
        open(&mut state);
        {
            let s = state.settings.as_mut().unwrap();
            s.size_w = 8;
            s.size_h = 8;
            s.size_d = 4;
        }
        resize(&mut state);
        assert_eq!(state.map.size(), (8, 8, 4), "map resized");
        assert_eq!(state.level.size, (8, 8, 4), "level size updated");
        assert_eq!(
            state.map.cell_at(1, 1, 0),
            re_world::CellKind::WallSolid,
            "overlap preserved"
        );
        assert_eq!(state.history.undo_len(), 1, "resize is one undo step");
        assert!(state.dirty, "resize marks dirty");
        assert!(state.settings.is_some(), "resize keeps the modal open");
        // A second, no-op resize does not stack another undo step.
        resize(&mut state);
        assert_eq!(state.history.undo_len(), 1, "no-op resize stacks no undo");
    }

    #[test]
    fn effects_helpers_clamp() {
        let mut s = fixture();
        s.snow_adjust(-1);
        assert_eq!(s.snow, 0.0, "snow clamps at 0");
        s.snow_adjust(20);
        assert!((s.snow - 1.0).abs() < 1e-6, "snow clamps at 1");
        assert!(!s.steam);
        s.toggle_steam();
        assert!(s.steam);
        s.matrix_adjust(3); // 0.3
        assert!((s.matrix - 0.3).abs() < 1e-6);
        s.glitch_adjust(5); // 0.5
        assert!((s.glitch - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fog_helpers_edit_and_mark_dist() {
        let mut s = fixture();
        for c in "#FF8800".chars() {
            s.fog_color_push(c);
        }
        assert_eq!(s.fog_color, "#FF8800");
        s.fog_color_pop();
        assert_eq!(s.fog_color, "#FF880");
        assert!(!s.has_fog_dist, "untouched near/far is not authored");
        s.fog_near_adjust(2); // +1.0
        assert!((s.fog_near - 1.0).abs() < 1e-6);
        assert!(
            s.has_fog_dist,
            "adjusting near marks the fog distance authored"
        );
        s.fog_far_adjust(-100);
        assert_eq!(s.fog_far, 0.0, "far clamps at 0");
    }

    #[test]
    fn sky_push_pop_routes_by_field() {
        let mut s = fixture();
        s.sky_push(SettingsField::SkyCylinder, 'a');
        s.sky_push(SettingsField::SkyCapTop, 'b');
        s.sky_push(SettingsField::SkyCapBottom, 'c');
        assert_eq!(
            (
                s.sky_cylinder.as_str(),
                s.sky_cap_top.as_str(),
                s.sky_cap_bottom.as_str()
            ),
            ("a", "b", "c")
        );
        s.sky_pop(SettingsField::SkyCylinder);
        assert!(s.sky_cylinder.is_empty());
        // A non-sky field is inert.
        s.sky_push(SettingsField::Name, 'x');
        assert_eq!(s.sky_cap_top, "b");
    }

    #[test]
    fn parse_hex_color_buf_accepts_forms_and_rejects_blank() {
        assert_eq!(parse_hex_color_buf("#FF8800"), Some(0xFF_8800));
        assert_eq!(parse_hex_color_buf("0xff8800"), Some(0xFF_8800));
        assert_eq!(parse_hex_color_buf("ff8800"), Some(0xFF_8800));
        assert_eq!(parse_hex_color_buf(""), None, "blank ⇒ no fog");
        assert_eq!(parse_hex_color_buf("#FFF"), None, "too short ⇒ no fog");
        assert_eq!(parse_hex_color_buf("#GGGGGG"), None, "non-hex ⇒ no fog");
    }

    #[test]
    fn open_seeds_atmosphere_and_apply_writes_it_back() {
        let mut state = EditorState::new_empty(8, 8, 4);
        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            st.snow = 0.4;
            st.toggle_steam();
            st.matrix = 0.2;
            st.glitch = 0.1;
            st.fog_color = "#102030".into();
            st.fog_near_adjust(0); // marks has_fog_dist with the seeded value
            st.fog_near = 2.0;
            st.fog_far = 9.0;
            st.sky_cylinder = "tex/sky.png".into();
            st.sky_cap_top = "tex/top.png".into();
            st.sky_cap_bottom = "tex/bot.png".into();
        }
        apply(&mut state);

        let lv = &state.level;
        assert!((lv.effects.snow - 0.4).abs() < 1e-6);
        assert!(lv.effects.steam);
        assert!((lv.effects.matrix - 0.2).abs() < 1e-6);
        assert!((lv.effects.glitch - 0.1).abs() < 1e-6);
        assert!(lv.palette.has_fog);
        assert_eq!(lv.palette.fog_color & 0xFF_FFFF, 0x10_2030);
        assert!(lv.palette.has_fog_dist);
        assert!((lv.palette.fog_near - 2.0).abs() < 1e-6);
        assert!((lv.palette.fog_far - 9.0).abs() < 1e-6);
        assert_eq!(lv.sky.cylinder.as_deref(), Some("tex/sky.png"));
        assert_eq!(lv.sky.cap_top.as_deref(), Some("tex/top.png"));
        assert_eq!(lv.sky.cap_bottom.as_deref(), Some("tex/bot.png"));
    }

    #[test]
    fn apply_with_no_atmosphere_leaves_flags_off() {
        // A defaulted level → open → apply with NO atmosphere edits must leave every
        // emit flag off so the serializer stays byte-identical (no spurious sections).
        let mut state = EditorState::new_empty(8, 8, 4);
        open(&mut state);
        apply(&mut state);
        let lv = &state.level;
        assert_eq!(lv.effects.snow, 0.0);
        assert!(!lv.effects.steam);
        assert_eq!(lv.effects.matrix, 0.0);
        assert_eq!(lv.effects.glitch, 0.0);
        assert!(!lv.palette.has_fog, "no fog tint authored");
        assert!(!lv.palette.has_fog_dist, "no fog distance authored");
        assert!(lv.sky.is_empty(), "no sky paths authored");
        assert!(lv.textures.is_empty(), "no textures authored");
    }

    #[test]
    fn texture_helpers_add_remove_and_type_by_column() {
        let mut s = fixture();
        assert!(s.textures.is_empty());
        s.focus = SettingsField::Textures;
        s.add_texture();
        assert_eq!(s.textures.len(), 1);
        assert_eq!(s.texture_cursor, 0);
        assert_eq!(s.texture_col, 0, "new row selects the NAME column");
        // Type a name.
        for c in "wall".chars() {
            s.texture_push(c);
        }
        assert_eq!(s.textures[0].0, "wall");
        // Switch to the FILE column and type a path.
        s.move_texture_col(1);
        assert_eq!(s.texture_col, 1);
        for c in "tex/a.webp".chars() {
            s.texture_push(c);
        }
        assert_eq!(s.textures[0].1, "tex/a.webp");
        s.texture_pop();
        assert_eq!(s.textures[0].1, "tex/a.web");
        // Column clamps.
        s.move_texture_col(5);
        assert_eq!(s.texture_col, 1);
        s.move_texture_col(-5);
        assert_eq!(s.texture_col, 0);
        // Remove.
        s.remove_texture();
        assert!(s.textures.is_empty());
    }

    #[test]
    fn open_seeds_textures_and_apply_writes_them_back_dropping_blanks() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.textures.push(re_level::LevelTexture {
            name: "seed".into(),
            file: "tex/seed.webp".into(),
            max_size: None,
            height: None,
            ..Default::default()
        });
        open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            assert_eq!(
                st.textures,
                vec![("seed".to_string(), "tex/seed.webp".to_string())]
            );
            // Add a fully-specified row + a blank row (the blank must be dropped).
            st.focus = SettingsField::Textures;
            st.add_texture();
            st.texture_push('a');
            st.move_texture_col(1);
            for c in "tex/a.webp".chars() {
                st.texture_push(c);
            }
            st.add_texture(); // blank name + blank file → dropped on apply
        }
        apply(&mut state);
        let lv = &state.level;
        assert_eq!(lv.textures.len(), 2, "the blank row is dropped");
        assert_eq!(lv.textures[0].name, "seed");
        assert_eq!(lv.textures[1].name, "a");
        assert_eq!(lv.textures[1].file, "tex/a.webp");
    }

    /// PROTECTS: `[texture] max_size` and `[texture] height` have no form row, so
    /// `apply` rebuilds the texture list WITHOUT them unless they are carried over by
    /// name. Dropping `height` would silently turn an authored lit-relief wall back
    /// into a flat decal on the next save - with no error anywhere, which is exactly
    /// the class of loss the span/subcell writer bug was.
    #[test]
    fn apply_carries_over_the_texture_keys_that_have_no_form_row() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.textures.push(re_level::LevelTexture {
            name: "seed".into(),
            file: "tex/seed.webp".into(),
            max_size: Some(512),
            height: Some("tex/seed_h.png".into()),
            ..Default::default()
        });
        open(&mut state);
        apply(&mut state);
        let t = &state.level.textures[0];
        assert_eq!(
            t.max_size,
            Some(512),
            "an authored cap survives a settings apply"
        );
        assert_eq!(
            t.height.as_deref(),
            Some("tex/seed_h.png"),
            "an authored relief map survives a settings apply"
        );
    }
}
