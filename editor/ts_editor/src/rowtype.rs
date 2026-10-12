//! FIELD CLASSIFIERS — map an inspector
//! FIELD to the [`crate::rowvalue::RowValue`] it should render as.
//!
//! `rowvalue.rs` gave a row a TYPE; this module decides WHICH type each of the
//! ~170 concrete fields (entity / emitter / material) actually carries. The three
//! entry points — [`entity_value`], [`emitter_value`], [`material_value`] — are
//! pure classifiers: given the live state and a field selector, they return the
//! richest [`RowValue`] that reproduces the field's EXISTING display text
//! byte-for-byte.
//!
//! ## The parse-back trick (why this file is small and hard to get wrong)
//!
//! Every field already has a ground-truth formatter — `entity::entity_field_value_of`,
//! `entity::emitter_field_value_of`, `material::field_value_of` — and every one of
//! those formatters is ALSO the thing [`RowValue::display`] must reproduce exactly.
//! Rather than re-deriving each field's underlying value from `EditorState`
//! (duplicating a second, fallible path to the same number and risking a type or
//! rounding mismatch), every classifier here:
//!
//! 1. calls the existing formatter to get the ground-truth string `s`,
//! 2. tries to PARSE `s` back into the typed shape (an `i64`, an `f32`, `true`/
//!    `false`, a `#RRGGBB` hex colour),
//! 3. on success, wraps it in the typed [`RowValue`] with the field's clamp/step
//!    (read from the real `adjust_*` stepper source, not guessed),
//! 4. on failure — the value is a sentinel like `"-"` (nothing selected), `"none"`/
//!    `"(none)"` (an unset `Option`), or genuinely composite text — falls back to
//!    `RowValue::Text(s)`, which reproduces `s` unconditionally.
//!
//! Step 4 is what makes this file SAFE by construction: a field's "no selection" /
//! "unset" state never needs a hand-written special case, because the sentinel
//! string simply fails to parse as a number/bool/colour and lands on the always-
//! correct `Text` fallback. `RowValue::Enum` needs no parse step at all — its
//! `display()` is just `current.clone()`, so `Enum { current: s, options }` is
//! ALWAYS exact regardless of whether `s` happens to be a sentinel (the enum's
//! `options` list simply won't contain it in that case, which is explicitly
//! permitted by `RowValue`'s own contract — see `rowvalue.rs`'s doc comment on
//! `Enum::current`).
//!
//! ## Where the option lists come from
//!
//! Every `Enum` row's `options` are produced by the SAME function the `c` cycle
//! itself steps through — `entity::node_glyph_options`, `entity::direction_options`,
//! `entity::glyph_set_options`, `entity::family_options`, `entity::material_name_options`,
//! `material::glyph_options`, `material::preset_options`, `state::facing_options`.
//! None of them is mirrored here, so an option list can never drift from the cycle
//! it describes, and — because those functions splice the live value in
//! (`crate::vocab`) — the list ALWAYS contains the row's current value. That is
//! what `every_enum_option_list_contains_its_current_value_across_the_shipped_corpus`
//! asserts against every object of every shipped level.

use crate::material::MaterialField;
use crate::rowvalue::{parse_hex, RowValue};
use crate::state::{EditorState, EmitterField, EntityField};

// ─────────────────────────── typed-row construction ───────────────────────────

/// Parse `s` as an [`i128`] and wrap it as a [`RowValue::Int`] with the given
/// clamp/step; falls back to [`RowValue::Text`] when `s` is not a bare integer
/// (the `"-"` no-selection sentinel, or any other non-numeric readout).
fn int_row(s: &str, min: i128, max: i128, step: i128) -> RowValue {
    s.parse::<i128>().map_or_else(
        |_| RowValue::Text(s.to_string()),
        |v| RowValue::Int { v, min, max, step },
    )
}

/// Parse `s` as an [`f32`] and wrap it as a [`RowValue::Float`] with the given
/// clamp/step/decimal-places; falls back to [`RowValue::Text`] when `s` is not a
/// bare float (the `"-"` / `"(none)"` sentinels, or composite text).
fn float_row(s: &str, min: f32, max: f32, step: f32, dp: usize) -> RowValue {
    s.parse::<f32>().map_or_else(
        |_| RowValue::Text(s.to_string()),
        |v| RowValue::Float {
            v,
            min,
            max,
            step,
            dp,
        },
    )
}

/// `"true"`/`"false"` -> [`RowValue::Bool`]; anything else (the `"-"`
/// no-selection sentinel) -> [`RowValue::Text`].
fn bool_row(s: &str) -> RowValue {
    match s {
        "true" => RowValue::Bool(true),
        "false" => RowValue::Bool(false),
        _ => RowValue::Text(s.to_string()),
    }
}

/// A `#RRGGBB` readout -> [`RowValue::Color`]; anything else (the `"-"`
/// no-selection sentinel) -> [`RowValue::Text`]. Reuses [`parse_hex`] so the
/// accepted grammar is identical to the colour-overlay's own hex entry.
fn color_row(s: &str) -> RowValue {
    parse_hex(s).map_or_else(|| RowValue::Text(s.to_string()), RowValue::Color)
}

/// Wrap `s` as the CURRENT selection of an enum row. Always exact — `Enum`'s
/// `display()` is just `current`, so this never needs a fallback (see the module
/// doc's "parse-back trick" section). `options` need not contain `s` when the
/// field is unselected (`"-"`) or the datum is stale; `rowvalue.rs` permits that.
fn enum_row(s: &str, options: Vec<String>) -> RowValue {
    RowValue::Enum {
        current: s.to_string(),
        options,
    }
}

// ───────────────────────────── option vocabularies ─────────────────────────────

/// `list.iter().map(ToString::to_string)`, collected — for a plain `&[&str]`
/// hand-coded cycle (one with no backing const array).
fn strs(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

/// The label half of every `(T, &str)` entry in a cycle const (`SPAWN_FACINGS`,
/// `NOTE_KINDS`, `DECAL_SURFACES`, `HOLO_SHAPES`, `SUBCELL_MODES`, `FIT_MODES`),
/// in array order — which is cycle order for every one of them.
fn tuple_labels<T>(list: &[(T, &str)]) -> Vec<String> {
    list.iter().map(|(_, l)| (*l).to_string()).collect()
}

/// The node-glyph cycle vocabulary for the SELECTED node — the canonical list
/// plus whatever off-list glyphs the session has seen, plus this node's own (see
/// [`crate::vocab`]). With no node selected the row reads `"-"`, which no option
/// list contains and none needs to.
fn node_glyph_options(state: &EditorState) -> Vec<char> {
    let cur = state
        .selected_node
        .and_then(|i| state.level.nodes.get(i))
        .map_or(crate::entity::NODE_GLYPHS[0], |n| n.glyph);
    crate::entity::node_glyph_options(state, cur)
}

/// The facing cycle vocabulary as LABELS, for a spawn or a note.
fn facing_labels(state: &EditorState, current: i32) -> Vec<String> {
    crate::state::facing_options(state, current)
        .into_iter()
        .map(crate::state::facing_label)
        .collect()
}

/// The emission-direction cycle vocabulary of the SELECTED emitter, as labels.
fn direction_labels(state: &EditorState) -> Vec<String> {
    let cur = state
        .selected_emitter
        .and_then(|i| state.level.emitters.get(i))
        .map_or([0.0, 0.0, 1.0], |e| {
            [e.direction.x, e.direction.y, e.direction.z]
        });
    crate::entity::direction_options(state, &cur)
        .iter()
        .map(crate::entity::direction_axis_label)
        .collect()
}

/// The `glyph_set` cycle vocabulary of the SELECTED emitter, as labels.
fn glyph_set_labels(state: &EditorState) -> Vec<String> {
    let cur = state
        .selected_emitter
        .and_then(|i| state.level.emitters.get(i))
        .and_then(|e| e.glyph_set.clone());
    crate::entity::glyph_set_options(state, cur.as_ref())
        .iter()
        .map(|g| crate::entity::glyph_set_option_label(g.as_ref()))
        .collect()
}

/// The SELECTED material's glyph (`None` when nothing is selected — the row then
/// reads `"-"`, which no option list contains and none needs to).
fn material_glyph(state: &EditorState) -> Option<char> {
    crate::material::sel_index(state)
        .and_then(|i| state.level.materials[i].glyph)
        .and_then(char::from_u32)
}

/// The SELECTED material's preset name.
fn material_preset(state: &EditorState) -> Option<String> {
    crate::material::sel_index(state).and_then(|i| state.level.materials[i].preset.clone())
}

/// The `(none)` + level-texture-name vocabulary `MaterialField::Texture` cycles
/// through (mirrors `material::cycle_enum`'s `Texture` arm).
fn texture_options(state: &EditorState) -> Vec<String> {
    let mut out = vec!["(none)".to_string()];
    out.extend(state.level.textures.iter().map(|t| t.name.clone()));
    out
}

/// The inclusive upper bound (`extent - 1`, saturating) of the live map's `axis`
/// (0=x, 1=y, 2=z) — the room clamp every position field's `+`/`-` stepper applies
/// via `entity::clamp_axis`.
fn room_max(state: &EditorState, axis: u8) -> i128 {
    let (sx, sy, sz) = state.map.size();
    let extent = match axis {
        0 => sx,
        1 => sy,
        _ => sz,
    };
    (extent as i128 - 1).max(0)
}

// ────────────────────────────────── entity ─────────────────────────────────────

/// Classify one [`EntityField`] of the currently placed/selected object into its
/// richest [`RowValue`]. `RowValue::display()` on the result is guaranteed to
/// equal `entity::entity_field_value_of(state, f)` exactly (see the module doc)
/// — that equality is what the `entity_value_matches_existing_display_*` test
/// below proves, exhaustively, for every field of every [`crate::state::EntityKind`].
#[must_use]
pub fn entity_value(state: &EditorState, f: EntityField) -> RowValue {
    use EntityField as F;
    let s = crate::entity::entity_field_value_of(state, f);
    match f {
        // ── Composite / cursor readouts / formats no typed variant can
        // reproduce exactly (see the field-by-field notes in the delivery
        // report). Text is the always-correct fallback for these. ──
        F::DoorTargetOn
        | F::DoorTargetOff
        | F::NodePool
        | F::NodePoolAlt
        | F::NodeEmits
        | F::TrigTarget
        | F::ZoneNode
        | F::PipeSwitch
        | F::ZoneInsets
        | F::LinePoints
        | F::NoteText
        | F::InsetChild
        // PortalRotate renders via bare `{}` (Rust's shortest-round-trip float
        // Display), which no fixed-`{:.dp}` RowValue::Float can reproduce for an
        // arbitrary (e.g. imported, non-multiple-of-15) rotate value.
        | F::PortalRotate => RowValue::Text(s),

        // ── Bool ──
        F::TrigOnce => bool_row(&s),

        // ── Color ──
        F::NodeColor | F::PartColor | F::PipeColor | F::DecalColorHead | F::DecalColorTail => {
            color_row(&s)
        }

        // ── Enum: static vocabularies ──
        F::NodeGlyph => enum_row(
            &s,
            node_glyph_options(state).iter().map(char::to_string).collect(),
        ),
        F::NodeIo => enum_row(&s, strs(&["neither", "input", "output"])),
        F::SpawnFacing => enum_row(&s, facing_labels(state, state.level.player.facing_yaw_col)),
        F::NoteFacing => {
            let cur = state
                .selected_note
                .and_then(|i| state.level.notes.get(i))
                .map_or(crate::state::SPAWN_FACINGS[0].0, |n| n.facing);
            enum_row(&s, facing_labels(state, cur))
        }
        F::TrigActionType => enum_row(&s, strs(&["activate", "event"])),
        F::PipeMode => enum_row(&s, strs(&["path", "spline"])),
        F::PipeState => enum_row(&s, strs(&["OFF", "ON"])),
        F::NoteKindField => enum_row(&s, tuple_labels(&crate::state::NOTE_KINDS)),
        F::DecalSurface => enum_row(&s, tuple_labels(&crate::state::DECAL_SURFACES)),
        F::HoloShapeField => enum_row(&s, tuple_labels(&crate::state::HOLO_SHAPES)),
        F::ZoneScale | F::LineScale => enum_row(&s, strs(&["none", "crisp", "self_similar"])),

        // ── Enum: dynamic vocabularies (level-authored names) ──
        F::PortalSkin | F::ZoneMaterial | F::LineMaterial => {
            enum_row(&s, crate::entity::material_name_options(state, &s))
        }

        // ── Float: fixed clamp/step (read from the real `adjust_*_numeric`) ──
        F::PipeDensity => float_row(&s, 0.0, 1.0, 0.1, 1),
        F::DecalDensity => float_row(&s, 0.0, 1.0, 0.1, 2),
        F::ZoneWarp | F::LineWarp => float_row(&s, 0.0, f32::MAX, 0.05, 2),
        F::ZoneTwist | F::LineTwist => float_row(&s, f32::MIN, f32::MAX, 0.1, 2),

        // ── Int: field-specific clamp/step ──
        // NodeThreshold and the octave range are unrelated domains that happen to
        // share the same "no static clamp" bound — merged so clippy's
        // `match_same_arms` doesn't flag the duplicate body.
        F::NodeThreshold | F::InsetOctaveLo | F::InsetOctaveHi => {
            int_row(&s, i128::from(i64::MIN), i128::from(i64::MAX), 1)
        }
        F::NodeAnchor => int_row(&s, 0, i128::from(i64::MAX), 1),
        F::PedVolW | F::PedVolH | F::PedVolD => int_row(&s, 1, 9, 1),
        F::PartSizeW | F::PartSizeH | F::PartSizeD => int_row(&s, 1, i128::from(i64::MAX), 1),
        F::HoloStyle => int_row(&s, 0, i128::from(u16::MAX), 1),
        F::ZoneArms | F::LineArms => int_row(&s, 0, i128::from(u8::MAX), 1),

        // ── Int: room-clamped position fields, grouped by axis ──
        F::DoorPosX
        | F::NodePosX
        | F::PedCenterX
        | F::PartOffX
        | F::SpawnPosX
        | F::TrigMinX
        | F::TrigMaxX
        | F::ZoneMinX
        | F::ZoneMaxX
        | F::NotePosX
        | F::DecalMinX
        | F::DecalMaxX
        | F::HoloMinX
        | F::HoloMaxX
        | F::PortalAx0
        | F::PortalAx1
        | F::PortalBx0
        | F::PortalBx1
        | F::ZoneRegMinX
        | F::ZoneRegMaxX
        | F::InsetRegMinX
        | F::InsetRegMaxX
        | F::FieldPosX => int_row(&s, 0, room_max(state, 0), 1),

        F::DoorPosY
        | F::NodePosY
        | F::PedCenterY
        | F::PartOffY
        | F::SpawnPosY
        | F::TrigMinY
        | F::TrigMaxY
        | F::ZoneMinY
        | F::ZoneMaxY
        | F::NotePosY
        | F::DecalMinY
        | F::DecalMaxY
        | F::HoloMinY
        | F::HoloMaxY
        | F::PortalAy0
        | F::PortalAy1
        | F::PortalBy0
        | F::PortalBy1
        | F::ZoneRegMinY
        | F::ZoneRegMaxY
        | F::InsetRegMinY
        | F::InsetRegMaxY
        | F::FieldPosY => int_row(&s, 0, room_max(state, 1), 1),

        F::DoorPosZ
        | F::NodePosZ
        | F::PedCenterZ
        | F::PartOffZ
        | F::SpawnPosZ
        | F::TrigMinZ
        | F::TrigMaxZ
        | F::ZoneMinZ
        | F::ZoneMaxZ
        | F::NotePosZ
        | F::DecalMinZ
        | F::DecalMaxZ
        | F::HoloMinZ
        | F::HoloMaxZ
        | F::PortalAz0
        | F::PortalAz1
        | F::PortalBz0
        | F::PortalBz1
        | F::ZoneRegMinZ
        | F::ZoneRegMaxZ
        | F::InsetRegMinZ
        | F::InsetRegMaxZ
        | F::FieldPosZ => int_row(&s, 0, room_max(state, 2), 1),

        // ── Field (`crate::field`). ShapeKind/Facing/Bind/Drive splice
        // through `state.vocab` — the SAME session-sticky mechanism
        // `entity::node_glyph_options` etc. use (`crate::vocab`'s module
        // doc) — so a hand-authored off-list value (a Box/Capsule shape, a
        // non-cardinal facing vector, a `budget:`/`seq:`/`ring:` target)
        // both shows up in its OWN options here (invariant 1) AND survives a
        // full lap of `c` (invariant 2): the plain "splice only the live
        // current" shape used everywhere else in this match works for a
        // ONE-SHOT display, but a cycle that recomputes it fresh from
        // whatever `c` just landed on loses the original the moment the
        // author steps off it — `crate::vocab`'s sticky `extras` list (seeded
        // by `Vocab::harvest` the moment the level opens) is what makes the
        // lap close. Archetype/Falloff/Occlude/Invert stay plain: every value
        // the parser accepts for them is already on their canonical list, so
        // there is no off-list case to remember. ──
        F::FieldArchetype => enum_row(&s, strs(&re_level::fields::FIELD_ARCHETYPES)),
        // FieldShapeKind stays on the transient `safe_options` splice, NOT the
        // sticky session vocab used by Facing/Bind/Drive above: `field_shape_kind`
        // is a single vocab list SHARED across every field in the level, so a
        // Box shape harvested from field #0 would leak "box" into field #1's
        // (a plain sphere) option list — but `cycle_shape_kind` below is a
        // 2-way sphere<->room toggle that PERMANENTLY refuses to touch a
        // box/capsule CURRENT value, so there is never a "step away and lose
        // it" risk to protect against for THIS field specifically. Splicing
        // only this field's own current value keeps the option count matched
        // to what the toggle actually cycles through (2, unless this field
        // itself is the hand-authored box/capsule).
        F::FieldShapeKind => enum_row(
            &s,
            crate::field::safe_options(vec!["sphere".to_string(), "room".to_string()], &s),
        ),
        F::FieldFalloff => enum_row(&s, strs(&crate::field::FALLOFF_OPTIONS)),
        F::FieldOcclude | F::FieldInvert => enum_row(&s, strs(&["yes", "no"])),
        F::FieldFacing => enum_row(
            &s,
            crate::vocab::options_at(
                &strs(&crate::field::FACING_OPTIONS),
                &state.vocab.field_facing,
                &s,
            )
            .0,
        ),
        F::FieldBind => enum_row(
            &s,
            crate::vocab::options_at(&crate::field::bind_options(&state.level), &state.vocab.field_bind, &s).0,
        ),
        F::FieldDrive => enum_row(
            &s,
            crate::vocab::options_at(&crate::field::drive_options(&state.level), &state.vocab.field_drive, &s).0,
        ),
        F::FieldRadius => float_row(&s, 0.5, f32::MAX, 0.5, 1),
        F::FieldInner => float_row(&s, 0.0, f32::MAX, 0.5, 1),
        F::FieldCone => float_row(&s, 1.0, 180.0, 5.0, 0),
        F::FieldBase | F::FieldDepth => float_row(&s, 0.0, 1.0, 0.05, 2),
        F::FieldPriority => int_row(&s, 0, 9, 1),
    }
}

// ────────────────────────────────── emitter ─────────────────────────────────────

/// Classify one [`EmitterField`] of the currently selected emitter into its
/// richest [`RowValue`]. `RowValue::display()` on the result is guaranteed to
/// equal `entity::emitter_field_value_of(state, f)` exactly — see the module doc
/// and the exhaustive test below.
#[must_use]
pub fn emitter_value(state: &EditorState, f: EmitterField) -> RowValue {
    use EmitterField as F;
    let s = crate::entity::emitter_field_value_of(state, f);
    match f {
        // ── Enum ──
        F::Kind => enum_row(&s, strs(&["Standard", "GodRays"])),
        F::DirectionAxis => enum_row(&s, direction_labels(state)),
        F::ColorMode => enum_row(&s, strs(&["RampOverAge", "RandomPick", "Fixed"])),
        F::Family => enum_row(&s, crate::entity::family_options(state, &s)),
        F::GlyphSet => enum_row(&s, glyph_set_labels(state)),
        F::Bold => enum_row(&s, strs(&["Off", "On", "ByProximity", "ByAge"])),
        F::Blend => enum_row(&s, strs(&["KeepBg", "Opaque"])),

        // ── Color ──
        F::Color0 | F::Color1 | F::Color2 => color_row(&s),

        // ── Float — pairs that coincide numerically (unrelated params, same
        // clamp/step) are merged so clippy's `match_same_arms` doesn't flag the
        // duplicate body. ──
        F::Speed | F::LifetimeSpread => float_row(&s, 0.0, f32::MAX, 0.1, 2),
        F::Rate => float_row(&s, 0.0, f32::MAX, 1.0, 1),
        F::SpeedSpread | F::PathJitter => float_row(&s, 0.0, f32::MAX, 0.05, 2),
        F::SpreadDeg => float_row(&s, 0.0, 180.0, 5.0, 0),
        F::Lifetime => float_row(&s, 0.01, f32::MAX, 0.25, 2),
        F::Size | F::Growth => float_row(&s, f32::EPSILON, f32::MAX, 0.1, 2),
        F::AccelX | F::AccelY | F::AccelZ | F::Swirl => float_row(&s, f32::MIN, f32::MAX, 0.05, 2),
        F::OriginX | F::OriginY | F::OriginZ => float_row(&s, f32::MIN, f32::MAX, 0.1, 1),

        // ── Int ──
        // Burst renders "none" (an unset one-shot count) when `Option::None`; the
        // parse fallback in `int_row` turns that into `Text("none")` automatically.
        // Below 1 the real stepper LEAVES the Int state entirely (clears to None),
        // so 1 is the practical lower bound while a burst count is displayed.
        F::Burst => int_row(&s, 1, i128::from(i64::MAX), 1),
        F::MaxParticles => int_row(&s, 1, i128::from(i64::MAX), 16),
        // Seed is a `u64` stepped by `wrapping_add_signed`. `RowValue::Int` is
        // `i128`, so the FULL u64 range is representable and no seed ever falls
        // back to Text — and the commit path parses/clamps in `i128` too, so a
        // 20-digit seed round-trips exactly rather than through 53 bits of f64.
        F::Seed => int_row(&s, 0, i128::from(u64::MAX), 1),
    }
}

// ────────────────────────────────── material ─────────────────────────────────────

/// Classify one [`MaterialField`] of the currently selected material into its
/// richest [`RowValue`]. `RowValue::display()` on the result is guaranteed to
/// equal `material::field_value_of(state, f)` exactly — see the module doc and
/// the exhaustive test below.
#[must_use]
pub fn material_value(state: &EditorState, f: MaterialField) -> RowValue {
    use MaterialField as F;
    let s = crate::material::field_value_of(state, f);
    match f {
        // ── Color ──
        F::Fg | F::Bg => color_row(&s),

        // ── Enum: static vocabularies ──
        F::Glyph => enum_row(
            &s,
            crate::material::glyph_options(state, material_glyph(state))
                .iter()
                .map(|g| g.map_or_else(|| "none".to_string(), |c| c.to_string()))
                .collect(),
        ),
        F::Recipe => enum_row(
            &s,
            crate::material::recipe_variant_options(state)
                .iter()
                .map(|r| crate::material::recipe_label(*r).to_string())
                .collect(),
        ),
        F::AnimKind => enum_row(&s, strs(&["none", "pulse", "shift", "flicker"])),
        F::Surface => enum_row(&s, strs(&crate::material::SURFACES)),
        F::Preset => enum_row(
            &s,
            crate::material::preset_options(state, material_preset(state).as_deref())
                .iter()
                .map(|p| p.clone().unwrap_or_else(|| "none".to_string()))
                .collect(),
        ),
        F::Subcell => enum_row(&s, tuple_labels(&crate::material::SUBCELL_MODES)),
        F::Fit => enum_row(&s, tuple_labels(&crate::material::FIT_MODES)),

        // ── Enum: dynamic vocabulary (level-authored texture names) ──
        F::Texture => enum_row(&s, texture_options(state)),

        // ── Bool ──
        F::Reflective => bool_row(&s),

        // ── Text always: composite `name=value` readout / no reproducible
        // format for the slot a recipe variant doesn't use. ──
        F::RecipeP0 | F::RecipeP1 | F::RecipeP2 => RowValue::Text(s),

        // ── Int ──
        F::Rank => int_row(&s, 0, 127, 1),

        // ── Float ──
        F::AnimSpeed => float_row(&s, 0.0, f32::MAX, 0.1, 2),
        F::AnimAmplitude => float_row(&s, 0.0, 4.0, 0.05, 2),
        F::Clarity => float_row(&s, 0.0, 1.0, 0.05, 2),
        // Span components render "(none)" while `image_span` is unset; the parse
        // fallback in `float_row` turns that into `Text("(none)")` automatically.
        // The along/up pairs are merged where their clamp/step coincide, so
        // clippy's `match_same_arms` doesn't flag the duplicate body.
        F::SpanAlongMin | F::SpanUpTop => float_row(&s, f32::MIN, f32::MAX, 0.5, 2),
        F::SpanAlongSize | F::SpanUpSize => float_row(&s, 0.5, f32::MAX, 0.5, 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::EntityKind;

    /// Build a state with exactly one object of `kind` placed AND selected at a
    /// harmless interior cell, mirroring the pattern every `entity.rs` test uses.
    /// `Pipe` is not point-placed (it is wired between two nodes), so it gets its
    /// own two-node connect flow; every other kind is placed with one
    /// `place_at_cursor` call (which also selects a freshly-created object).
    fn placed(kind: EntityKind) -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(kind);
        if kind == EntityKind::Pipe {
            s.level.push_node(1, 1, 0);
            s.level.push_node(4, 3, 0);
            s.cursor = (1, 1);
            s.z = 0;
            crate::pipe::start_pipe_create(&mut s);
            s.cursor = (4, 3);
            crate::pipe::pick_node(&mut s, 1);
        } else {
            s.cursor = (2, 2);
            s.z = 0;
            crate::entity::place_at_cursor(&mut s);
        }
        s
    }

    /// The 17 [`EntityKind`] variants, via the object-selector's own `next()`
    /// cycle starting at `Door` — so this list can never drift from the real
    /// selector order/coverage.
    fn all_kinds() -> Vec<EntityKind> {
        let mut k = EntityKind::Door;
        let mut out = Vec::new();
        for _ in 0..17 {
            out.push(k);
            k = k.next();
        }
        out
    }

    // ── INVARIANT 1/3: entity_value reproduces entity_field_value_of exactly,
    // for every field of every kind, on a freshly placed+selected object. This is
    // the proof that classifying fields changes no rendered text. ──
    #[test]
    fn entity_value_matches_existing_display_for_every_kind_and_field() {
        for kind in all_kinds() {
            let s = placed(kind);
            for &f in crate::state::entity_fields_for(kind) {
                let expected = crate::entity::entity_field_value_of(&s, f);
                let got = entity_value(&s, f).display();
                assert_eq!(got, expected, "kind {kind:?} field {f:?}");
            }
        }
    }

    #[test]
    fn emitter_value_matches_existing_display_for_every_field() {
        let s = placed(EntityKind::Emitter);
        for &f in &crate::state::EMITTER_FIELDS {
            let expected = crate::entity::emitter_field_value_of(&s, f);
            let got = emitter_value(&s, f).display();
            assert_eq!(got, expected, "field {f:?}");
        }
    }

    #[test]
    fn material_value_matches_existing_display_for_every_field() {
        let mut s = EditorState::new_empty(16, 16, 4);
        crate::material::new_material(&mut s);
        let unique =
            &crate::material::MATERIAL_FIELDS[..crate::material::MATERIAL_FIELDS.len() - 1];
        for &f in unique {
            let expected = crate::material::field_value_of(&s, f);
            let got = material_value(&s, f).display();
            assert_eq!(got, expected, "field {f:?}");
        }
    }

    // ── DELIVERABLE: a typing check that can actually FAIL ──────────────────
    //
    // The three `*_value_matches_existing_display_*` tests above are exact but
    // NEARLY VACUOUS on their own: they assert `value(f).display() == formatter(f)`,
    // and for any field that falls back to `RowValue::Text(s)`, `display()` IS `s`.
    // So `s == s` — the assertion passes identically whether the field is correctly
    // typed or has silently degraded to Text. They prove "the text did not change";
    // they cannot prove "the field has the right type".
    //
    // `representative_fields_land_on_the_expected_variant` covered 5 of ~170 fields.
    // The tables below cover ALL of them: every field's INTENDED variant, checked
    // against what the classifier really returns.
    //
    // The one legitimate escape is a SENTINEL readout — `"-"` (nothing selected),
    // `"none"` / `"(none)"` (an unset `Option`), `"(empty)"` — which by design
    // parses as nothing and lands on the always-correct `Text` fallback. Those are
    // allowed, and ONLY those: a field whose readout is real data but whose row is
    // `Text` is a silent degradation and fails here, naming the field.

    /// A readout that legitimately has no typed shape.
    fn is_sentinel(s: &str) -> bool {
        matches!(s, "-" | "none" | "(none)" | "(empty)")
    }

    /// The variant name a [`RowValue`] carries.
    fn variant_of(v: &RowValue) -> &'static str {
        match v {
            RowValue::Text(_) => "Text",
            RowValue::Int { .. } => "Int",
            RowValue::Float { .. } => "Float",
            RowValue::Bool(_) => "Bool",
            RowValue::Enum { .. } => "Enum",
            RowValue::Color(_) => "Color",
        }
    }

    /// Assert one row landed on `want`, or on `Text` because its readout is a
    /// sentinel.
    fn assert_variant(v: &RowValue, want: &str, at: &str, failures: &mut Vec<String>) {
        let got = variant_of(v);
        if got == want {
            return;
        }
        if got == "Text" && is_sentinel(&v.display()) {
            return;
        }
        failures.push(format!(
            "  {at}: expected {want}, got {got} ({:?})",
            v.display()
        ));
    }

    /// The intended [`RowValue`] variant of EVERY [`EntityField`].
    const ENTITY_VARIANTS: &[(EntityField, &str)] = &[
        (EntityField::DecalColorHead, "Color"),
        (EntityField::DecalColorTail, "Color"),
        (EntityField::DecalDensity, "Float"),
        (EntityField::DecalMaxX, "Int"),
        (EntityField::DecalMaxY, "Int"),
        (EntityField::DecalMaxZ, "Int"),
        (EntityField::DecalMinX, "Int"),
        (EntityField::DecalMinY, "Int"),
        (EntityField::DecalMinZ, "Int"),
        (EntityField::DecalSurface, "Enum"),
        (EntityField::DoorPosX, "Int"),
        (EntityField::DoorPosY, "Int"),
        (EntityField::DoorPosZ, "Int"),
        (EntityField::DoorTargetOff, "Text"),
        (EntityField::DoorTargetOn, "Text"),
        (EntityField::HoloMaxX, "Int"),
        (EntityField::HoloMaxY, "Int"),
        (EntityField::HoloMaxZ, "Int"),
        (EntityField::HoloMinX, "Int"),
        (EntityField::HoloMinY, "Int"),
        (EntityField::HoloMinZ, "Int"),
        (EntityField::HoloShapeField, "Enum"),
        (EntityField::HoloStyle, "Int"),
        (EntityField::InsetChild, "Text"),
        (EntityField::InsetOctaveHi, "Int"),
        (EntityField::InsetOctaveLo, "Int"),
        (EntityField::InsetRegMaxX, "Int"),
        (EntityField::InsetRegMaxY, "Int"),
        (EntityField::InsetRegMaxZ, "Int"),
        (EntityField::InsetRegMinX, "Int"),
        (EntityField::InsetRegMinY, "Int"),
        (EntityField::InsetRegMinZ, "Int"),
        (EntityField::LineArms, "Int"),
        (EntityField::LineMaterial, "Enum"),
        (EntityField::LinePoints, "Text"),
        (EntityField::LineScale, "Enum"),
        (EntityField::LineTwist, "Float"),
        (EntityField::LineWarp, "Float"),
        (EntityField::NodeAnchor, "Int"),
        (EntityField::NodeColor, "Color"),
        (EntityField::NodeEmits, "Text"),
        (EntityField::NodeGlyph, "Enum"),
        (EntityField::NodeIo, "Enum"),
        (EntityField::NodePool, "Text"),
        (EntityField::NodePoolAlt, "Text"),
        (EntityField::NodePosX, "Int"),
        (EntityField::NodePosY, "Int"),
        (EntityField::NodePosZ, "Int"),
        (EntityField::NodeThreshold, "Int"),
        (EntityField::NoteFacing, "Enum"),
        (EntityField::NoteKindField, "Enum"),
        (EntityField::NotePosX, "Int"),
        (EntityField::NotePosY, "Int"),
        (EntityField::NotePosZ, "Int"),
        (EntityField::NoteText, "Text"),
        (EntityField::PartColor, "Color"),
        (EntityField::PartOffX, "Int"),
        (EntityField::PartOffY, "Int"),
        (EntityField::PartOffZ, "Int"),
        (EntityField::PartSizeD, "Int"),
        (EntityField::PartSizeH, "Int"),
        (EntityField::PartSizeW, "Int"),
        (EntityField::PedCenterX, "Int"),
        (EntityField::PedCenterY, "Int"),
        (EntityField::PedCenterZ, "Int"),
        (EntityField::PedVolD, "Int"),
        (EntityField::PedVolH, "Int"),
        (EntityField::PedVolW, "Int"),
        (EntityField::PipeColor, "Color"),
        (EntityField::PipeDensity, "Float"),
        (EntityField::PipeMode, "Enum"),
        (EntityField::PipeState, "Enum"),
        (EntityField::PipeSwitch, "Text"),
        (EntityField::PortalAx0, "Int"),
        (EntityField::PortalAx1, "Int"),
        (EntityField::PortalAy0, "Int"),
        (EntityField::PortalAy1, "Int"),
        (EntityField::PortalAz0, "Int"),
        (EntityField::PortalAz1, "Int"),
        (EntityField::PortalBx0, "Int"),
        (EntityField::PortalBx1, "Int"),
        (EntityField::PortalBy0, "Int"),
        (EntityField::PortalBy1, "Int"),
        (EntityField::PortalBz0, "Int"),
        (EntityField::PortalBz1, "Int"),
        (EntityField::PortalRotate, "Text"),
        (EntityField::PortalSkin, "Enum"),
        (EntityField::SpawnFacing, "Enum"),
        (EntityField::SpawnPosX, "Int"),
        (EntityField::SpawnPosY, "Int"),
        (EntityField::SpawnPosZ, "Int"),
        (EntityField::TrigActionType, "Enum"),
        (EntityField::TrigMaxX, "Int"),
        (EntityField::TrigMaxY, "Int"),
        (EntityField::TrigMaxZ, "Int"),
        (EntityField::TrigMinX, "Int"),
        (EntityField::TrigMinY, "Int"),
        (EntityField::TrigMinZ, "Int"),
        (EntityField::TrigOnce, "Bool"),
        (EntityField::TrigTarget, "Text"),
        (EntityField::ZoneArms, "Int"),
        (EntityField::ZoneInsets, "Text"),
        (EntityField::ZoneMaterial, "Enum"),
        (EntityField::ZoneMaxX, "Int"),
        (EntityField::ZoneMaxY, "Int"),
        (EntityField::ZoneMaxZ, "Int"),
        (EntityField::ZoneMinX, "Int"),
        (EntityField::ZoneMinY, "Int"),
        (EntityField::ZoneMinZ, "Int"),
        (EntityField::ZoneNode, "Text"),
        (EntityField::ZoneRegMaxX, "Int"),
        (EntityField::ZoneRegMaxY, "Int"),
        (EntityField::ZoneRegMaxZ, "Int"),
        (EntityField::ZoneRegMinX, "Int"),
        (EntityField::ZoneRegMinY, "Int"),
        (EntityField::ZoneRegMinZ, "Int"),
        (EntityField::ZoneScale, "Enum"),
        (EntityField::ZoneTwist, "Float"),
        (EntityField::ZoneWarp, "Float"),
        (EntityField::FieldArchetype, "Enum"),
        (EntityField::FieldShapeKind, "Enum"),
        (EntityField::FieldPosX, "Int"),
        (EntityField::FieldPosY, "Int"),
        (EntityField::FieldPosZ, "Int"),
        (EntityField::FieldRadius, "Float"),
        (EntityField::FieldInner, "Float"),
        (EntityField::FieldFalloff, "Enum"),
        (EntityField::FieldOcclude, "Enum"),
        (EntityField::FieldFacing, "Enum"),
        (EntityField::FieldCone, "Float"),
        (EntityField::FieldBind, "Enum"),
        (EntityField::FieldInvert, "Enum"),
        (EntityField::FieldDrive, "Enum"),
        (EntityField::FieldBase, "Float"),
        (EntityField::FieldDepth, "Float"),
        (EntityField::FieldPriority, "Int"),
    ];

    /// The intended [`RowValue`] variant of EVERY [`EmitterField`].
    const EMITTER_VARIANTS: &[(EmitterField, &str)] = &[
        (EmitterField::Kind, "Enum"),
        (EmitterField::Rate, "Float"),
        (EmitterField::Burst, "Int"),
        (EmitterField::MaxParticles, "Int"),
        (EmitterField::Speed, "Float"),
        (EmitterField::SpeedSpread, "Float"),
        (EmitterField::DirectionAxis, "Enum"),
        (EmitterField::SpreadDeg, "Float"),
        (EmitterField::Lifetime, "Float"),
        (EmitterField::LifetimeSpread, "Float"),
        (EmitterField::Size, "Float"),
        (EmitterField::Growth, "Float"),
        (EmitterField::AccelX, "Float"),
        (EmitterField::AccelY, "Float"),
        (EmitterField::AccelZ, "Float"),
        (EmitterField::Swirl, "Float"),
        (EmitterField::PathJitter, "Float"),
        (EmitterField::Color0, "Color"),
        (EmitterField::Color1, "Color"),
        (EmitterField::Color2, "Color"),
        (EmitterField::ColorMode, "Enum"),
        (EmitterField::Family, "Enum"),
        (EmitterField::GlyphSet, "Enum"),
        (EmitterField::Bold, "Enum"),
        (EmitterField::Blend, "Enum"),
        (EmitterField::Seed, "Int"),
        (EmitterField::OriginX, "Float"),
        (EmitterField::OriginY, "Float"),
        (EmitterField::OriginZ, "Float"),
    ];

    /// The intended [`RowValue`] variant of EVERY [`MaterialField`].
    const MATERIAL_VARIANTS: &[(MaterialField, &str)] = &[
        (MaterialField::Fg, "Color"),
        (MaterialField::Bg, "Color"),
        (MaterialField::Glyph, "Enum"),
        (MaterialField::Recipe, "Enum"),
        (MaterialField::RecipeP0, "Text"),
        (MaterialField::RecipeP1, "Text"),
        (MaterialField::RecipeP2, "Text"),
        (MaterialField::AnimKind, "Enum"),
        (MaterialField::AnimSpeed, "Float"),
        (MaterialField::AnimAmplitude, "Float"),
        (MaterialField::Rank, "Int"),
        (MaterialField::Surface, "Enum"),
        (MaterialField::Reflective, "Bool"),
        (MaterialField::Clarity, "Float"),
        (MaterialField::Preset, "Enum"),
        (MaterialField::Texture, "Enum"),
        (MaterialField::Subcell, "Enum"),
        (MaterialField::Fit, "Enum"),
        (MaterialField::SpanAlongMin, "Float"),
        (MaterialField::SpanAlongSize, "Float"),
        (MaterialField::SpanUpTop, "Float"),
        (MaterialField::SpanUpSize, "Float"),
    ];

    /// Every entity field of every kind lands on its INTENDED variant — not on a
    /// silent `Text` fallback.
    #[test]
    fn every_entity_field_lands_on_its_intended_variant() {
        let mut failures = Vec::new();
        let mut covered = 0usize;
        for kind in all_kinds() {
            let s = placed(kind);
            for &f in crate::state::entity_fields_for(kind) {
                let want = ENTITY_VARIANTS
                    .iter()
                    .find(|(g, _)| *g == f)
                    .unwrap_or_else(|| panic!("ENTITY_VARIANTS has no entry for {f:?}"))
                    .1;
                assert_variant(
                    &entity_value(&s, f),
                    want,
                    &format!("{kind:?}.{f:?}"),
                    &mut failures,
                );
                covered += 1;
            }
        }
        assert_eq!(
            covered,
            ENTITY_VARIANTS.len(),
            "every EntityField belongs to exactly one kind, so the sweep must visit              the whole table exactly once"
        );
        assert!(
            failures.is_empty(),
            "silently mistyped rows:
{}",
            failures.join(
                "
"
            )
        );
    }

    /// Same for every emitter field.
    #[test]
    fn every_emitter_field_lands_on_its_intended_variant() {
        let s = placed(EntityKind::Emitter);
        let mut failures = Vec::new();
        for &(f, want) in EMITTER_VARIANTS {
            assert_variant(
                &emitter_value(&s, f),
                want,
                &format!("emitter.{f:?}"),
                &mut failures,
            );
        }
        assert_eq!(EMITTER_VARIANTS.len(), crate::state::EMITTER_FIELDS.len());
        assert!(
            failures.is_empty(),
            "silently mistyped rows:
{}",
            failures.join(
                "
"
            )
        );
    }

    /// Same for every material field.
    #[test]
    fn every_material_field_lands_on_its_intended_variant() {
        let mut s = EditorState::new_empty(16, 16, 4);
        crate::material::new_material(&mut s);
        let mut failures = Vec::new();
        for &(f, want) in MATERIAL_VARIANTS {
            assert_variant(
                &material_value(&s, f),
                want,
                &format!("material.{f:?}"),
                &mut failures,
            );
        }
        assert_eq!(
            MATERIAL_VARIANTS.len(),
            crate::material::MATERIAL_FIELDS.len() - 1,
            "the table must cover every unique material field"
        );
        assert!(
            failures.is_empty(),
            "silently mistyped rows:
{}",
            failures.join(
                "
"
            )
        );
    }

    /// The tables are only honest if they are COMPLETE — a field added to the
    /// inspector without a table entry must fail here, not be skipped silently.
    #[test]
    fn the_variant_tables_cover_every_field() {
        for kind in all_kinds() {
            for f in crate::state::entity_fields_for(kind) {
                assert!(
                    ENTITY_VARIANTS.iter().any(|(g, _)| g == f),
                    "ENTITY_VARIANTS is missing {f:?}"
                );
            }
        }
        for f in &crate::state::EMITTER_FIELDS {
            assert!(
                EMITTER_VARIANTS.iter().any(|(g, _)| g == f),
                "EMITTER_VARIANTS is missing {f:?}"
            );
        }
        let unique =
            &crate::material::MATERIAL_FIELDS[..crate::material::MATERIAL_FIELDS.len() - 1];
        for f in unique {
            assert!(
                MATERIAL_VARIANTS.iter().any(|(g, _)| g == f),
                "MATERIAL_VARIANTS is missing {f:?}"
            );
        }
    }

    // ── Sanity: a representative sample landed on the intended variant, not a
    // silent Text fallback. ──
    #[test]
    fn representative_fields_land_on_the_expected_variant() {
        // TrigOnce -> Bool (needs a selected trigger; unselected renders "-").
        // A freshly-placed trigger defaults `once: true` (entity::place_at_cursor).
        let mut s = placed(EntityKind::Trigger);
        assert!(matches!(
            entity_value(&s, EntityField::TrigOnce),
            RowValue::Bool(true)
        ));

        // NodeColor -> Color.
        s = placed(EntityKind::Node);
        assert!(matches!(
            entity_value(&s, EntityField::NodeColor),
            RowValue::Color(_)
        ));

        // HoloShapeField -> Enum with the three HOLO_SHAPES labels, current "box".
        s = placed(EntityKind::Holo);
        match entity_value(&s, EntityField::HoloShapeField) {
            RowValue::Enum { current, options } => {
                assert_eq!(current, "box");
                assert_eq!(options, vec!["box", "pyramid", "sphere"]);
            }
            other => panic!("expected Enum, got {other:?}"),
        }

        // HoloStyle -> Int.
        assert!(matches!(
            entity_value(&s, EntityField::HoloStyle),
            RowValue::Int { .. }
        ));

        // DecalDensity -> Float, dp 2, step 0.1, max 1.0 (mirrors PipeDensity's
        // clamp but keeps the field's own 2-decimal display).
        s = placed(EntityKind::WallDecal);
        match entity_value(&s, EntityField::DecalDensity) {
            RowValue::Float { dp, step, max, .. } => {
                assert_eq!(dp, 2);
                assert!((step - 0.1).abs() < 1e-6);
                assert!((max - 1.0).abs() < 1e-6);
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    // ── INVARIANT-adjacent: the classified step/min/max actually match the real
    // stepper — read the field, call the real `adjust_*`, and check the new value
    // equals the OLD RowValue clamped by its own step. A later "set an exact
    // value" feature computes `delta = round((target - v) / step)` and replays the
    // real stepper that many times, so a wrong step/min/max here would silently
    // corrupt every direct-entry edit. ──
    #[test]
    fn classified_step_and_clamp_match_the_real_steppers() {
        // Door position: room-clamped Int.
        let mut s = placed(EntityKind::Door);
        s.place = Some(EntityKind::Door);
        s.entity_field = EntityField::DoorPosX;
        let RowValue::Int {
            v: before,
            min,
            max,
            step,
        } = entity_value(&s, EntityField::DoorPosX)
        else {
            panic!("DoorPosX should classify as Int");
        };
        crate::entity::adjust_property(&mut s, 1);
        let after: i128 = crate::entity::entity_field_value_of(&s, EntityField::DoorPosX)
            .parse()
            .unwrap();
        assert_eq!(after, (before + step).clamp(min, max));

        // Pedestal volume: custom clamp [1, 9].
        let mut s = placed(EntityKind::Pedestal);
        s.entity_field = EntityField::PedVolW;
        let RowValue::Int {
            v: before,
            min,
            max,
            step,
        } = entity_value(&s, EntityField::PedVolW)
        else {
            panic!("PedVolW should classify as Int");
        };
        crate::entity::adjust_property(&mut s, 1);
        let after: i128 = crate::entity::entity_field_value_of(&s, EntityField::PedVolW)
            .parse()
            .unwrap();
        assert_eq!(after, (before + step).clamp(min, max));

        // Hologram style: unbounded-above u16 counter.
        let mut s = placed(EntityKind::Holo);
        s.entity_field = EntityField::HoloStyle;
        let RowValue::Int {
            v: before,
            min,
            max,
            step,
        } = entity_value(&s, EntityField::HoloStyle)
        else {
            panic!("HoloStyle should classify as Int");
        };
        crate::entity::adjust_property(&mut s, 1);
        let after: i128 = crate::entity::entity_field_value_of(&s, EntityField::HoloStyle)
            .parse()
            .unwrap();
        assert_eq!(after, (before + step).clamp(min, max));

        // Pipe density: 0.1 step, [0,1] clamp, stepped DOWN from the 1.0 default
        // so the clamp is not immediately saturating.
        let mut s = placed(EntityKind::Pipe);
        s.entity_field = EntityField::PipeDensity;
        let RowValue::Float {
            v: before,
            min,
            max,
            step,
            ..
        } = entity_value(&s, EntityField::PipeDensity)
        else {
            panic!("PipeDensity should classify as Float");
        };
        crate::entity::adjust_property(&mut s, -1);
        let after: f32 = crate::entity::entity_field_value_of(&s, EntityField::PipeDensity)
            .parse()
            .unwrap();
        assert!((after - (before - step).clamp(min, max)).abs() < 1e-6);

        // Material rank: [0, 127] clamp.
        let mut s = EditorState::new_empty(16, 16, 4);
        crate::material::new_material(&mut s);
        s.material_field = MaterialField::Rank;
        let RowValue::Int {
            v: before,
            min,
            max,
            step,
        } = material_value(&s, MaterialField::Rank)
        else {
            panic!("Rank should classify as Int");
        };
        crate::material::adjust_field(&mut s, 1);
        let after: i128 = crate::material::field_value_of(&s, MaterialField::Rank)
            .parse()
            .unwrap();
        assert_eq!(after, (before + step).clamp(min, max));

        // Emitter max_particles: 16-step, floor 1.
        let mut s = placed(EntityKind::Emitter);
        s.emitter_field = EmitterField::MaxParticles;
        let RowValue::Int {
            v: before,
            min,
            max,
            step,
        } = emitter_value(&s, EmitterField::MaxParticles)
        else {
            panic!("MaxParticles should classify as Int");
        };
        crate::entity::adjust_property(&mut s, 1);
        let after: i128 = crate::entity::emitter_field_value_of(&s, EmitterField::MaxParticles)
            .parse()
            .unwrap();
        assert_eq!(after, (before + step).clamp(min, max));
    }

    // ── INVARIANT 4: every Enum row's options list CONTAINS its current value,
    // on a freshly placed+selected object — a popup that cannot show the current
    // selection is a bug. ──
    #[test]
    fn every_enum_option_list_contains_its_current_value() {
        for kind in all_kinds() {
            let s = placed(kind);
            for &f in crate::state::entity_fields_for(kind) {
                if let RowValue::Enum { current, options } = entity_value(&s, f) {
                    assert!(
                        options.contains(&current),
                        "kind {kind:?} field {f:?}: options {options:?} missing current {current:?}"
                    );
                }
            }
        }

        let s = placed(EntityKind::Emitter);
        for &f in &crate::state::EMITTER_FIELDS {
            if let RowValue::Enum { current, options } = emitter_value(&s, f) {
                assert!(
                    options.contains(&current),
                    "emitter field {f:?}: options {options:?} missing current {current:?}"
                );
            }
        }

        let mut s = EditorState::new_empty(16, 16, 4);
        crate::material::new_material(&mut s);
        let unique =
            &crate::material::MATERIAL_FIELDS[..crate::material::MATERIAL_FIELDS.len() - 1];
        for &f in unique {
            if let RowValue::Enum { current, options } = material_value(&s, f) {
                assert!(
                    options.contains(&current),
                    "material field {f:?}: options {options:?} missing current {current:?}"
                );
            }
        }
    }
}
