//! `text_bind` — headless replica of the host game's level-text binder,
//! scoped to exactly what `re_imgqa` needs: a level's text SEEN by the shade
//! path, with no puzzle/game side effects.
//!
//! ## Why this exists
//!
//! `re_render`'s wall/note text is resolved through a per-level
//! [`TextSurfaceRegistry`] (`re_render::scene_text`) that a HOST publishes at
//! load time — the shipped game is that host in a real playthrough. Before
//! this module, `renderqa` never published anything, so every
//! `CodeWall`/`NoteWall` material sampled the empty fallback registry and
//! rendered BLANK. This module is `re_imgqa`'s own host, built without
//! depending on the game binary.
//!
//! ## The note→wall placement algorithm
//!
//! `re_imgqa` must not depend on the host game (an engine crate cannot depend
//! on a binary consumer), so this crate
//! used to keep a **hand-kept duplicate** of the game's private resolver
//! functions (`stamp_notes` / `resolve_note_wall` / `reader_eye_layer` /
//! `wall_in_plane` / `slide_to_free_wall` / `face_is_visible` /
//! `is_paintable_wall` / the search- and slide-distance constants), kept in
//! sync by hand. **This already bit once:** the game learned to snap a
//! placard to the READER'S EYE LAYER while this copy still stamped the
//! authored `z`, so `renderqa` scored a world the game no longer built.
//!
//! That generic mechanism was later extracted into `re_runtime::notes`, an
//! engine crate `re_imgqa` may depend on without reaching into the host game.
//! This module calls `re_runtime::notes::build_level_text` directly — there is
//! no more hand-mirrored copy to drift out of sync.
//!
//! ## The embedded code-panel-0 source
//!
//! `re_imgqa` lives in the engine repository's `editor/` and must not reach
//! across a repository boundary into the host game's `src/` by relative path.
//! The bytes below are therefore a **frozen crate-local snapshot** —
//! `fixtures/code_panel_0.rs.txt`, a verbatim copy of `re_render/src/
//! code_wall.rs` taken at the moment the game and engine were split into
//! separate repositories — not a live cross-repository `include_str!`. It is
//! deliberately the SAME snapshot the shipped game embeds as its own panel 0,
//! so the QA measurement and the shipped wall showed the same text on the day
//! of the split; the two copies do NOT stay in sync automatically afterward
//! (that is the point: the game's visuals no longer change just because
//! engine source changes).
//!
//! ## Lifecycle contract for callers
//!
//! [`bind_level_text`] builds a registry and stamps placards into `map`; it
//! does **not** publish the registry. The caller installs it (e.g. via
//! `re_render::SceneTextScope`) for exactly the span of that level's render
//! section, per `re_render::scene_text`'s "installed by level-load, read by
//! render, possibly different threads" contract — see `main.rs` for the
//! threading argument (levels are processed strictly serially; only the
//! POSES within one level's render section run in parallel, and they only
//! READ the already-installed registry).

use re_level::Level;
use re_render::TextSurfaceRegistry;
use re_runtime::notes::NoteBindReport;
use re_world::Map;

/// `09-code-wall.lvl`'s `recipe = codewall panel=0` shows this same file in
/// the shipped game. Embedding the frozen snapshot of it here keeps the QA
/// measurement on REAL shipped content instead of placeholder filler (see the
/// module docs).
const CODE_PANEL_0_LABEL: &str = "re_render/src/code_wall.rs";
const CODE_PANEL_0_SRC: &str = include_str!("../fixtures/code_panel_0.rs.txt");

/// Build this level's text registry (embedded code panel 0 + one note panel
/// per `[note]` in parse order + the pipe pool table) and stamp its placards
/// onto real wall faces in `map`.
///
/// Call once per level, BEFORE that level's poses are raycast, and publish
/// the returned registry (e.g. `re_render::SceneTextScope::new(registry)`)
/// for the span of that level's render section only.
///
/// Delegates the placement mechanism to [`re_runtime::notes::build_level_text`]
/// — this crate's only remaining job is to supply ITS panel content (`renderqa`
/// has no story/interp layer, so `transform` is `None`).
pub fn bind_level_text(level: &Level, map: &mut Map) -> (TextSurfaceRegistry, NoteBindReport) {
    let panels = [(CODE_PANEL_0_LABEL, CODE_PANEL_0_SRC)];
    re_runtime::notes::build_level_text(level, map, &panels, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::FaceDir;
    use re_level::{LevelNote, NoteKind};
    use re_world::{CellKind, MaterialRecipe};

    /// The active registry `re_render::scene_text` publishes is PROCESS-GLOBAL
    /// (by design — see `re_render::scene_text`'s module docs), and `cargo
    /// test` runs a crate's tests in parallel threads. Every test here that
    /// installs/reads the active registry takes this lock first so they
    /// cannot race each other.
    fn registry_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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

    /// An 8x8x3 room: solid perimeter at z=1, air inside.
    fn room() -> Map {
        let mut m = Map::new(8, 8, 3);
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..3 {
                    let perim = x == 0 || y == 0 || x == 7 || y == 7;
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

    /// Byte-identity proof for the frozen fixture. `fixtures/code_panel_0.rs.txt`
    /// started as a `cp`+`diff`-verified copy of `re_render/src/code_wall.rs` at
    /// the moment of the repository split, and was re-pinned once since (a
    /// pre-public-export text scrub that touched only comments, not code). This
    /// pins the fixture's CURRENT frozen bytes with a length + a dependency-free
    /// FNV-1a content hash (no `sha2` dep in this crate — see `Cargo.toml`), so an
    /// accidental edit to the fixture (not just the engine file it can no longer
    /// reach) fails loudly.
    #[test]
    fn code_panel_0_fixture_is_byte_identical_to_the_split_moment() {
        fn fnv1a(s: &str) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            // CRs are skipped so a CRLF checkout hashes the same as LF.
            for b in s.as_bytes().iter().filter(|b| **b != b'\r') {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }
        assert_eq!(
            CODE_PANEL_0_SRC.bytes().filter(|b| *b != b'\r').count(),
            81_977,
            "fixture length changed since the split"
        );
        assert_eq!(
            fnv1a(CODE_PANEL_0_SRC),
            0xe202_7a77_3354_fe7a,
            "fixture content changed since the split"
        );
    }

    #[test]
    fn a_level_with_no_notes_publishes_a_registry_with_only_the_code_panel() {
        let _serial = registry_test_lock();
        let mut map = room();
        let level = Level::default();
        let (registry, report) = bind_level_text(&level, &mut map);
        assert_eq!(report, NoteBindReport::default());
        assert_eq!(
            registry.code_len(),
            1,
            "the embedded code panel is always registered"
        );
        assert_eq!(registry.note_len(), 0);
    }

    #[test]
    fn binding_publishes_a_nonblank_note_panel_and_stamps_the_wall() {
        let _serial = registry_test_lock();
        let mut map = room();
        let level = Level {
            notes: vec![note(NoteKind::Researcher, 3, 3, 1, 360, "seen and named")],
            ..Default::default()
        };
        let (registry, report) = bind_level_text(&level, &mut map);
        assert_eq!(report.placed, 1);
        assert!(report.unplaced.is_empty());
        assert_eq!(registry.note_len(), 1);

        // The wall got a synthesized NoteWall material.
        let mats = map.materials();
        assert!(matches!(
            mats[5].recipe,
            Some(MaterialRecipe::NoteWall {
                note: 0,
                face: FaceDir::NegX
            })
        ));
        assert_eq!(map.side_data_at(7, 3, 1), 1);

        // The published panel actually samples the authored words, not blank —
        // this is the regression guard: the registry must carry the real text.
        let _s = re_render::SceneTextScope::new(registry);
        // Span 0.0 on both axes = "unknown" ⇒ the plain fitted stretch mapping.
        let g = re_render::scene_text::sample_note_panel(0, 0.02, 0.5, 0.0, 0.0);
        assert_ne!(
            g.glyph.glyph,
            u32::from(' '),
            "note panel must not sample blank at its first word"
        );
    }
}
