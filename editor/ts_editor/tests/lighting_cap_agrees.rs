//! The punctual-light cap is written down TWICE — once in the renderer, which
//! bakes at most that many lamps, and once in `re_lvlcheck`, which warns the
//! author when a level exceeds it. The linter cannot depend on the renderer (the
//! dependency direction is linter → data crates only), so the second copy is
//! hand-pinned and could silently drift; a drift would make the linter promise a
//! lamp that never lights, or warn about one that does.
//!
//! `ts_editor` is the only crate that depends on BOTH, so the guard lives here.

#[test]
fn the_linters_light_cap_matches_the_renderers() {
    assert_eq!(
        re_lvlcheck::MAX_POINT_LIGHTS,
        re_render::ascii_shader::MAX_POINT_LIGHTS,
        "re_lvlcheck's hand-pinned copy of MAX_POINT_LIGHTS has drifted from \
         re_render::ascii_shader::MAX_POINT_LIGHTS (the authority) — update the \
         linter's constant so its warning still describes what the renderer does"
    );
}
