//! Missing/placeholder-asset notice collection over a loaded [`re_level::Level`].
//! No seams: a pure read of a parsed level, with no game type involved.
//!
//! The load-time notice for a level asset that is not there.
//!
//! ## Why the game does NOT refuse to launch
//!
//! `re_level` can resolve every file a `.lvl` names and report the ones that are
//! missing ([`re_level::Level::missing_assets`]), and `re_lvlcheck` turns exactly
//! that into a hard FAIL. The game deliberately does not: refusing to start
//! because a decorative wall texture was moved is worse for a player than an
//! untextured wall, and `build_map`'s best-effort skip is the right default for
//! the runtime.
//!
//! ## Why it must not be SILENT either
//!
//! The lenient default was implemented as *nothing at all*. Deleting a level's
//! bound texture file made the game render an untextured level with
//! zero output — the asset was authored, parsed, resolved, found missing, the
//! miss was formatted into a message naming every path probed… and then dropped
//! on the floor, because nothing outside `re_level`'s own unit tests ever called
//! the function that returned it. An authored thing that is not on screen, with
//! nothing anywhere saying why, is this project's signature defect. So the
//! fallback stays and it announces itself, once per failed asset, at load.
//!
//! ## Why the CHEAP query
//!
//! This uses [`missing_assets`](re_level::Level::missing_assets) (pure path
//! resolution — a few `Path::exists` calls) and NOT
//! [`audit_assets`](re_level::Level::audit_assets) or `build_map_checked`, both
//! of which re-decode every image from scratch to answer the question. The load
//! path already decodes each texture exactly once inside `build_map`; paying for
//! a second full decode to produce a log line would double the cost of loading a
//! textured level. The linter pays that price, the game does not — so a corrupt
//! (as opposed to absent) image is `lvlcheck`'s catch, not this one's.
//!
//! Pure: [`missing_asset_lines`] formats, [`warn_missing_assets`] is the one
//! thin `eprintln!` wrapper over it, so the wording is unit-testable without
//! capturing a stream.

use re_level::Level;

/// The stderr report for `level`'s missing assets.
///
/// A header line, then one entry per failed asset — each entry itself multi-line,
/// because it names every candidate path that was probed, which is the part that
/// makes the message actionable.
///
/// EMPTY when every asset resolves — a clean level says nothing at all.
///
/// `label` is how the level should be named in the message (the path the game
/// loaded it from).
#[must_use]
pub fn missing_asset_lines(level: &Level, label: &str) -> Vec<String> {
    let issues = level.missing_assets();
    if issues.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(issues.len() + 1);
    out.push(format!(
        "terminal-state: level '{label}' is missing {} asset file(s) — it will still load, \
         rendering those surfaces untextured/procedural:",
        issues.len()
    ));
    out.extend(issues.iter().map(|i| format!("  {i}")));
    out
}

/// Print [`missing_asset_lines`] to stderr. Call once per level load, right after
/// the level parses. A level with no missing assets prints nothing.
pub fn warn_missing_assets(level: &Level, label: &str) {
    for line in missing_asset_lines(level, label) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn level_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../levels")
            .join(name)
    }

    /// A level whose assets are all present says NOTHING — the notice is for
    /// defects only, so a normal launch is not noisier than it was.
    #[test]
    fn a_healthy_level_prints_nothing() {
        let lv = re_level::load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        assert!(!lv.asset_refs().is_empty(), "TX5 does reference an asset");
        assert!(
            missing_asset_lines(&lv, "levels/TX5-genplate.lvl").is_empty(),
            "a level with all its assets is silent"
        );
    }

    /// THE DEFECT, made audible: a missing texture is named, the level is named,
    /// the fallback is stated, and EVERY candidate path that was probed is
    /// listed — which is what turns "it looks wrong" into "it is looking here".
    #[test]
    fn a_missing_texture_is_named_with_every_path_probed() {
        let mut lv = re_level::load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        lv.textures[0].file = "assets/texture/genplate_tex_GONE.png".into();
        let lines = missing_asset_lines(&lv, "levels/TX5-genplate.lvl");
        assert_eq!(lines.len(), 2, "header + one asset: {lines:?}");
        assert!(lines[0].contains("levels/TX5-genplate.lvl"), "{}", lines[0]);
        assert!(lines[0].contains("missing 1 asset"), "{}", lines[0]);
        assert!(
            lines[0].contains("still load"),
            "the header states the fallback, not a failure: {}",
            lines[0]
        );
        let body = &lines[1];
        assert!(body.contains("[texture genplate]"), "{body}");
        assert!(body.contains("genplate_tex_GONE.png"), "{body}");
        assert!(body.contains("looked in"), "{body}");
        // Three anchors: beside the level file, beside the level tree, the cwd.
        assert_eq!(
            body.matches("genplate_tex_GONE.png").count(),
            5,
            "the authored path, the message, and all three candidates: {body}"
        );
    }

    /// Every missing asset is reported, not just the first — `build_map_checked`
    /// stops at one, which would have hidden the second and third.
    ///
    /// Uses a healthy, asset-free level (`TX5-genplate.lvl` already carries its own
    /// present texture) and injects a sky + a vox reference that are not authored in
    /// the file at all — [`missing_assets`](re_level::Level::missing_assets) reads
    /// the in-memory `Level`, not the source text, so this exercises the same
    /// multi-miss reporting the deleted `CYBER-sky.lvl` fixture used to.
    #[test]
    fn every_missing_asset_is_reported() {
        let mut lv = re_level::load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        lv.sky.cylinder = Some("assets/texture/gone_cylinder.png".into());
        lv.sky.cap_top = Some("assets/texture/gone_cap.png".into());
        lv.voxes.push(re_level::LevelVox {
            name: "gone".into(),
            file: "levels/GoneBuilding.vox".into(),
            ..Default::default()
        });
        let lines = missing_asset_lines(&lv, "levels/TX5-genplate.lvl");
        assert_eq!(lines.len(), 4, "header + three assets: {lines:?}");
        let joined = lines.join("\n");
        for expected in ["gone_cylinder.png", "gone_cap.png", "GoneBuilding.vox"] {
            assert!(joined.contains(expected), "{expected} not in:\n{joined}");
        }
    }
}
