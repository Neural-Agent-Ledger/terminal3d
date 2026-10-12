//! The ENGINE half of the golden-framebuffer corpus.
//!
//! These 23 keys are the engine/example and archive levels — the ones a
//! second game built on this engine would also want pinned. A host game's own
//! campaign story levels are checked by its own corpus against its own store
//! file, calling the same [`re_golden::check_corpus`].
//!
//! All the render/hash/classify/report logic lives in [`re_golden::check_corpus`];
//! this file supplies only the corpus (which levels, which store) — see that
//! function's doc for the fail-closed accounting rules.
//!
//! # Refresh procedure
//!
//! ```sh
//! TS_BLESS_GOLDEN=1 cargo test -p re_golden --test engine_corpus
//! ```
//! Commit the updated `tests/golden/engine_fb_hashes.txt` together with the
//! render change and a dated note in `tests/golden/README.md`. Never bless to
//! silence an unexpected failure — investigate first.

use std::path::PathBuf;

/// Return the workspace root: `CARGO_MANIFEST_DIR` (which points to
/// `crates/re_golden`) stepped up two levels to the repo root.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// The engine/example/archive half of the curated level list (23 keys). A
/// host game's own campaign story keys are curated separately, in its own
/// test file.
fn curated_levels() -> Vec<(&'static str, PathBuf)> {
    let root = workspace_root();

    macro_rules! lvl {
        ($key:literal, $file:literal) => {
            ($key, root.join("levels").join($file))
        };
    }

    vec![
        // ── Archive levels (the shared `levels/01.lvl`..`08.lvl` tree) ──────
        lvl!("01", "01.lvl"),
        lvl!("02", "02.lvl"),
        lvl!("03", "03.lvl"),
        lvl!("04", "04.lvl"),
        lvl!("05", "05.lvl"),
        lvl!("06", "06.lvl"),
        lvl!("07", "07.lvl"),
        lvl!("08", "08.lvl"),
        // ── Material / texture demos ────────────────────────────────────────
        lvl!("08-material-demo", "08-material-demo.lvl"),
        lvl!("09-code-wall", "09-code-wall.lvl"), // flowing code-wall text
        lvl!("10-calibration-vault", "10-calibration-vault.lvl"),
        lvl!("10-hologram-demo", "10-hologram-demo.lvl"),
        lvl!("TX5-genplate", "TX5-genplate.lvl"),
        // ── Feature anchors ─────────────────────────────────────────────────
        lvl!("U1-underline", "U1-underline.lvl"), // underline substyles
        lvl!("A1-zone-materials", "A1-zone-materials.lvl"), // overlay zones
        lvl!("A2-lines", "A2-lines.lvl"),         // overlay line strokes
        lvl!("A5-inset", "A5-inset.lvl"),         // overlay insets
        lvl!("97-semantic-surfaces", "97-semantic-surfaces.lvl"), // semantic materials
        lvl!("92-fractal-portal", "92-fractal-portal.lvl"), // fractal + portal
        lvl!("21-emitter-showcase", "21-emitter-showcase.lvl"), // particle emitters
        // ── Multi-floor staircase demo levels ───────────────────────────────
        lvl!("20-stairs-straight", "20-stairs-straight.lvl"), // straight 45° staircase
        lvl!("21-stairs-spiral", "21-stairs-spiral.lvl"),     // spiral staircase, multi-floor
        lvl!("22-tower", "22-tower.lvl"), // three-storey tower, two PosX staircases
    ]
}

#[test]
fn engine_golden_framebuffer_hashes() {
    let store = workspace_root().join("tests/golden/engine_fb_hashes.txt");
    re_golden::check_corpus(&store, &curated_levels());
}
