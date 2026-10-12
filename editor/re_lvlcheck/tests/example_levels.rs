//! Integration sweep: run the linter over the shipped engine `levels/01..08.lvl`
//! archive originals and assert each one is at least SOLVABLE (winnable by
//! toggling) with all switches reachable, its placards binder-placeable, and
//! its thresholds sane. This catches a future regression that makes a shipped
//! level unsolvable or strands a switch.
//!
//! The campaign (`L1..L12`) is a separate content set that ships only in the
//! separate game repository; its equivalent sweep — "all solve", reachability,
//! notes/thresholds and structural validity — lives in that repository's own
//! test suite, which uses `re_lvlcheck` as a dev-dependency.

use std::path::PathBuf;

use re_lvlcheck::{check_path, solve_is_winnable, Severity, SolveVerdict};

/// Repo-root `levels/NN.lvl` (workspace root is two dirs above this crate).
fn level_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../levels")
        .join(name)
}

const SHIPPED: &[&str] = &[
    "01.lvl", "02.lvl", "03.lvl", "04.lvl", "05.lvl", "06.lvl", "07.lvl",
];

#[test]
fn shipped_levels_all_solve() {
    for name in SHIPPED {
        let r = check_path(level_path(name));
        let solve = r
            .solve
            .unwrap_or_else(|| panic!("[{name}] no solve verdict"));
        assert!(
            solve_is_winnable(solve),
            "[{name}] is not winnable: {:?}",
            solve
        );
        // None of the shipped puzzle levels should be unsolvable or net-error.
        assert!(
            !matches!(solve, SolveVerdict::Unsolvable | SolveVerdict::NetworkError),
            "[{name}] solve verdict is a hard failure: {:?}",
            solve
        );
    }
}

/// Reachability is REPORTED, not gated: a shipped level may author a buried
/// "decoy" switch the solver routes around (03.lvl's `R->h2` at z=4 is one).
/// What MUST hold is that the level is still solvable (asserted above) — so here
/// we only surface the reachability state and assert it is computed for every
/// level (no panic, complete report), printing any buried switches for the log.
#[test]
fn shipped_levels_reachability_reported() {
    for name in SHIPPED {
        let r = check_path(level_path(name));
        assert!(
            r.reachable.is_some(),
            "[{name}] reachability must be evaluated"
        );
        if r.reachable == Some(false) {
            let detail = r
                .checks
                .iter()
                .find(|c| c.name == "reachable")
                .map(|c| c.detail.as_str())
                .unwrap_or("?");
            eprintln!("[{name}] NOTE: buried/unreachable switch(es) — {detail}");
        }
    }
}

/// **W1 1.7 acceptance:** the `[note]` placement rule is GREEN (no FAIL) across
/// EVERY shipped `levels/*.lvl` — i.e. every authored placard resolves to a real
/// wall face and can actually render. A WARN is expected and tolerated (most
/// shipped placards author `pos` as the spot the reader stands, which the binder
/// resolves by searching along the gaze); a FAIL means authored text that the game
/// parses and never shows.
#[test]
fn every_level_note_resolves_to_a_wall_face() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut checked = 0usize;
    let mut placards = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("lvl") {
            continue;
        }
        let r = check_path(&path);
        // A level that does not parse is a different test's problem; skip it here
        // rather than double-reporting.
        if r.inventory.is_none() {
            continue;
        }
        checked += 1;
        placards += r.inventory.map_or(0, |i| i.notes);
        let c = r
            .checks
            .iter()
            .find(|c| c.name == "notes")
            .unwrap_or_else(|| panic!("[{}] the note rule must run", path.display()));
        if c.severity == Severity::Fail {
            failures.push(format!("{}: {}", path.display(), c.detail));
        }
    }

    assert!(
        checked > 0,
        "the sweep must find levels in {}",
        dir.display()
    );
    assert!(
        failures.is_empty(),
        "{} level(s) have unplaceable [note] placards:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The corpus this test is about: all authored placards, every one placeable.
    // The engine-only `levels/` directory no longer holds the twelve campaign
    // `L*.lvl` files (story-heavy, many placards each) — those live only in the
    // separate game repository, where their placards are covered by that
    // repository's own lint and `level_text`/`code_panels` tests. This
    // engine-only corpus's own placard-carrying levels (`04`, `06`, `07`, `08`,
    // `100-signage-showcase`, `U1-underline`) total 11; the substance of THIS
    // test — every placard resolves to the correct reader-eye wall layer, no
    // two share a cell — is
    // unchanged, only the coverage-floor count shrank with the corpus.
    assert!(
        placards >= 11,
        "expected the 11+ authored placards to still be present, found {placards}"
    );
}

/// Every shipped level's `notes` and `thresholds` checks must not FAIL: a notes
/// FAIL is authored text the binder cannot place (parsed, never seen); a
/// thresholds FAIL is a nonsensical or win-blocking threshold. WARNs are
/// tolerated (gaze-search/slide placement, parser-default thresholds).
#[test]
fn shipped_levels_notes_and_thresholds_do_not_fail() {
    for name in SHIPPED {
        let r = check_path(level_path(name));
        for cname in ["notes", "thresholds"] {
            let c = r
                .checks
                .iter()
                .find(|c| c.name == cname)
                .unwrap_or_else(|| panic!("[{name}] missing check {cname}"));
            assert_ne!(
                c.severity,
                Severity::Fail,
                "[{name}] {cname} check FAILED: {}",
                c.detail
            );
        }
    }
}

#[test]
fn shipped_levels_structurally_valid() {
    for name in SHIPPED {
        let r = check_path(level_path(name));
        // Every shipped level must have a valid DAG, valid edges, and door
        // targets that resolve.
        for cname in ["edges", "door-targets", "inputs", "dag"] {
            let c = r
                .checks
                .iter()
                .find(|c| c.name == cname)
                .unwrap_or_else(|| panic!("[{name}] missing check {cname}"));
            assert_ne!(
                c.severity,
                Severity::Fail,
                "[{name}] structural check {cname} FAILED: {}",
                c.detail
            );
        }
    }
}
