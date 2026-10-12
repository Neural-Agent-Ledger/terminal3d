//! `lvlcheck` — the STATIC LEVEL-CONTENT linter CLI for terminal3d `.lvl` files.
//!
//! Usage:
//!
//! ```text
//! lvlcheck levels/*.lvl            # lint each file
//! lvlcheck levels/                 # lint every *.lvl in a directory
//! lvlcheck levels/01.lvl levels/   # mix files and dirs
//! ```
//!
//! Prints a per-level report (every check, PASS/WARN/FAIL with specifics) plus a
//! summary table across all levels. Exits non-zero if ANY level FAILs, so it can
//! gate CI.
//!
//! Pure analysis: no rendering, no terminal control — just text on stdout.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use re_lvlcheck::{check_path, Check, LevelReport, Location, Severity, SolveVerdict};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: lvlcheck <level.lvl | dir> [more...]");
        eprintln!(
            "  lints terminal3d .lvl content (elements, solvability, reachability, coherence)"
        );
        return ExitCode::from(2);
    }

    let paths = collect_lvl_paths(&args);
    if paths.is_empty() {
        eprintln!("lvlcheck: no .lvl files found in the given arguments");
        return ExitCode::from(2);
    }

    let reports: Vec<LevelReport> = paths.iter().map(check_path).collect();

    for r in &reports {
        print_report(r);
    }

    print_summary(&reports);

    let any_fail = reports.iter().any(LevelReport::failed);
    if any_fail {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Expand the CLI arguments into a sorted, de-duplicated list of `.lvl` files.
/// A directory argument contributes every `*.lvl` directly inside it.
fn collect_lvl_paths(args: &[String]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for a in args {
        let p = Path::new(a);
        if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(p) {
                for e in entries.flatten() {
                    let path = e.path();
                    if path.extension().and_then(|x| x.to_str()) == Some("lvl") {
                        out.push(path);
                    }
                }
            }
        } else {
            // A file (or a glob already expanded by the shell).
            out.push(p.to_path_buf());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Print one level's full report.
fn print_report(r: &LevelReport) {
    println!("════════════════════════════════════════════════════════════════════");
    let name = if r.level_name.is_empty() {
        r.source.clone()
    } else {
        format!("{} ({})", r.level_name, r.source)
    };
    println!("LEVEL  {name}");
    if !r.title.is_empty() {
        println!("TITLE  {}", r.title);
    }
    if r.size != (0, 0, 0) {
        println!("ROOM   {}×{}×{}", r.size.0, r.size.1, r.size.2);
    }
    println!("────────────────────────────────────────────────────────────────────");
    for c in &r.checks {
        // The check LINE keeps its exact historical shape - two-space indent,
        // bracketed tag, 14-wide name, detail - because something out there may
        // be parsing it. A located check adds a CONTINUATION line beneath it
        // rather than reshaping the line above.
        println!("  [{}] {:<14} {}", c.severity.tag(), c.name, c.detail);
        if let Some(at) = location_line(c) {
            println!("{at}");
        }
    }
    println!("────────────────────────────────────────────────────────────────────");
    println!("  VERDICT  {}", r.verdict);
    println!("  WORST    {}", r.worst().tag());
    println!();
}

/// Print the cross-level summary table.
fn print_summary(reports: &[LevelReport]) {
    println!("════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        "{:<26} {:>5} {:>4} {:<14} {:>9} {:>9} {:<9} worst",
        "level", "nodes", "sw", "solvable?", "min-tog", "reach?", "coherent"
    );
    println!("────────────────────────────────────────────────────────────────────");
    for r in reports {
        let label = short_label(r);
        let nodes = r
            .inventory
            .map(|i| i.nodes_total.to_string())
            .unwrap_or_else(|| "-".into());
        let sw = r
            .inventory
            .map(|i| i.switches.to_string())
            .unwrap_or_else(|| "-".into());
        let (solvable, mintog) = solve_cells(r.solve);
        let reach = match r.reachable {
            Some(true) => "yes",
            Some(false) => "NO",
            None => "-",
        };
        let coherent = match r.coherent {
            Some(true) => "yes",
            Some(false) => "NO",
            None => "-",
        };
        println!(
            "{:<26} {:>5} {:>4} {:<14} {:>9} {:>9} {:<9} {}",
            label,
            nodes,
            sw,
            solvable,
            mintog,
            reach,
            coherent,
            r.worst().tag()
        );
    }
    println!("────────────────────────────────────────────────────────────────────");
    let n = reports.len();
    let fails = reports
        .iter()
        .filter(|r| r.worst() == Severity::Fail)
        .count();
    let warns = reports
        .iter()
        .filter(|r| r.worst() == Severity::Warn)
        .count();
    let pass = n - fails - warns;
    println!("{n} level(s): {pass} PASS · {warns} WARN · {fails} FAIL");
    if fails > 0 {
        println!("RESULT: FAIL — {fails} level(s) are not coherent playable puzzles (see above).");
    } else if warns > 0 {
        println!("RESULT: PASS-WITH-WARNINGS — all levels solvable; {warns} need layout work.");
    } else {
        println!("RESULT: PASS — every level is a coherent, solvable puzzle.");
    }
}

/// How many locations a check prints before it starts saying "+N more". A
/// pipe-eaten `[emit]` figure can lose a dozen cells and the point of the line is
/// to be readable at a glance; the full list is in `Check::locations` for a
/// consumer that wants to walk it.
const MAX_LOCATIONS_SHOWN: usize = 6;

/// The `↳ at …` continuation line for a located check, or `None` when the check
/// is level-wide (then nothing is printed at all and the output is byte-identical
/// to what it always was).
fn location_line(c: &Check) -> Option<String> {
    if c.locations.is_empty() {
        return None;
    }
    let shown: Vec<String> = c
        .locations
        .iter()
        .take(MAX_LOCATIONS_SHOWN)
        .map(Location::to_string)
        .collect();
    let more = c.locations.len().saturating_sub(shown.len());
    let tail = if more > 0 {
        format!(" · +{more} more")
    } else {
        String::new()
    };
    Some(format!("         ↳ at {}{tail}", shown.join(" · ")))
}

/// A compact summary label: the file stem (e.g. `01`, `00-test-load`).
fn short_label(r: &LevelReport) -> String {
    Path::new(&r.source)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| r.source.clone())
}

/// Map a solve verdict into (solvable-cell, min-toggle-cell) for the table.
fn solve_cells(v: Option<SolveVerdict>) -> (String, String) {
    match v {
        Some(SolveVerdict::Solvable { min_toggles, .. }) => ("yes".into(), min_toggles.to_string()),
        Some(SolveVerdict::TriviallySolved) => ("trivial".into(), "0".into()),
        Some(SolveVerdict::Unsolvable) => ("NO".into(), "-".into()),
        Some(SolveVerdict::TooManyToBruteForce { .. }) => ("too-many".into(), "?".into()),
        Some(SolveVerdict::NetworkError) => ("net-error".into(), "-".into()),
        None => ("-".into(), "-".into()),
    }
}
