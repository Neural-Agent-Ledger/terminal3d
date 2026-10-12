//! Save the edited level to .lvl, capturing painted geometry on the way out.
//!
//! Serializing `state.level` directly would carry only the *procedural*
//! geometry (`[room]`/`[part]`/`[vox]`) and lose every freely painted cell. So
//! the save path first captures the live `Map` into a dense `[cells]` grid
//! (`CellGrid::from_map`) and attaches it to the level, so `to_lvl_text` emits
//! a `[cells]` section that round-trips the exact painted geometry (`re_level`
//! stamps `[cells]` LAST / authoritative on reload).
//!
//! # Save post-passes (text transforms over the serialized `.lvl`)
//!
//! The serialized text then flows through pure text post-passes — the seam
//! [`rewrite_spline_pipes`] established:
//! 1. `rewrite_spline_pipes` — re-emit editor-flagged spline pipes.
//! 2. [`reattach_preserved`] — re-emit the `#` comments and the
//!    verbatim `[map]`/`[win_when]` blocks captured at load
//!    ([`crate::open::capture_preserved`]), so an open+save never silently
//!    drops a comment or a parseable-but-untyped section.
//!
//! ## Trailing inline comments: CAPTURE + RE-ATTACH
//!
//! Trailing inline comments (`speed = 2  # fast blink`) are captured keyed to
//! `(section-header, key)` and re-attached to the matching emitted line —
//! NOT scoped out via a named exception list. Rationale: an exception list
//! would already be stale (no shipped level carries a genuine trailing
//! comment today — every ` #` in the corpus is a `#RRGGBB` color value, which
//! `re_level`'s `strip_comment` explicitly does NOT treat as a comment and
//! neither do we), and any authoring pass may write a comment against any key
//! in any level, so key-anchored re-attachment is the only shape that stays
//! correct as content evolves. A trailing comment whose line the writer no
//! longer emits (key dropped/renamed) is demoted to a full-line comment in the
//! file-tail leftovers — never silently dropped.
//!
//! ## Reparse guard
//!
//! [`save_lvl_text`] re-parses the EXACT output text via `re_level::load_string`
//! BEFORE writing. On a parse error the original file is REFUSED (never
//! overwritten with unreadable text) and the rejected text is kept in a
//! `<path>.bak` sidecar (an escape hatch — work is never permanently
//! unsavable), with the parse error in the returned status verdict.
//!
//! ## The write itself is crash-safe
//!
//! Having proved the text parses, the save must not lose the file to a failure
//! DURING the write. `std::fs::write` truncates its destination first, so a
//! disk-full / network-drop / antivirus-lock partway through left the original
//! truncated AND the edit unwritten. Every real save now goes through
//! [`write_atomic`] (temp file + rename, the same shape [`write_autosave`]
//! always used), so the destination only ever holds its complete old contents or
//! its complete new contents, and [`backup_previous`] keeps one generation of
//! the replaced bytes in `<path>.bak`.
use crate::state::{EditorState, PreservedText, TrailKey};

/// Capture the painted `Map` into `level.cells` and serialize the level to `.lvl`
/// text, returning the text. SERIALIZE-ONLY: it does NOT write to disk and does
/// NOT touch the dirty flag — it is the shared first half of the save path and
/// the round-trip used by tests. The actual disk write + dirty-clear lives in
/// [`save_lvl_text`] (which has a destination path).
pub fn capture_and_serialize(state: &mut EditorState) -> String {
    // Capture the live geometry so the save is lossless — but BAKE it only when
    // it is actually needed (see `should_bake_cells`). Baking unconditionally
    // silently flattened every procedural level the editor merely opened.
    let mut captured = re_level::CellGrid::from_map(&state.map);
    let bake = should_bake_cells(state, &captured);
    if bake {
        // `[prop]` output is re-expanded on every load, so it must never be baked
        // into `[cells]`: one save would otherwise duplicate
        // every prop and the next load would stamp over itself. Stripped against
        // the level AS LOADED (its `cells` not yet replaced), after the bake
        // decision so the divergence test still compares like with like.
        re_level::props::strip_prop_cells(&state.level, &mut captured);
    }
    state.level.cells = bake.then_some(captured);
    let text = re_voximport::to_lvl_text(&state.level);
    // The shared serializer always emits a pipe's points as `path = …` (the
    // `re_level::Pipe` model has no spline flag). Re-emit them as `spline = …`
    // for pipes the editor flagged spline-mode (`state.spline_pipes`). Exactly one
    // of `path`/`spline` is written per pipe because we REWRITE the existing
    // `path =` line in place — never add a second line.
    let text = rewrite_spline_pipes(&text, state);
    // Re-emit the `#` comments and verbatim `[map]`/`[win_when]`
    // blocks captured at load. Runs AFTER the spline rewrite so a trailing
    // comment captured on a `spline =` line re-attaches to the rewritten line.
    reattach_preserved(&text, &state.preserved)
}

/// Should this save write an explicit `[cells]` block?
///
/// ## The bug this closes
///
/// Every save used to bake `[cells]` unconditionally, and `had_cells_at_load` —
/// recorded at open precisely so the save could refuse to flatten a procedural
/// level — was never consulted by any code path, only by tests. So merely OPENING
/// a procedural level and pressing save froze its whole grid into a dense block.
/// That is irreversible from the editor's side: `re_level` stamps `[cells]` LAST
/// and authoritative, so from then on every `[room]`/`[part]`/`[vox]` edit is
/// silently overwritten by the baked grid.
///
/// ## The rule
///
/// Bake when EITHER:
/// * the file already carried a `[cells]` section (`had_cells_at_load`) — it is
///   already an explicit-geometry level and must stay one; or
/// * the live map DIVERGES from the procedural build recorded at load
///   (`preview_baseline`) — the author has actually painted something, and a
///   hand-painted cell has nowhere else to live.
///
/// A level with no baseline (a fresh `new_empty` document, which has no
/// procedural source at all) always bakes: there is nothing else to persist its
/// geometry.
///
/// Comparing `CellGrid::from_map` on both sides compares EXACTLY the bytes the
/// bake would write, so "the bake would be a no-op" and "don't bake" are the same
/// question. `preview_baseline` is set to a clone of the freshly-built map in
/// `open::load_path`, so an open→save with no edits is bit-identical by
/// construction and never bakes.
///
/// Side benefit: a procedural level that never grows a `[cells]` block never
/// grows the verbatim body that swallows the following section's comment (see
/// `open::recover_verbatim_tail`) — this removes that trap's cause, not just its
/// symptom.
fn should_bake_cells(state: &EditorState, captured: &re_level::CellGrid) -> bool {
    if state.had_cells_at_load {
        return true;
    }
    state
        .preview_baseline
        .as_ref()
        .is_none_or(|base| re_level::CellGrid::from_map(base) != *captured)
}

/// Re-emit the [`PreservedText`] captured at load into the serialized `.lvl`
/// text. Pure text transform, the same post-pass seam as
/// [`rewrite_spline_pipes`]; with nothing preserved it is the identity.
///
/// - A comment block anchored BEFORE a section header is inserted immediately
///   above the first emitted line matching that exact header string; a block
///   anchored to a body key is inserted above the first matching `key = …` line
///   of that section. Anchors are consumed in order, so duplicate headers
///   (several `[trigger]` sections) resolve by file order.
/// - A trailing comment re-attaches to the end of its matching emitted line
///   (see the module docs' capture-and-re-attach section).
/// - Anything that finds no matching line (section/key deleted or renamed
///   without updating anchors — see `NOTE(rename)` in `crate::state`) is
///   appended at the end of the file as full-line comments: NEVER silently
///   dropped, and always parser-safe.
/// - The verbatim `[map]`/`[win_when]` raw blocks are appended last (their
///   position is irrelevant to the parser, which skips them wholesale).
///
/// Key-matching is suppressed inside emitted `[cells]`/`[part]`/`[note]`
/// bodies: no anchors are ever captured for those bodies, and their
/// glyph/text rows must not be key-matched either.
fn reattach_preserved(text: &str, p: &PreservedText) -> String {
    if p.is_empty() {
        return text.to_string();
    }
    let mut anchor_used = vec![false; p.anchors.len()];
    let mut trail_used = vec![false; p.trailing.len()];
    let mut out = String::with_capacity(text.len() + 256);
    // The trimmed header of the section currently being emitted.
    let mut cur_header = String::new();
    // Inside a [cells]/[part]/[note] body of the OUTPUT: no key matching.
    let mut in_verbatim = false;
    // Inside a `[note]` body specifically — where a bracketed line is a header
    // only in COLUMN 0 (parser parity, see `open::capture_preserved`).
    let mut in_note_body = false;

    for line in text.lines() {
        let t = line.trim();
        // Parser parity: an INDENTED bracketed line inside a `[note]` body is
        // placard TEXT, not a header. Reading it as one here would end the note
        // early, clear `in_verbatim`, and start key-matching real placard prose.
        let is_header = t.len() >= 2
            && t.starts_with('[')
            && t.ends_with(']')
            && (!in_note_body || line.starts_with('['));
        if is_header {
            cur_header = t.to_string();
            let interior = t[1..t.len() - 1].trim();
            let kw = interior.split_whitespace().next().unwrap_or("");
            in_verbatim = matches!(kw, "cells" | "part" | "note");
            in_note_body = kw == "note";
            // Leading comment block for this header (first unconsumed match).
            if let Some(i) = p
                .anchors
                .iter()
                .enumerate()
                .position(|(i, a)| !anchor_used[i] && a.key.is_none() && a.header == cur_header)
            {
                anchor_used[i] = true;
                for l in &p.anchors[i].lines {
                    out.push_str(l);
                    out.push('\n');
                }
            }
            out.push_str(line);
            // Trailing comment on the header line itself.
            if let Some(i) = p.trailing.iter().enumerate().position(|(i, tc)| {
                !trail_used[i] && tc.key == TrailKey::Header && tc.header == cur_header
            }) {
                trail_used[i] = true;
                out.push_str(&p.trailing[i].suffix);
            }
            out.push('\n');
            continue;
        }
        if !in_verbatim && !t.is_empty() && !t.starts_with('#') {
            let key = t
                .split_once('=')
                .map_or_else(|| t.to_string(), |(k, _)| k.trim().to_string());
            // Interior comment block anchored above this key line.
            if let Some(i) = p.anchors.iter().enumerate().position(|(i, a)| {
                !anchor_used[i] && a.header == cur_header && a.key.as_deref() == Some(key.as_str())
            }) {
                anchor_used[i] = true;
                for l in &p.anchors[i].lines {
                    out.push_str(l);
                    out.push('\n');
                }
            }
            out.push_str(line);
            // Trailing comment re-attached to this line.
            if let Some(i) = p.trailing.iter().enumerate().position(|(i, tc)| {
                !trail_used[i]
                    && tc.header == cur_header
                    && matches!(&tc.key, TrailKey::Key(k) if *k == key)
            }) {
                trail_used[i] = true;
                out.push_str(&p.trailing[i].suffix);
            }
            out.push('\n');
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }

    // LEFTOVERS — never silently drop a captured comment. Unmatched blocks and
    // trailing suffixes become full-line comments at the end of the file
    // (parser-safe), followed by the file-tail comments and the raw blocks.
    let mut extras: Vec<String> = Vec::new();
    for (i, a) in p.anchors.iter().enumerate() {
        if !anchor_used[i] {
            extras.extend(a.lines.iter().cloned());
        }
    }
    for (i, tc) in p.trailing.iter().enumerate() {
        if !trail_used[i] {
            extras.push(tc.suffix.trim_start().to_string());
        }
    }
    extras.extend(p.tail.iter().cloned());
    if !extras.is_empty() {
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        for l in &extras {
            out.push_str(l);
            out.push('\n');
        }
    }
    for block in &p.raw_blocks {
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str(block);
        out.push('\n');
    }
    out
}

/// Rewrite the `path = …` line to `spline = …` inside every `[pipe A->B]` block
/// whose `(A, B)` pair is in `state.spline_pipes`. Pure text transform over the
/// already-serialized `.lvl`; a pipe NOT in the set is left byte-identical, so a
/// level with no spline pipes round-trips exactly as before.
///
/// Mutual exclusivity is structural: the writer emits exactly one `path =` per
/// pipe and we replace that single line — we never emit both `path` and `spline`.
///
/// ONE-PIPE-PER-PAIR (S2): spline-ness is keyed by `(from, to)`, so if a file
/// somehow held TWO `[pipe A->B]` blocks (the editor never creates these, but
/// `re_level` parses them), only the FIRST is rewritten — a second block with the
/// same endpoints is left as `path = …`. This matches the editor's own model (one
/// pipe per node pair) and avoids silently flagging an unrelated duplicate block.
fn rewrite_spline_pipes(text: &str, state: &EditorState) -> String {
    // Nothing flagged ⇒ identity (and zero allocation churn on the common path).
    if state.spline_pipes.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    // Are we inside a `[pipe A->B]` block flagged spline-mode? (Reset at each
    // section header; a pipe block ends at the next `[` header or EOF.)
    let mut in_spline_pipe = false;
    // `(from, to)` pairs already rewritten — guarantees one rewrite per pair even
    // if a hand-authored file duplicates a pipe block (S2).
    let mut rewritten: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            in_spline_pipe = match pipe_header_pair(trimmed) {
                // Flag only the FIRST not-yet-rewritten block for this pair.
                Some(pair) => state.spline_pipes.contains(&pair) && !rewritten.contains(&pair),
                None => false,
            };
            // Stash the active pair so the rewrite can mark it done.
            if in_spline_pipe {
                if let Some(pair) = pipe_header_pair(trimmed) {
                    rewritten.insert(pair);
                }
            }
        } else if in_spline_pipe {
            // Rewrite ONLY the `path =` key inside a flagged pipe block, preserving
            // the line's leading indentation (pipe blocks are unindented today, but
            // be robust). Once rewritten, stop watching this block's `path`.
            let indent_len = line.len() - trimmed.len();
            if let Some(rest) = trimmed.strip_prefix("path") {
                let after = rest.trim_start();
                if let Some(value) = after.strip_prefix('=') {
                    out.push_str(&line[..indent_len]);
                    out.push_str("spline =");
                    out.push_str(value);
                    out.push('\n');
                    in_spline_pipe = false; // the one path line is consumed
                    continue;
                }
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Parse a `[pipe FROM->TO]` header line into its `(FROM, TO)` node-name pair.
/// Returns `None` for any non-pipe or malformed header. Accepts BOTH the ASCII
/// `->` separator (the only one the editor EMITS) and the unicode `→` that
/// `re_level` also parses (N1), so a hand-authored arrow header is still matched.
fn pipe_header_pair(header: &str) -> Option<(String, String)> {
    let inner = header
        .strip_prefix("[pipe ")
        .and_then(|s| s.strip_suffix(']'))?;
    let (from, to) = inner.split_once("->").or_else(|| inner.split_once('→'))?;
    Some((from.trim().to_string(), to.trim().to_string()))
}

/// Capture the painted `Map` into `level.cells`, serialize to `.lvl`, write the
/// file, validate, and return the lvlcheck verdict line. Mutates `state` (sets
/// `level.cells`, and clears the dirty flag ONLY after a successful disk write).
///
/// REQUIRES a destination: when `state.path` is `None` this writes nothing,
/// leaves `dirty` set, and returns a sentinel verdict — the `save_system` opens
/// the Save-As prompt instead of calling this with no path. (A "no path" save
/// must never claim success.)
pub fn save_lvl_text(state: &mut EditorState) -> String {
    // 1. Capture + serialize the live geometry (no dirty change here).
    let text = capture_and_serialize(state);

    // 2. Without a destination there is nothing to write — refuse (the caller
    //    routes a path-less save to the Save-As prompt). Keep `dirty` set.
    let Some(path) = state.path.clone() else {
        return "no path".to_string();
    };

    // 3. REPARSE GUARD: re-parse the EXACT text we are about to write.
    //    A save must never produce a file the parser rejects (rename cascades,
    //    pass-throughs, comment re-emission — this is the universal backstop).
    //    On Err: REFUSE to touch the original, keep `dirty` set, and keep
    //    the rejected text in a `<path>.bak` sidecar so the work is never
    //    permanently unsavable. (`load_string` is cheap: Parser::run + finalize,
    //    no build_map, no disk.)
    if let Err(e) = re_level::load_string(&text, &path) {
        let bak = format!("{path}.bak");
        return match std::fs::write(&bak, &text) {
            Ok(()) => format!("REFUSED, not written — reparse failed: {e}; text kept at {bak}"),
            Err(we) => {
                format!("REFUSED, not written — reparse failed: {e}; .bak write failed: {we}")
            }
        };
    }

    // 4. Back up ONE generation of the file we are about to replace, then write
    //    ATOMICALLY. A plain `fs::write` TRUNCATES first, so a
    //    disk-full / network-drop / antivirus-lock mid-write left the original
    //    truncated AND the edit unwritten — total loss. `write_atomic` puts the
    //    bytes in `<path>.tmp` and renames over the destination, so a failure
    //    anywhere leaves the ORIGINAL file byte-intact. The `.bak` copy is
    //    best-effort: it is a convenience generation, not the safety mechanism
    //    (the rename is), so a backup that cannot be made never blocks the save.
    //    (`<path>.bak` is the same sidecar step 3 uses for REJECTED text — one
    //    `.bak` per document, always "the previous contents of this file".)
    backup_previous(&path);
    if let Err(e) = write_atomic(&path, &text) {
        return format!("write failed: {e}");
    }

    // 5. Validate, clear dirty (write succeeded), and report. A successful REAL
    //    save supersedes any crash-recovery sidecar for this stem, so delete it
    //    (ignoring errors — a missing sidecar is the common case) and reset the
    //    autosave timer. The FULL report (every check + severity, not
    //    just the verdict line) is captured for the read-only VALIDATION
    //    DRAWER (`V`).
    let report = re_lvlcheck::check_level(&state.level, &path);
    state.last_check = full_report_lines(&report);
    state.dirty = false;
    state.autosave_elapsed = 0.0;
    delete_autosave(&autosave_path_for(Some(&path)));
    // A `[texture]`/`[sky]`/`[vox]`/`[inset]` file that is not on disk is worth a
    // WORD on the status line, because the preview will just render without it and
    // an author cannot tell that apart from "the material is wrong". It is NEVER a
    // refusal: the editor keeps loading and rebuilding best-effort (`vox::rebuild_map`
    // uses `build_map`, not `build_map_checked`) so an asset that is momentarily
    // absent mid-edit cannot lock anyone out of their own level. The full report —
    // every broken path with every candidate probed — is one `V` away in the
    // validation drawer; this is only the pointer to it.
    let assets = asset_status_note(&report);
    let verdict = format!("{}{assets}", report.verdict);
    // A procedural level that just became an EXPLICIT-GEOMETRY one is a one-way
    // door: `re_level` stamps `[cells]` last and authoritative, so from now on the
    // procedural sections are overridden by the baked grid. It needs no
    // confirmation (it only happens once the author has actually painted), but it
    // should not happen mutely either — one line on the status.
    if !state.had_cells_at_load && state.level.cells.is_some() {
        state.had_cells_at_load = true;
        return format!("{verdict} — baked [cells] (now explicit geometry)");
    }
    verdict
}

/// The status-line fragment for missing asset files: empty when every
/// `[texture]`/`[sky]`/`[vox]`/`[inset]` file resolves (the normal case, which
/// must stay silent), otherwise a short count pointing at the drawer.
///
/// Reads the linter's own count rather than re-auditing: `check_level` already
/// resolved and decoded every reference, and doing it twice per save would put a
/// full image decode on the save path for nothing.
fn asset_status_note(report: &re_lvlcheck::LevelReport) -> String {
    match report.broken_assets {
        0 => String::new(),
        n => format!(" — {n} asset file(s) NOT FOUND (V for the paths searched)"),
    }
}

/// CRASH-SAFE write: put `text` in `<dest>.tmp`, then RENAME it over
/// `dest`. The rename is the atomic step — the destination is only ever observed
/// as its full old contents or its full new contents, never as the truncated
/// prefix `fs::write` leaves when the write fails partway. Creates the parent
/// directory when it is missing, and removes the temp file on a failed rename so
/// no `.tmp` litter survives.
///
/// # Errors
///
/// Returns the underlying I/O error of the directory-create, the temp write, or
/// the rename. In every failure case `dest` is left exactly as it was.
fn write_atomic(dest: &str, text: &str) -> std::io::Result<()> {
    if let Some(dir) = std::path::Path::new(dest).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let tmp = format!("{dest}.tmp");
    std::fs::write(&tmp, text)?;
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Keep ONE generation of the file `path` currently holds in `<path>.bak`,
/// so the contents a save replaces are still recoverable. Best-effort:
/// a missing original (first save of a new name) and any copy error are silently
/// ignored — the atomic rename in [`write_atomic`] is what actually protects the
/// file; this is the extra generation on top of it.
fn backup_previous(path: &str) {
    if std::path::Path::new(path).exists() {
        let _ = std::fs::copy(path, format!("{path}.bak"));
    }
}

/// Format the FULL lvlcheck report for the validation drawer: the verdict
/// header, then one `TAG name — detail` line per check. Pure.
fn full_report_lines(report: &re_lvlcheck::LevelReport) -> Vec<String> {
    let mut out = Vec::with_capacity(report.checks.len() + 1);
    out.push(format!("verdict: {}", report.verdict));
    for c in &report.checks {
        out.push(format!("{} {} — {}", c.severity.tag(), c.name, c.detail));
    }
    out
}

/// AUTOSAVE interval: seconds of DIRTY editing between sidecar autosaves.
/// The timer (`EditorState::autosave_elapsed`) only accumulates while the
/// document is dirty and resets on every autosave / real save / clean state.
pub const AUTOSAVE_INTERVAL_SECS: f32 = 60.0;

/// AUTOSAVE SIDECAR path: `levels/.autosave/<stem>-<key>.lvl`, where
/// `<stem>` is the basename of the document's path without its `.lvl` suffix and
/// `<key>` is a hash of its CANONICAL ABSOLUTE path (else plain `untitled.lvl`
/// for a never-saved document, which has no path to key on).
///
/// Keying on the CANONICAL ABSOLUTE path matters: keying on the basename alone
/// would make opening `/other/tree/levels/03.lvl` from this repo write to —
/// and later restore from, and on save/clean-quit delete — this repo's
/// `03.lvl` sidecar, silently mixing the contents of two unrelated files that
/// merely share a name. The stem stays in the name for human legibility; the
/// hash is what makes the path unique per document.
///
/// The path base is fixed and relative (like [`crate::open::scan_lvl_files`]);
/// tests that need a scratch location instead pass an explicit destination to
/// [`write_autosave`].
pub fn autosave_path_for(path: Option<&str>) -> String {
    let Some(path) = path.filter(|p| !file_stem(p).is_empty()) else {
        return "levels/.autosave/untitled.lvl".to_string();
    };
    format!(
        "levels/.autosave/{}-{:016x}.lvl",
        file_stem(path),
        fnv1a64(&canonical_key(path))
    )
}

/// The basename of `path` without its `.lvl` suffix (`""` when there is none).
/// Accepts both separators — a Windows path reaches here as often as a `/` one.
fn file_stem(path: &str) -> &str {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    base.strip_suffix(".lvl").unwrap_or(base)
}

/// The STABLE IDENTITY of a document path for sidecar keying: its
/// canonical absolute form, so two files that merely share a basename can never
/// share a sidecar.
///
/// We canonicalize the PARENT DIRECTORY and re-join the file name rather than
/// canonicalizing the file: the directory exists even before the document's
/// first save, whereas `canonicalize` on the file itself would fail while it does
/// not yet exist and then silently change the key the moment it did — orphaning
/// the sidecar written moments earlier. An uncanonicalizable parent (missing
/// directory) falls back to the cwd-joined path, and finally to the raw path;
/// each fallback is still per-document, never per-basename.
fn canonical_key(path: &str) -> String {
    let p = std::path::Path::new(path);
    let name = p.file_name().unwrap_or(p.as_os_str());
    let parent = match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    let dir = std::fs::canonicalize(&parent)
        .or_else(|_| std::env::current_dir().map(|cwd| cwd.join(&parent)))
        .unwrap_or(parent);
    dir.join(name).to_string_lossy().into_owned()
}

/// FNV-1a 64-bit over `s`. Our own three-line hash rather than
/// `DefaultHasher`, whose output is explicitly not stable across Rust versions —
/// a sidecar key must resolve to the same file next release, or every pending
/// crash recovery is silently orphaned.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Serialize the CURRENT document (dirty state included) and write it to the
/// `dest` autosave sidecar. Runs the SAME reparse guard as a real save —
/// on a parse error NOTHING is written. The write is atomic-ish: the text goes
/// to `<dest>.tmp` first and is then renamed over `dest`, so a crash mid-write
/// never leaves a truncated sidecar. Does NOT touch `state.path`, the dirty
/// flag, or the ORIGINAL file on disk.
///
/// # Errors
///
/// Returns a short reason when the serialized text fails the reparse guard or
/// any of the directory-create / temp-write / rename steps fails. The caller
/// (the autosave tick) surfaces it in the status line and skips the autosave.
pub fn write_autosave(state: &mut EditorState, dest: &str) -> Result<(), String> {
    let text = capture_and_serialize(state);
    // Reparse guard: an autosave the parser rejects is worse than no autosave —
    // a crash recovery must never offer unloadable text.
    if let Err(e) = re_level::load_string(&text, dest) {
        return Err(format!("reparse failed: {e}"));
    }
    // Write-to-temp + rename so the sidecar is never observed half-written —
    // the same [`write_atomic`] the real save uses.
    write_atomic(dest, &text).map_err(|e| format!("autosave write failed: {e}"))
}

/// Delete an autosave sidecar, ignoring every error — a missing sidecar
/// is the common case (clean quit / save with no crash in between). Called from
/// the `save_lvl_text` success path and the clean-quit path in `main.rs`.
pub fn delete_autosave(sidecar: &str) {
    let _ = std::fs::remove_file(sidecar);
}

/// PLAYTEST SIDECAR path: `levels/_scratch/preview-<stem>-<pid>.lvl`,
/// derived from the document's file stem (else a sanitized level name, else
/// `untitled`). The sidecar keeps the playtest OFF the original file — F5 never
/// overwrites the document on disk and never clears the dirty flag.
///
/// ## Why the PID is in the name
///
/// The stem alone is not an identity: two editors open on the same level (or on
/// two same-named levels in different trees) both wrote `preview-<stem>.lvl`,
/// and F5 in one clobbered the file the other had just handed to its game child.
/// The pid makes the sidecar private to one editor process while staying stable
/// ACROSS that process's own F5 presses, so a session still reuses one file
/// instead of leaving one per playtest.
///
/// ## Why the DIRECTORY is still `levels/_scratch/`, CWD-relative
///
/// Asset resolution decides this, and it decides it in favour of leaving it
/// alone. `re_level::assets::level_asset_candidates` probes, for a sidecar at
/// `levels/_scratch/x.lvl`: `levels/_scratch/<rel>`, then `levels/<rel>`, then
/// the bare CWD-relative `<rel>`. EVERY asset reference in EVERY shipped level
/// is content-root-relative (`assets/texture/*.png`, `levels/*.vox`), so only
/// the third candidate — the CWD one — can ever match. Making the sidecar path
/// absolute would therefore not buy CWD-independence at playtest time: the game
/// child inherits the editor's CWD and still resolves its textures through it.
/// It would only scatter `_scratch` directories into whatever directory the
/// author's level happens to live in.
///
/// (There IS a latent defect underneath: a sidecar one level below `levels/`
/// loses the root anchor that `levels/<name>.lvl` itself gets from candidate 2,
/// so a playtest launched with a CWD outside the content root cannot find
/// textures. Fixing that means moving the sidecar to a sibling of `levels/` and
/// re-specifying where playtest files land — a separate change with its own
/// behavioural contract, not a rename.)
pub fn playtest_sidecar_path(path: Option<&str>, level_name: &str) -> String {
    let stem = path
        .and_then(|p| {
            let base = p.rsplit(['/', '\\']).next().unwrap_or(p);
            let base = base.strip_suffix(".lvl").unwrap_or(base);
            if base.is_empty() {
                None
            } else {
                Some(base.to_string())
            }
        })
        .or_else(|| {
            let safe: String = level_name
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if safe.is_empty() {
                None
            } else {
                Some(safe)
            }
        })
        .unwrap_or_else(|| "untitled".to_string());
    format!("levels/_scratch/preview-{stem}-{}.lvl", std::process::id())
}

/// Serialize the CURRENT document (dirty state included) to the playtest
/// sidecar and return its path. Runs the SAME reparse guard as a real
/// save — the game must never be launched on a file it cannot load. Does NOT
/// touch `state.path` or the dirty flag: the original file on disk is left
/// exactly as it was.
pub fn save_playtest_sidecar(state: &mut EditorState) -> Result<String, String> {
    let text = capture_and_serialize(state);
    let sidecar = playtest_sidecar_path(state.path.as_deref(), &state.level.name);
    if let Err(e) = re_level::load_string(&text, &sidecar) {
        return Err(format!("sidecar refused — reparse failed: {e}"));
    }
    // Derive the directory from the path rather than repeating the literal, so
    // the two can never disagree if the layout is ever re-specified.
    if let Some(dir) = std::path::Path::new(&sidecar).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("sidecar dir failed: {e}"))?;
    }
    std::fs::write(&sidecar, &text).map_err(|e| format!("sidecar write failed: {e}"))?;
    Ok(sidecar)
}

/// Test-only recipe-variant label (the parser does not round-trip recipe PARAMS
/// byte-identically for every recipe, so the round-trip asserts on the variant).
#[cfg(test)]
fn recipe_label_for_test(r: Option<re_world::MaterialRecipe>) -> &'static str {
    crate::material::recipe_label(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::Shape;
    use re_world::{CellKind, DiagXY, RampDir};

    /// Saving a never-saved document (path == None) must NOT write and
    /// must NOT clear dirty — `save_lvl_text` refuses with a sentinel verdict and
    /// leaves the work unsaved. (The `save_system` then opens Save-As; that branch
    /// is verified in `saveas.rs::save_with_no_path_opens_saveas_and_keeps_dirty`.)
    #[test]
    fn save_with_no_path_does_not_write_or_clear_dirty() {
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.dirty = true;
        assert!(state.path.is_none());
        let verdict = save_lvl_text(&mut state);
        assert_eq!(
            verdict, "no path",
            "no-path save reports a sentinel, not success"
        );
        assert!(state.dirty, "no-path save keeps the document dirty");
        // It still captured cells (serialize half ran), but wrote nothing.
        assert!(state.level.cells.is_some());
    }

    /// Process- AND call-unique fixture paths under the OS temp dir. Hoisted into
    /// [`crate::testutil`] so `io.rs`, `open.rs`, and `saveas.rs` share ONE
    /// uniqueness scheme instead of three; the full rationale lives there.
    use crate::testutil::scratch_path;

    /// END-TO-END: `[map]`/`[win_when]` blocks, a file-header
    /// comment, and a trailing inline comment all survive a full disk
    /// open → save round-trip, and a second round-trip is BYTE-IDENTICAL
    /// (the re-emission is stable under its own capture).
    #[test]
    fn map_win_when_and_comments_round_trip_through_save() {
        // A canonical serialized document…
        let mut base = EditorState::new_empty(4, 4, 2);
        base.map.set(1, 1, 0, re_world::CellKind::WallSolid);
        let canonical = capture_and_serialize(&mut base);
        // …decorated the way a hand-author would: header comment, trailing
        // comment, and the two Section::Skip blocks the typed model drops.
        let mut text = String::from("# HEADER kept\n\n");
        text.push_str(&canonical);
        let mut text = text.replace("name = untitled", "name = untitled  # named by hand");
        text.push_str("\n# above map\n[map]\nXY\nYX\n\n[win_when]\nall_on = yes\n");

        let path = scratch_path("w15-map-rt.lvl");
        std::fs::write(&path, &text).expect("write fixture");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "fixture loads");
        assert!(
            !state.preserved.is_empty(),
            "comments + raw blocks captured"
        );
        assert_eq!(
            state.preserved.raw_blocks.len(),
            2,
            "[map] + [win_when] held"
        );

        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert!(
            saved.starts_with("# HEADER kept"),
            "header comment re-emitted first"
        );
        assert!(
            saved.contains("name = untitled  # named by hand"),
            "trailing comment re-attached to its key line:\n{saved}"
        );
        assert!(
            saved.contains("# above map\n[map]\nXY\nYX"),
            "[map] block (with its comments) re-emitted verbatim:\n{saved}"
        );
        assert!(
            saved.contains("[win_when]\nall_on = yes"),
            "[win_when] block re-emitted verbatim:\n{saved}"
        );

        // Stability: open + save AGAIN → byte-identical.
        let mut state2 = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state2, &path));
        let _ = save_lvl_text(&mut state2);
        let saved2 = std::fs::read_to_string(&path).expect("read re-saved");
        assert_eq!(saved2, saved, "second open+save is byte-identical");
        let _ = std::fs::remove_file(&path);
        // …and the `backup_previous` generation the second save left beside it.
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// A successful save captures the FULL lvlcheck report — the verdict
    /// header plus one `TAG name — detail` line per check (notes, thresholds,
    /// inventory, …) — for the read-only validation drawer (`V`).
    #[test]
    fn save_captures_full_lvlcheck_report_for_the_drawer() {
        let mut state = EditorState::new_empty(4, 4, 2);
        state.map.set(1, 1, 0, re_world::CellKind::WallSolid);
        let path = scratch_path("w34-report.lvl");
        state.path = Some(path.clone());
        state.mark_dirty();
        assert!(
            state.last_check.is_empty(),
            "no report before the first save"
        );
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        // The DRAWER carries the `re_lvlcheck` report VERBATIM; the returned string
        // is the editor's STATUS line, which may append a note of its own (the
        // one-way `[cells]` bake). So the drawer header is a PREFIX of the status,
        // not necessarily equal to it — asserting equality would silently couple
        // the linter's vocabulary to the editor's.
        let header = state
            .last_check
            .first()
            .cloned()
            .expect("drawer has a header");
        let header_verdict = header
            .strip_prefix("verdict: ")
            .expect("the drawer's first line is the verdict header");
        assert!(
            verdict.starts_with(header_verdict),
            "the status begins with the lvlcheck verdict: status={verdict:?} header={header:?}"
        );
        let joined = state.last_check.join("\n");
        for name in ["notes", "thresholds", "inventory"] {
            assert!(
                joined.contains(name),
                "drawer lists the '{name}' check:\n{joined}"
            );
        }
        assert!(
            state.last_check[1..]
                .iter()
                .all(|l| l.starts_with("PASS") || l.starts_with("WARN") || l.starts_with("FAIL")),
            "every check line carries a severity tag:\n{joined}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Sound-field authoring: `re_lvlcheck`'s `fields` rule findings reach the same
    /// drawer as every other check with NO new plumbing — `check_level` already
    /// runs `fields_rule::check_fields` (`re_lvlcheck/src/lib.rs`) and
    /// `full_report_lines` renders every `Check` verbatim regardless of which
    /// rule produced it. This exercises the WARN path specifically (3
    /// Hold-using fields exceeds `HOLD_FIELD_MAX = 2`) so the field editor's
    /// own lint surface is proven end-to-end, not just inferred from the
    /// generic drawer test above.
    #[test]
    fn a_field_rule_warning_reaches_the_save_drawer() {
        let mut state = EditorState::new_empty(20, 20, 4);
        let path = scratch_path("field-lint-drawer.lvl");
        state.path = Some(path.clone());
        for (i, name) in ["hold_a", "hold_b", "hold_c"].iter().enumerate() {
            state.level.field_defs.push(re_level::fields::FieldDef {
                name: (*name).to_string(),
                line: 0,
                host: re_level::fields::FieldHost::Standalone,
                keys: vec![
                    re_level::fields::FieldKey {
                        key: "kind".into(),
                        value: "mnemos_hold".into(),
                        line: 0,
                    },
                    re_level::fields::FieldKey {
                        key: "at".into(),
                        value: format!("{},2,1", 2 + i * 3),
                        line: 0,
                    },
                ],
            });
        }
        crate::field::re_resolve(&mut state);
        state.mark_dirty();
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        let joined = state.last_check.join("\n");
        assert!(
            joined.contains("WARN fields") && joined.contains("Hold-using"),
            "the fields rule's Hold-count WARN reaches the drawer:\n{joined}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A `[texture]` file that is not on disk is REPORTED on the status line and
    /// in the drawer — and the save still succeeds. The editor stays lenient by
    /// design (`vox::rebuild_map` builds best-effort): an author whose asset is
    /// momentarily absent must never be locked out of their own level, but they
    /// must also not be left wondering why the wall renders untextured.
    #[test]
    fn a_missing_texture_file_is_reported_on_save_but_never_blocks_it() {
        let mut state = EditorState::new_empty(6, 6, 2);
        state.level.textures.push(re_level::LevelTexture {
            name: "pic".into(),
            file: "assets/texture/absent_in_the_editor.png".into(),
            max_size: None,
            height: None,
            ..Default::default()
        });
        let path = scratch_path("assets-missing.lvl");
        state.path = Some(path.clone());
        state.mark_dirty();

        let status = save_lvl_text(&mut state);
        assert!(
            !status.starts_with("REFUSED") && !status.starts_with("write failed"),
            "a missing asset never blocks the save: {status}"
        );
        assert!(
            std::path::Path::new(&path).is_file(),
            "the file was really written"
        );
        assert!(!state.dirty, "a successful save clears dirty");
        assert!(
            status.contains("1 asset file(s) NOT FOUND"),
            "the status line says so: {status}"
        );
        let drawer = state.last_check.join("\n");
        assert!(
            drawer.contains("absent_in_the_editor.png") && drawer.contains("looked in"),
            "the drawer carries the authored path and every candidate probed:\n{drawer}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The counterweight: a level with no broken asset says NOTHING extra, so the
    /// status line of a healthy save is exactly the linter's verdict.
    #[test]
    fn a_clean_save_adds_no_asset_note() {
        let mut state = EditorState::new_empty(6, 6, 2);
        let path = scratch_path("assets-clean.lvl");
        state.path = Some(path.clone());
        state.mark_dirty();
        let status = save_lvl_text(&mut state);
        assert!(
            !status.contains("NOT FOUND"),
            "no asset references ⇒ no note: {status}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// An IMPORTED decal surface — `hologram` / `portal` / `object`, the three kinds
    /// `vox_to_level` mints from `.vox` tags 13/14/15 — survives the whole editor
    /// session: open → edit something ELSE → save → reload, unchanged. Then a full
    /// lap of `c` on the decal's `surface` field returns each one to itself.
    ///
    /// Before `DECAL_SURFACES` carried all six kinds, the second half destroyed the
    /// data: `cycle_decal_surface` fell through its `None` arm and rewrote an
    /// imported surface to `trace` on the FIRST press, with no way back.
    #[test]
    fn imported_decal_surfaces_survive_an_edit_save_reload_and_a_full_cycle_lap() {
        use crate::entity;
        use crate::state::{EntityField, DECAL_SURFACES};
        use re_world::CellKind as CK;

        // A hand-authored document carrying one decal per imported surface kind.
        let mut text = String::from(
            "[meta]\nname = imported-decals\n\n\
             [room]\nsize = 8 8 4\n\n\
             [pedestal]\ncenter = 4 4 0\nvolume = 1 1 1\n\n\
             [door]\npos = 0 4 0\n\n\
             [player]\nspawn = 4 4 0\n\n",
        );
        let imported = [
            ("holo_d", "hologram", CK::Hologram),
            ("port_d", "portal", CK::Portal),
            ("obj_d", "object", CK::Object),
        ];
        for (i, (name, label, _)) in imported.iter().enumerate() {
            let y = 1 + i as i32;
            text.push_str(&format!(
                "[wall_decal {name}]\nsurface = {label}\nregion = (1,{y},0)-(2,{y},0)\n\n"
            ));
        }
        let path = scratch_path("imported-decal-surfaces.lvl");
        std::fs::write(&path, &text).expect("write fixture");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "fixture opens");
        for (name, label, kind) in imported {
            let d = state
                .level
                .decals
                .iter()
                .find(|d| d.name == name)
                .expect(name);
            assert_eq!(d.surface_kind, kind, "'{label}' parsed as {kind:?} at load");
        }

        // Edit something ELSE entirely, then save: the decals are untouched work.
        state.map.set(5, 5, 0, CellKind::WallSolid);
        state.mark_dirty();
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );

        let mut reloaded = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut reloaded, &path),
            "saved level reopens"
        );
        for (name, label, kind) in imported {
            let d = reloaded
                .level
                .decals
                .iter()
                .find(|d| d.name == name)
                .expect(name);
            assert_eq!(
                d.surface_kind, kind,
                "'{label}' must survive open → unrelated edit → save → reload"
            );
        }
        assert_eq!(
            reloaded.map.cell_at(5, 5, 0),
            CellKind::WallSolid,
            "the real edit landed"
        );

        // …and the `c` cycle is now lossless for each of them: one lap returns the
        // surface to itself instead of collapsing it to `trace`.
        reloaded.place = Some(crate::state::EntityKind::WallDecal);
        for (i, (name, label, kind)) in imported.iter().enumerate() {
            let idx = reloaded
                .level
                .decals
                .iter()
                .position(|d| &d.name == name)
                .expect(name);
            reloaded.selected_decal = Some(idx);
            reloaded.entity_field = EntityField::DecalSurface;
            entity::cycle_property(&mut reloaded);
            let first = reloaded.level.decals[idx].surface_kind;
            assert_ne!(
                first, *kind,
                "decal #{i} '{label}' steps to a different surface"
            );
            for _ in 1..DECAL_SURFACES.len() {
                entity::cycle_property(&mut reloaded);
            }
            assert_eq!(
                reloaded.level.decals[idx].surface_kind, *kind,
                "a full `c` lap returns '{label}' to {kind:?}"
            );
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// Disk-side acceptance: a MULTI-LINE note body survives a full
    /// `save_lvl_text` → `load_path` round-trip with its content intact, and a
    /// second save is byte-identical (stable under the comment-reattachment
    /// machinery, which must never comment-scan `[note]` bodies).
    #[test]
    fn multiline_note_body_roundtrips_through_disk_save() {
        let path = scratch_path("w23-note-disk.lvl");
        let mut state = EditorState::new_empty(6, 6, 2);
        state.level.notes.push(re_level::LevelNote {
            kind: re_level::NoteKind::Researcher,
            x: 2,
            y: 2,
            z: 0,
            facing: 0,
            text: "one\nB two\nthree".into(),
        });
        state.path = Some(path.clone());
        state.dirty = true;
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        let mut s2 = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s2, &path),
            "saved level reloads"
        );
        assert_eq!(
            s2.level.notes[0].text.trim_end(),
            "one\nB two\nthree",
            "multi-line body identical after reload"
        );
        let before = std::fs::read_to_string(&path).expect("read");
        let _ = save_lvl_text(&mut s2);
        let after = std::fs::read_to_string(&path).expect("re-read");
        assert_eq!(after, before, "second save is byte-identical");
        let _ = std::fs::remove_file(&path);
        // …and the `backup_previous` generation the second save left beside it.
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// ACCEPTANCE: a `[cells]`-bearing shipped level (scratch copy of
    /// 08-material-demo, which also carries a hand-restored `#` header block)
    /// round-trips BYTE-IDENTICAL through a full disk open → save. The `#` rows
    /// of `[cells]` are GEOMETRY — a naive comment capture would destroy them
    /// while the file still parsed cleanly.
    #[test]
    fn cells_bearing_level_round_trips_byte_identical() {
        // Normalize the fixture to LF: the repo file is LF, but a Windows
        // checkout can materialize CRLF (autocrlf), and the save path emits the
        // parser-canonical LF. Byte-identity is asserted on the content bytes.
        let orig = std::fs::read_to_string("../../levels/08-material-demo.lvl")
            .expect("read 08")
            .replace("\r\n", "\n");
        let path = scratch_path("g1-cells-rt.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "08 copy loads");
        assert!(state.had_cells_at_load, "08 carries [cells]");
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert_eq!(
            saved, orig,
            "a [cells]-bearing level round-trips BYTE-IDENTICAL (comments AND geometry)"
        );
        let _ = std::fs::remove_file(&path);
        // …and the `backup_previous` generation the second save left beside it.
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// An unknown key in the opened file reaches the editor's VALIDATION DRAWER
    /// on save, as a WARN that names the section, line and key.
    ///
    /// This is the last hop of the diagnostic chain: `re_level` collects the key
    /// the parser discarded, `re_lvlcheck` turns it into a check, and the editor
    /// puts it in front of the author. Before it existed, a mistyped key was
    /// dropped in total silence at every step.
    #[test]
    fn an_unknown_key_reaches_the_editor_validation_drawer() {
        let raw = std::fs::read_to_string("../../levels/01.lvl")
            .expect("read 01")
            .replace("\r\n", "\n")
            .replace("ambient     = trace-pattern", "ambeint     = trace-pattern");
        let path = scratch_path("unknown-key-drawer.lvl");
        std::fs::write(&path, &raw).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut state, &path),
            "a typo must not block the open"
        );
        assert_eq!(state.level.unknown_keys.len(), 1, "the parser collected it");

        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "an unknown key must never block the save: {verdict}"
        );
        let joined = state.last_check.join("\n");
        assert!(
            joined.contains("WARN unknown_keys"),
            "the drawer carries the unknown_keys check as a WARN:\n{joined}"
        );
        assert!(
            joined.contains("[room] unknown key 'ambeint'"),
            "…naming section, line and key:\n{joined}"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// A PROCEDURAL level opened and saved with NO edits stays procedural — no
    /// `[cells]` block is baked into it.
    ///
    /// Every save used to bake one unconditionally, and `had_cells_at_load` (set
    /// at open for exactly this purpose) was consulted by nothing but tests. So
    /// opening `01.lvl` and pressing save froze its whole grid — irreversibly,
    /// since `re_level` stamps `[cells]` LAST and authoritative, after which every
    /// `[room]`/`[part]`/`[vox]` edit is silently overwritten by the baked grid.
    #[test]
    fn opening_a_procedural_level_and_saving_does_not_bake_cells() {
        let orig = std::fs::read_to_string("../../levels/01.lvl")
            .expect("read 01")
            .replace("\r\n", "\n");
        let path = scratch_path("bake-procedural.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "01 copy loads");
        assert!(
            !state.had_cells_at_load,
            "01 is procedural — no [cells] on disk"
        );

        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        assert!(
            state.level.cells.is_none(),
            "no [cells] attached to the level"
        );
        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert!(
            !saved.contains("[cells]"),
            "an untouched procedural level must not grow a [cells] block:\n{saved}"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// …but the moment the author PAINTS a cell, the geometry has nowhere else to
    /// live and the bake happens. The gate is "is it needed", not "never".
    #[test]
    fn painting_a_cell_bakes_cells_into_a_procedural_level() {
        let orig = std::fs::read_to_string("../../levels/01.lvl")
            .expect("read 01")
            .replace("\r\n", "\n");
        let path = scratch_path("bake-painted.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "01 copy loads");
        // One hand edit that the procedural build could never produce.
        state.map.set(4, 5, 1, CellKind::WallMatrix);

        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        assert!(state.level.cells.is_some(), "the painted grid is baked");
        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert!(
            saved.contains("[cells]"),
            "the painted level grows a [cells] block"
        );

        // And the edit really survives the round-trip.
        let mut s2 = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s2, &path),
            "the saved file reopens"
        );
        assert!(s2.had_cells_at_load, "it is now an explicit-geometry level");
        assert_eq!(
            s2.map.cell_at(4, 5, 1),
            CellKind::WallMatrix,
            "the painted cell survived"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// A level that ALREADY carried `[cells]` keeps it, even with no edits — it is
    /// an explicit-geometry level and un-baking it would delete its authored grid.
    #[test]
    fn a_cells_bearing_level_keeps_its_cells_with_no_edits() {
        let orig = std::fs::read_to_string("../../levels/08-material-demo.lvl")
            .expect("read 08")
            .replace("\r\n", "\n");
        let path = scratch_path("bake-keeps.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &path), "08 copy loads");
        assert!(state.had_cells_at_load, "08 carries [cells]");
        let _ = save_lvl_text(&mut state);
        assert!(state.level.cells.is_some(), "the [cells] block is kept");
        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert!(saved.contains("[cells]"), "08 keeps its [cells] block");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// The five INERT authoring keys `re_level` parses and `re_voximport` writes
    /// survive a FULL editor open → save → reopen.
    ///
    /// `00-test-load.lvl` is the only shipped level carrying all five at once
    /// (`[meta] version`, `[meta] spec_rev`, `[room] floor`, `[room] ceiling`,
    /// `[door] target_model`). Before the fix the editor deleted every one of them
    /// on save — silently, because the writer emitted none and the text-vs-text
    /// round-trip sweep therefore saw no difference. `spec_rev` in particular is
    /// read straight off disk by `re_game`'s `spec_rev_stays_in_the_real_world`
    /// guard, so erasing it disarmed a live test.
    #[test]
    fn inert_authoring_keys_survive_a_full_open_save() {
        let orig = std::fs::read_to_string("../../levels/00-test-load.lvl")
            .expect("read 00-test-load")
            .replace("\r\n", "\n");
        let path = scratch_path("inert-keys-rt.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut s1 = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s1, &path),
            "00-test-load copy loads"
        );
        // Sanity: the file really does carry all five before we save.
        assert_eq!(s1.level.version, "1");
        assert_eq!(s1.level.spec_rev, "2026-06-15");
        assert_eq!(s1.level.room_floor, "grid");
        assert_eq!(s1.level.room_ceiling, "dim");
        assert_eq!(s1.level.door.target_model, "test");

        let verdict = save_lvl_text(&mut s1);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );

        let mut s2 = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s2, &path),
            "the saved file reopens"
        );
        assert_eq!(s2.level.version, "1", "[meta] version survived the save");
        assert_eq!(
            s2.level.spec_rev, "2026-06-15",
            "[meta] spec_rev survived the save"
        );
        assert_eq!(
            s2.level.room_floor, "grid",
            "[room] floor survived the save"
        );
        assert_eq!(
            s2.level.room_ceiling, "dim",
            "[room] ceiling survived the save"
        );
        assert_eq!(
            s2.level.door.target_model, "test",
            "[door] target_model survived the save"
        );

        // And the LINES are really in the file, not just recovered as comments.
        let saved = std::fs::read_to_string(&path).expect("read saved");
        for line in [
            "version = 1",
            "spec_rev = 2026-06-15",
            "floor = grid",
            "ceiling = dim",
            "target_model = test",
        ] {
            assert!(
                saved.contains(line),
                "saved text must carry `{line}`:\n{saved}"
            );
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    // The shipped `.lvl` corpus enumeration lives in `testutil` so this sweep and
    // `corpus.rs`'s cycle sweep can never end up checking different level sets.
    use crate::testutil::shipped_levels;

    /// CORPUS BYTE-STABILITY — the whole-corpus counterpart to
    /// [`cells_bearing_level_round_trips_byte_identical`], which covers exactly ONE
    /// of the 63 shipped levels.
    ///
    /// ## Why full disk-byte identity CANNOT be asserted for the whole corpus
    ///
    /// `to_lvl_text` has one canonical output form: `key = value`, one blank line
    /// between sections, and (through the editor) an explicit `[cells]` block. Almost
    /// every shipped `.lvl` is hand-authored in a DIFFERENT but equally valid form —
    /// column-aligned `=` (`name        = 01-first-pulse`), its own blank-line
    /// rhythm, and mostly no `[cells]` section at all. Those are formatting choices
    /// the typed `Level` model does not carry, so no writer can reproduce them, and
    /// `08-material-demo.lvl` is byte-identical only because it was ALREADY saved in
    /// canonical form. Demanding disk-byte identity corpus-wide would therefore be a
    /// demand to re-write 62 tracked level files — not a bug to fix.
    ///
    /// ## What IS asserted for every level — and it is the property that matters
    ///
    /// 1. Every shipped level OPENS and SAVES without the reparse guard refusing it.
    /// 2. The save reaches a **byte-stable fixed point**: the bytes after the second
    ///    full open → save are identical to the bytes after the first. This is the
    ///    real anti-drift guarantee — a save that kept mutating the file (comments
    ///    migrating, a body growing a line, a section re-expanding) fails here even
    ///    though its first pass looked fine.
    /// 3. Every level that IS already canonical STAYS canonical (the baseline set
    ///    below). A regression that stops reproducing an already-canonical file's
    ///    bytes fails, naming the level.
    ///
    /// Note on `A5-inset.lvl`: its `[inset] child` path resolves relative to the
    /// level's own directory, so the scratch copy cannot find the child scene. That
    /// changes what `build_map` produces — and therefore the baked `[cells]` — but
    /// not the fixed-point property, which is what this test asserts.
    #[test]
    fn every_shipped_level_is_byte_stable_through_the_editor() {
        // Levels already stored in the writer's canonical form. Growth is welcome
        // (a level canonicalized on purpose just joins the set); SHRINKAGE is the
        // regression this guards, so the assertion is "contains", checked per name.
        const CANONICAL: &[&str] = &["08-material-demo.lvl"];

        let mut failures: Vec<String> = Vec::new();
        let mut canonical_seen: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for (name, src) in shipped_levels() {
            // LF-normalize for the same reason the single-level test does: the repo
            // files are LF but a Windows checkout can materialize CRLF, while the
            // save path always emits LF.
            let orig = std::fs::read_to_string(&src)
                .unwrap_or_else(|e| panic!("read {name}: {e}"))
                .replace("\r\n", "\n");
            let path = scratch_path(&format!("corpus-{name}"));
            std::fs::write(&path, &orig).expect("write scratch copy");
            // `backup_previous` keeps one generation next to the file; clean both.
            let cleanup = |p: &str| {
                let _ = std::fs::remove_file(p);
                let _ = std::fs::remove_file(format!("{p}.bak"));
            };

            let mut s1 = EditorState::new_empty(4, 4, 2);
            if !crate::open::load_path(&mut s1, &path) {
                failures.push(format!("  {name}: the editor could not open it"));
                cleanup(&path);
                continue;
            }
            let v1 = save_lvl_text(&mut s1);
            if v1.starts_with("REFUSED") || v1.starts_with("write failed") || v1 == "no path" {
                failures.push(format!("  {name}: first save did not succeed: {v1}"));
                cleanup(&path);
                continue;
            }
            let t1 = std::fs::read_to_string(&path).expect("read after save 1");

            let mut s2 = EditorState::new_empty(4, 4, 2);
            if !crate::open::load_path(&mut s2, &path) {
                failures.push(format!("  {name}: the SAVED file no longer opens"));
                cleanup(&path);
                continue;
            }
            let v2 = save_lvl_text(&mut s2);
            if v2.starts_with("REFUSED") || v2.starts_with("write failed") {
                failures.push(format!("  {name}: second save did not succeed: {v2}"));
                cleanup(&path);
                continue;
            }
            let t2 = std::fs::read_to_string(&path).expect("read after save 2");

            checked += 1;
            if t1 != t2 {
                failures.push(format!(
                    "  {name}: open→save is NOT a byte-stable fixed point ({} bytes then {} bytes){}",
                    t1.len(),
                    t2.len(),
                    first_line_diff(&t1, &t2)
                ));
            }
            if t1 == orig {
                canonical_seen.push(name.clone());
            } else if CANONICAL.contains(&name.as_str()) {
                failures.push(format!(
                    "  {name}: was byte-identical to disk and no longer is{}",
                    first_line_diff(&orig, &t1)
                ));
            }
            cleanup(&path);
        }

        assert!(
            checked >= 60,
            "expected the ~63-level shipped corpus, only found {checked}"
        );
        assert!(
            failures.is_empty(),
            "{} shipped level(s) are not byte-stable through an editor open → save \
             (canonical-on-disk levels seen: {canonical_seen:?}):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// The first differing line between two texts, as a one-line suffix for a
    /// failure message (empty when they match).
    fn first_line_diff(a: &str, b: &str) -> String {
        let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
        for i in 0..la.len().max(lb.len()) {
            let x = la.get(i).copied().unwrap_or("<eof>");
            let y = lb.get(i).copied().unwrap_or("<eof>");
            if x != y {
                return format!("\n      first diff at line {i}: `{x}` | `{y}`");
            }
        }
        String::new()
    }

    /// A serialize the parser rejects is REFUSED — the original
    /// path is never written, `dirty` stays set, and the rejected text is kept
    /// in a `<path>.bak` sidecar so the work is never permanently unsavable.
    #[test]
    fn reparse_guard_refuses_corrupt_serialize_and_keeps_bak() {
        let path = scratch_path("w18-guard.lvl");
        let bak = format!("{path}.bak");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak);

        let mut state = EditorState::new_empty(4, 4, 2);
        state.dirty = true;
        state.path = Some(path.clone());
        // A pipe whose endpoints name no node serializes to
        // `[pipe ghost->phantom]`, which `re_level` finalize rejects.
        state.level.pipes.push(re_level::Pipe {
            from: "ghost".into(),
            to: "phantom".into(),
            ..Default::default()
        });

        let verdict = save_lvl_text(&mut state);
        assert!(
            verdict.starts_with("REFUSED"),
            "guard refuses the write: {verdict}"
        );
        assert!(state.dirty, "refused save keeps the document dirty");
        assert!(
            !std::path::Path::new(&path).exists(),
            "the original path is never written with unparseable text"
        );
        let bak_text = std::fs::read_to_string(&bak).expect(".bak sidecar written");
        assert!(
            bak_text.contains("[pipe ghost->phantom]"),
            ".bak holds the rejected text"
        );
        let _ = std::fs::remove_file(&bak);
    }

    /// Sidecar naming — file stem preferred, sanitized level name as the
    /// fallback, `untitled` as the last resort; always under `levels/_scratch/`.
    #[test]
    fn playtest_sidecar_path_naming() {
        // The pid suffix is spelled out rather than pattern-matched, so the
        // assertion still pins the WHOLE path (directory, `preview-` prefix,
        // stem, extension) and not just a prefix of it.
        let pid = std::process::id();
        assert_eq!(
            playtest_sidecar_path(Some("levels/08-material-demo.lvl"), "x"),
            format!("levels/_scratch/preview-08-material-demo-{pid}.lvl")
        );
        assert_eq!(
            playtest_sidecar_path(Some("levels\\sub\\deep.lvl"), "x"),
            format!("levels/_scratch/preview-deep-{pid}.lvl"),
            "backslash paths reduce to the stem"
        );
        assert_eq!(
            playtest_sidecar_path(None, "my level!"),
            format!("levels/_scratch/preview-mylevel-{pid}.lvl"),
            "no path: sanitized level name"
        );
        assert_eq!(
            playtest_sidecar_path(None, "!!!"),
            format!("levels/_scratch/preview-untitled-{pid}.lvl"),
            "nothing usable: untitled"
        );
        // The pid is what keeps two concurrent editors off each other's sidecar,
        // and it must NOT vary between two calls in one process — otherwise a
        // single session would strand one stale file per F5 press.
        assert_eq!(
            playtest_sidecar_path(None, "!!!"),
            playtest_sidecar_path(None, "!!!"),
            "stable within one process"
        );
    }

    /// The sidecar save never touches the original path or the dirty
    /// flag, and its reparse guard refuses an unloadable serialize BEFORE any
    /// disk work.
    #[test]
    fn playtest_sidecar_leaves_original_untouched_and_guards_reparse() {
        // A dirty document pointing at an original path that does NOT exist on
        // disk — the sidecar save must not create it.
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.dirty = true;
        // The "original" lives under this process's private scratch root, so no
        // concurrent test process can create it and turn the "never written"
        // assertion below into a false failure.
        let original = crate::testutil::scratch_path("w16-original.lvl");
        state.path = Some(original.clone());
        let sidecar = save_playtest_sidecar(&mut state).expect("sidecar saves");
        // Adopt the sidecar for cleanup AFTER the save — writing it is the code
        // under test's job — so Drop removes it even if an assertion below panics.
        let sidecar_guard = crate::testutil::ScratchFile::adopt(sidecar.clone());
        let stem = std::path::Path::new(&original)
            .file_stem()
            .expect("fixture has a stem")
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            sidecar,
            format!("levels/_scratch/preview-{stem}-{}.lvl", std::process::id()),
            "sidecar = scratch dir + `preview-` + the ORIGINAL's stem + this pid"
        );
        assert!(state.dirty, "sidecar save never clears dirty");
        assert!(
            !std::path::Path::new(&original).exists(),
            "the ORIGINAL file is never written by a playtest save"
        );
        let text = std::fs::read_to_string(sidecar_guard.path()).expect("sidecar written");
        assert!(text.contains("[cells]"), "sidecar carries the live edits");
        drop(sidecar_guard);
        // The `levels/_scratch` and `levels` directories are deliberately NOT
        // removed: `remove_dir` succeeds only on an EMPTY directory, i.e. exactly
        // when a concurrent test process has just created it and not yet written
        // into it — deleting it there makes that process's write fail. An empty
        // leftover directory is invisible to git and harmless.

        // Reparse guard: an unloadable document is refused with NO write.
        let mut bad = EditorState::new_empty(4, 4, 2);
        bad.level.pipes.push(re_level::Pipe {
            from: "ghost".into(),
            to: "phantom".into(),
            ..Default::default()
        });
        let err = save_playtest_sidecar(&mut bad).expect_err("reparse guard refuses");
        assert!(err.contains("reparse failed"), "refusal reason: {err}");
        assert!(
            !std::path::Path::new(&playtest_sidecar_path(bad.path.as_deref(), &bad.level.name))
                .exists(),
            "nothing written on refusal"
        );
    }

    /// Autosave sidecar naming — the human-legible basename
    /// without `.lvl`, then a hash of the canonical absolute path; `untitled` for
    /// a never-saved document; always under `levels/.autosave/`.
    #[test]
    fn autosave_path_for_naming() {
        let p = autosave_path_for(Some("levels/08-material-demo.lvl"));
        assert!(
            p.starts_with("levels/.autosave/08-material-demo-") && p.ends_with(".lvl"),
            "stem stays legible, hash suffix appended: {p}"
        );
        let q = autosave_path_for(Some("levels\\sub\\deep.lvl"));
        assert!(
            q.starts_with("levels/.autosave/deep-"),
            "backslash paths reduce to the stem: {q}"
        );
        assert_eq!(
            autosave_path_for(None),
            "levels/.autosave/untitled.lvl",
            "no path: untitled (nothing to key on)"
        );
        assert_eq!(
            autosave_path_for(Some("levels/08-material-demo.lvl")),
            p,
            "the key is deterministic for a given path"
        );
    }

    /// Two DIFFERENT absolute paths that share a basename get
    /// DIFFERENT sidecar paths — keying on the basename alone would make
    /// opening `/other/tree/levels/03.lvl` write to (and restore from, and
    /// delete) this tree's `03.lvl` sidecar instead, so one `s` could overwrite
    /// your level with another file's contents.
    #[test]
    fn same_basename_in_different_dirs_gets_different_sidecars() {
        // Two real directories so the canonical-parent key is exercised on disk.
        // Process-unique for the same reason `scratch_path` is: these used to be
        // fixed shared names, so a concurrent run could delete them mid-test.
        let a_dir = scratch_path("h4-a");
        let b_dir = scratch_path("h4-b");
        let (a_dir, b_dir) = (a_dir.as_str(), b_dir.as_str());
        std::fs::create_dir_all(a_dir).expect("dir a");
        std::fs::create_dir_all(b_dir).expect("dir b");
        let a = format!("{a_dir}/03.lvl");
        let b = format!("{b_dir}/03.lvl");

        let sa = autosave_path_for(Some(&a));
        let sb = autosave_path_for(Some(&b));
        assert_ne!(sa, sb, "same basename, different dirs ⇒ different sidecars");
        assert!(
            sa.starts_with("levels/.autosave/03-") && sb.starts_with("levels/.autosave/03-"),
            "both keep the legible stem: {sa} / {sb}"
        );

        // The key is stable across the document's own CREATION — canonicalizing
        // the file itself would change the key the moment it first exists.
        std::fs::write(&a, "x").expect("create a");
        assert_eq!(
            autosave_path_for(Some(&a)),
            sa,
            "key stable once the file exists"
        );

        // Two spellings of the SAME file resolve to the SAME sidecar.
        let a_dotted = format!("{a_dir}/./03.lvl");
        assert_eq!(
            autosave_path_for(Some(&a_dotted)),
            sa,
            "the same document always resolves to one sidecar"
        );

        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_dir(a_dir);
        let _ = std::fs::remove_dir(b_dir);
    }

    /// A save that fails MID-WRITE leaves the ORIGINAL file
    /// byte-intact. We make the write fail deterministically by parking a
    /// DIRECTORY at `<path>.tmp`, so the temp write cannot succeed — the same
    /// shape as a disk-full / antivirus-lock failure. A naive `fs::write`
    /// straight to the destination would truncate the original first, losing
    /// both the file and the edit.
    #[test]
    fn failed_save_leaves_the_original_file_intact() {
        let path = scratch_path("h3-atomic.lvl");
        let tmp = format!("{path}.tmp");
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);

        // An original on disk with content we can recognise afterwards.
        let original = "# PRECIOUS ORIGINAL\n[meta]\nname = keepme\n\n[room]\nsize = 4,4,2\n";
        std::fs::write(&path, original).expect("fixture");
        // Block the temp write: `fs::write` cannot write over a directory.
        std::fs::create_dir_all(&tmp).expect("block the tmp path");

        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.path = Some(path.clone());
        state.dirty = true;

        let verdict = save_lvl_text(&mut state);
        assert!(
            verdict.starts_with("write failed"),
            "the save reports failure: {verdict}"
        );
        assert!(state.dirty, "a failed save keeps the document dirty");
        assert_eq!(
            std::fs::read_to_string(&path).expect("original still readable"),
            original,
            "the ORIGINAL file is byte-intact after a failed write"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// A SUCCESSFUL save keeps one generation of the previous file
    /// contents in `<path>.bak`, and leaves no `.tmp` behind.
    #[test]
    fn successful_save_keeps_one_bak_generation() {
        let path = scratch_path("h3-bak.lvl");
        let bak = format!("{path}.bak");
        // Start from a clean slate — "the first save makes no .bak" is only a
        // meaningful assertion when the document does not exist yet.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak);

        // First save: nothing to back up (the file does not exist yet).
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.path = Some(path.clone());
        state.dirty = true;
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("write failed"),
            "first save succeeds: {verdict}"
        );
        let first = std::fs::read_to_string(&path).expect("first save on disk");
        assert!(
            !std::path::Path::new(&bak).exists(),
            "no .bak for a first save — there was no previous content"
        );
        assert!(
            !std::path::Path::new(&format!("{path}.tmp")).exists(),
            "no .tmp left after the rename"
        );

        // Second save: the FIRST save's bytes are preserved in the .bak.
        state.map.set(2, 2, 0, CellKind::WallSolid);
        state.dirty = true;
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("write failed"),
            "second save succeeds: {verdict}"
        );
        assert_eq!(
            std::fs::read_to_string(&bak).expect(".bak written"),
            first,
            ".bak holds the contents the save replaced"
        );
        assert_ne!(
            std::fs::read_to_string(&path).expect("read saved"),
            first,
            "the file itself carries the NEW content"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak);
    }

    /// `write_autosave` writes the serialized document to the sidecar via
    /// temp + rename (no `.tmp` left behind), never touches the original path,
    /// and never clears the dirty flag.
    #[test]
    fn autosave_writes_sidecar_and_leaves_original_untouched() {
        let dest = scratch_path("w24-as-write.lvl");
        let tmp = format!("{dest}.tmp");
        let _ = std::fs::remove_file(&dest);

        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(2, 2, 0, CellKind::WallSolid);
        state.dirty = true;
        // An original path that does NOT exist on disk — autosave must not create
        // it. Process-unique, so no concurrent run (and no stale artifact from a
        // crashed one) can create it and break the assertion below.
        let original = scratch_path("w24-as-original.lvl");
        state.path = Some(original.clone());

        write_autosave(&mut state, &dest).expect("autosave writes");
        assert!(state.dirty, "autosave never clears dirty");
        assert!(
            !std::path::Path::new(&original).exists(),
            "the ORIGINAL file is never written by an autosave"
        );
        assert!(
            !std::path::Path::new(&tmp).exists(),
            "no .tmp left after the rename"
        );
        let text = std::fs::read_to_string(&dest).expect("sidecar written");
        let reloaded = re_level::load_string(&text, &dest).expect("sidecar parses");
        assert_eq!(
            reloaded.build_map().cell_at(2, 2, 0),
            CellKind::WallSolid,
            "sidecar carries the live (dirty) edits"
        );
        let _ = std::fs::remove_file(&dest);
    }

    /// The autosave reparse guard — a serialize the parser rejects writes
    /// NOTHING (no sidecar, no `.tmp`).
    #[test]
    fn autosave_reparse_guard_writes_nothing() {
        let dest = scratch_path("w24-as-guard.lvl");
        let _ = std::fs::remove_file(&dest);
        let mut bad = EditorState::new_empty(4, 4, 2);
        bad.dirty = true;
        // Endpoints naming no node serialize to a header finalize rejects.
        bad.level.pipes.push(re_level::Pipe {
            from: "ghost".into(),
            to: "phantom".into(),
            ..Default::default()
        });
        let err = write_autosave(&mut bad, &dest).expect_err("guard refuses");
        assert!(err.contains("reparse failed"), "refusal reason: {err}");
        assert!(
            !std::path::Path::new(&dest).exists(),
            "no sidecar on refusal"
        );
        assert!(
            !std::path::Path::new(&format!("{dest}.tmp")).exists(),
            "no .tmp on refusal"
        );
    }

    /// `delete_autosave` removes an existing sidecar and is a silent no-op
    /// on a missing one.
    #[test]
    fn delete_autosave_removes_sidecar_and_ignores_missing() {
        let side = scratch_path("w24-as-delete.lvl");
        std::fs::write(&side, "stale").expect("fixture");
        delete_autosave(&side);
        assert!(!std::path::Path::new(&side).exists(), "sidecar deleted");
        delete_autosave(&side); // second call: missing file, must not panic
    }

    /// A SUCCESSFUL real save deletes the document's autosave sidecar (at
    /// the production `levels/.autosave/<stem>.lvl` path, cwd-relative like
    /// `scan_lvl_files`) and resets the autosave timer.
    #[test]
    fn successful_save_deletes_autosave_sidecar_and_resets_timer() {
        let path = scratch_path("w24-save-clears.lvl");
        // The PRODUCTION sidecar path for this document (crate-cwd relative).
        let sidecar = autosave_path_for(Some(&path));
        std::fs::create_dir_all("levels/.autosave").expect("sidecar dir");
        std::fs::write(&sidecar, "stale autosave").expect("fake sidecar");

        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.dirty = true;
        state.autosave_elapsed = 42.0;
        state.path = Some(path.clone());
        let verdict = save_lvl_text(&mut state);
        assert!(
            !verdict.starts_with("REFUSED") && !verdict.starts_with("write failed"),
            "save succeeds: {verdict}"
        );
        assert!(!state.dirty, "save clears dirty");
        assert_eq!(
            state.autosave_elapsed, 0.0,
            "save resets the autosave timer"
        );
        assert!(
            !std::path::Path::new(&sidecar).exists(),
            "a successful save deletes the sidecar for its stem"
        );
        let _ = std::fs::remove_file(&path);
        // (No shared-dir cleanup: removing `levels/.autosave` here races the
        // parallel tests that are writing their own sidecars into it — an empty
        // leftover dir is invisible to git and harmless.)
    }

    #[test]
    fn lvl_text_roundtrips_byte_identical() {
        let src = std::fs::read_to_string("../../levels/01.lvl").expect("read level 01");
        let level = re_level::load_string(&src, "01.lvl").expect("parse");
        let out = re_voximport::to_lvl_text(&level);
        let level2 = re_level::load_string(&out, "01-roundtrip.lvl").expect("reparse");
        let out2 = re_voximport::to_lvl_text(&level2);
        assert_eq!(out, out2, "to_lvl_text must be idempotent under reparse");
    }

    /// The critical proof: paint → save → reload yields the same cells. We
    /// drive the full save path (capture + serialize) and reload through the
    /// real parser, comparing the rebuilt map cell-for-cell.
    #[test]
    fn paint_save_reload_preserves_cells() {
        // A small room with a couple of deliberate, distinctive edits.
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(2, 3, 1, CellKind::WallSolid);
        state.map.set(4, 1, 2, CellKind::WallMatrix);
        state.map.set_cell(1, 1, 1, CellKind::Object, 1); // tagged node
        state.map.set(0, 0, 0, CellKind::Floor);

        // Serialize via the SERIALIZE-ONLY path (no disk write, no dirty change —
        // this test exercises the capture + round-trip, not the save-destination
        // policy, which is tested separately in saveas.rs).
        let text = capture_and_serialize(&mut state);
        assert!(state.level.cells.is_some(), "capture must set [cells]");

        // Reload through the real parser and rebuild the map.
        let reloaded = re_level::load_string(&text, "roundtrip.lvl").expect("reparse");
        let rebuilt = reloaded.build_map();

        // Every cell must match the painted map.
        let (sx, sy, sz) = state.map.size();
        assert_eq!(rebuilt.size(), (sx, sy, sz));
        for z in 0..sz as i32 {
            for y in 0..sy as i32 {
                for x in 0..sx as i32 {
                    assert_eq!(
                        rebuilt.cell_at(x, y, z),
                        state.map.cell_at(x, y, z),
                        "kind mismatch at {x},{y},{z}"
                    );
                    assert_eq!(
                        rebuilt.side_data_at(x, y, z),
                        state.map.side_data_at(x, y, z),
                        "side_data mismatch at {x},{y},{z}"
                    );
                }
            }
        }
    }

    /// Entity round-trip: place a door, a node (with an io role and a
    /// threshold), and a spawn through the object-placement path, save, reload,
    /// and assert the entities survive alongside the painted cells. Entities live
    /// in typed `level.*` fields that `to_lvl_text` already serializes, so this
    /// proves the editor's placement reaches the saved file losslessly.
    #[test]
    fn place_entities_save_reload_preserves_them() {
        use crate::entity;
        use crate::state::EntityKind;

        let mut state = EditorState::new_empty(8, 8, 4);
        // Some geometry so the cells path is exercised too.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Door at (5,6,2).
        state.place = Some(EntityKind::Door);
        state.cursor = (5, 6);
        state.z = 2;
        entity::place_at_cursor(&mut state);

        // Spawn at (1,2,1) facing +x (cycle once from default +y).
        state.place = Some(EntityKind::Spawn);
        state.cursor = (1, 2);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        state.entity_field = crate::state::EntityField::SpawnFacing;
        entity::cycle_property(&mut state); // +y -> +x
        let want_facing = state.level.player.facing_yaw_col;

        // Node at (3,3,0): make it an output with threshold 2.
        state.place = Some(EntityKind::Node);
        state.cursor = (3, 3);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        state.entity_field = crate::state::EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither -> input
        entity::cycle_property(&mut state); // input -> output
        state.entity_field = crate::state::EntityField::NodeThreshold;
        entity::adjust_property(&mut state, 2); // threshold +2
        let node_name = state.level.nodes[0].name.clone();

        // Save (no disk; path is None) and reload through the real parser.
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "entities-rt.lvl").expect("reparse");

        // Door position survived.
        assert_eq!(
            (reloaded.door.x, reloaded.door.y, reloaded.door.z),
            (5, 6, 2),
            "door position round-trips"
        );
        // Spawn position + facing survived.
        assert_eq!(
            (reloaded.player.x, reloaded.player.y, reloaded.player.z),
            (1, 2, 1),
            "spawn position round-trips"
        );
        assert_eq!(
            reloaded.player.facing_yaw_col, want_facing,
            "spawn facing round-trips"
        );
        // Node survived with its name, io role, and threshold.
        let nidx = reloaded
            .node_index(&node_name)
            .expect("node name round-trips");
        let n = &reloaded.nodes[nidx];
        assert_eq!((n.x, n.y, n.z), (3, 3, 0), "node position round-trips");
        assert!(n.is_output && !n.is_input, "node io role round-trips");
        assert_eq!(n.threshold, 2, "node threshold round-trips");

        // The painted floor cell still round-trips (entities didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// PIPE round-trip: wire two nodes into a pipe through the editor's
    /// pipe-create flow (auto-routed L-shaped path), set state ON, a switch_at on a
    /// path cell, and a non-default density, save, reload through the real parser,
    /// and assert the pipe + its endpoints + auto-routed path + every prop survive.
    /// Proves the editor's pipe authoring reaches the saved file losslessly.
    #[test]
    fn wire_pipe_save_reload_preserves_it() {
        use crate::state::EntityField;

        let mut state = EditorState::new_empty(12, 12, 4);
        // Two nodes the pipe connects (placed directly; the create flow connects
        // them by cell).
        let a = state.level.push_node(1, 1, 0);
        let b = state.level.push_node(5, 3, 0);
        let a_name = state.level.nodes[a].name.clone();
        let b_name = state.level.nodes[b].name.clone();

        // Wire a -> b via the create flow.
        state.cursor = (1, 1);
        state.z = 0;
        crate::pipe::start_pipe_create(&mut state);
        state.cursor = (5, 3);
        assert!(crate::pipe::pick_node(&mut state, b), "pipe wired");
        let pi = state.selected_pipe.expect("pipe selected");

        // Record the auto-routed path for the post-reload comparison.
        let routed_path = state.level.pipes[pi].path.clone();
        assert_eq!(routed_path.first(), Some(&(1, 1, 0)));
        assert_eq!(routed_path.last(), Some(&(5, 3, 0)));

        // state ON.
        state.entity_field = EntityField::PipeState;
        crate::pipe::cycle_enum(&mut state);
        // switch_at on a mid-path cell.
        let switch_cell = routed_path[2];
        state.cursor = (switch_cell.0 as usize, switch_cell.1 as usize);
        state.z = switch_cell.2 as usize;
        state.entity_field = EntityField::PipeSwitch;
        crate::pipe::cycle_enum(&mut state);
        // density down to 0.5.
        state.entity_field = EntityField::PipeDensity;
        crate::pipe::adjust_numeric(&mut state, -1); // 1.0 -> 0.9
        for _ in 0..4 {
            crate::pipe::adjust_numeric(&mut state, -1);
        } // -> 0.5

        // Save + reload through the real parser (which re-resolves endpoints +
        // switch_at against the expanded path in `finalize`).
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "pipe-rt.lvl").expect("reparse");

        assert_eq!(reloaded.pipes.len(), 1, "one pipe survives");
        let p = &reloaded.pipes[0];
        // Endpoints by name (re-resolved to indices on load).
        assert_eq!(p.from, a_name);
        assert_eq!(p.to, b_name);
        assert_eq!(p.from_node, reloaded.node_index(&a_name).unwrap());
        assert_eq!(p.to_node, reloaded.node_index(&b_name).unwrap());
        // The auto-routed path survives cell-for-cell.
        assert_eq!(p.path, routed_path, "auto-routed path round-trips");
        // Props.
        assert_eq!(p.state, re_level::PipeState::On, "state round-trips");
        assert!(
            (p.density - 0.5).abs() < 1e-6,
            "density round-trips: {}",
            p.density
        );
        // switch_at re-resolves to the SAME path cell.
        let sw = p.switch_cell.and_then(|c| p.path.get(c)).copied();
        assert_eq!(
            sw,
            Some(switch_cell),
            "switch_at round-trips to the path cell"
        );
    }

    /// Wire a small two-node pipe and return the editor state with it selected.
    #[cfg(test)]
    fn wired_pipe_state() -> (EditorState, String, String) {
        let mut state = EditorState::new_empty(12, 12, 4);
        let a = state.level.push_node(1, 1, 0);
        let b = state.level.push_node(5, 3, 0);
        let a_name = state.level.nodes[a].name.clone();
        let b_name = state.level.nodes[b].name.clone();
        state.cursor = (1, 1);
        state.z = 0;
        crate::pipe::start_pipe_create(&mut state);
        state.cursor = (5, 3);
        assert!(crate::pipe::pick_node(&mut state, b), "pipe wired");
        (state, a_name, b_name)
    }

    /// SPLINE MODE round-trip: wire a pipe, flip it to spline mode via the focused
    /// field, serialize, and assert the writer emits `spline = …` (NOT `path = …`)
    /// with the pipe's points as control points — and writes EXACTLY ONE of the
    /// two. Then write the text to disk, reload through `open::load_path`, and
    /// assert the editor re-flags the pipe as a spline (the `re_level` model drops
    /// the fact, so the load-time scan must recover it) with ≥2 control points.
    #[test]
    fn pipe_spline_mode_save_reload_round_trips() {
        use crate::state::{EntityField, EntityKind};

        let (mut state, a_name, b_name) = wired_pipe_state();
        let pi = state.selected_pipe.expect("pipe selected");
        let points = state.level.pipes[pi].path.len();
        assert!(
            points >= 2,
            "a wired pipe has ≥2 cells to use as control points"
        );
        // The EXACT authored path — Catmull-Rom + supercover voxelization must
        // reproduce it bit-for-bit on reload (S3: assert path identity, not length).
        let authored_path = state.level.pipes[pi].path.clone();

        // Flip to SPLINE mode via the real focused-field toggle.
        state.place = Some(EntityKind::Pipe);
        state.entity_field = EntityField::PipeMode;
        crate::pipe::cycle_enum(&mut state);
        assert!(
            state
                .spline_pipes
                .contains(&(a_name.clone(), b_name.clone())),
            "PipeMode toggle flagged the pipe spline-mode"
        );

        // Serialize and inspect: exactly one of `spline =` / `path =` is emitted.
        let text = capture_and_serialize(&mut state);
        let spline_lines = text
            .lines()
            .filter(|l| l.trim_start().starts_with("spline ="))
            .count();
        let path_lines = text
            .lines()
            .filter(|l| l.trim_start().starts_with("path ="))
            .count();
        assert_eq!(
            spline_lines, 1,
            "spline pipe emits exactly one `spline =`:\n{text}"
        );
        assert_eq!(
            path_lines, 0,
            "spline pipe emits NO `path =` (never both):\n{text}"
        );

        // The reparsed level still has the pipe (re_level voxelizes the spline into
        // a contiguous path), with ≥2 cells.
        let reloaded = re_level::load_string(&text, "spline-rt.lvl").expect("reparse");
        assert_eq!(reloaded.pipes.len(), 1, "spline pipe survives reparse");
        assert_eq!(
            reloaded.pipes[0].path, authored_path,
            "the spline voxelizes back to EXACTLY the authored path (Catmull-Rom \
             + supercover is a faithful round-trip), not merely a ≥2-cell path"
        );

        // Full disk round-trip through the editor's loader: the spline flag must be
        // recovered from the raw `.lvl` text.
        let file = scratch_path("spline-rt.lvl");
        std::fs::write(&file, &text).expect("write temp lvl");
        let mut fresh = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut fresh, &file),
            "reload via load_path succeeds"
        );
        let _ = std::fs::remove_file(&file);
        assert_eq!(fresh.level.pipes.len(), 1, "one pipe after reload");
        let p = &fresh.level.pipes[0];
        assert!(
            crate::pipe::is_spline(&fresh, p),
            "the reloaded pipe is re-flagged as a spline (scan recovered it)"
        );
        // field_value reports `spline` for the reloaded pipe.
        fresh.place = Some(EntityKind::Pipe);
        fresh.selected_pipe = Some(0);
        fresh.entity_field = EntityField::PipeMode;
        assert_eq!(
            crate::pipe::field_value_of(&fresh, fresh.entity_field),
            "spline"
        );
    }

    /// PATH MODE is the default and unchanged: a wired pipe NOT flipped to spline
    /// emits `path = …` (and NO `spline = …`), and reloads as a path-mode pipe.
    #[test]
    fn pipe_path_mode_is_default_and_round_trips() {
        use crate::state::{EntityField, EntityKind};

        let (mut state, _a, _b) = wired_pipe_state();
        assert!(
            state.spline_pipes.is_empty(),
            "fresh pipe defaults to path mode"
        );

        let text = capture_and_serialize(&mut state);
        let spline_lines = text
            .lines()
            .filter(|l| l.trim_start().starts_with("spline ="))
            .count();
        let path_lines = text
            .lines()
            .filter(|l| l.trim_start().starts_with("path ="))
            .count();
        assert_eq!(
            path_lines, 1,
            "path pipe emits exactly one `path =`:\n{text}"
        );
        assert_eq!(
            spline_lines, 0,
            "path pipe emits NO `spline =` (never both):\n{text}"
        );

        // Disk round-trip: the pipe reloads as path-mode (not in spline_pipes).
        let file = scratch_path("path-rt.lvl");
        std::fs::write(&file, &text).expect("write temp lvl");
        let mut fresh = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut fresh, &file));
        let _ = std::fs::remove_file(&file);
        assert_eq!(fresh.level.pipes.len(), 1);
        assert!(
            !crate::pipe::is_spline(&fresh, &fresh.level.pipes[0]),
            "the reloaded pipe is path-mode (scan found no spline key)"
        );
        fresh.place = Some(EntityKind::Pipe);
        fresh.selected_pipe = Some(0);
        fresh.entity_field = EntityField::PipeMode;
        assert_eq!(
            crate::pipe::field_value_of(&fresh, fresh.entity_field),
            "path"
        );
    }

    /// EXHAUSTIVE object editing round-trip: place a node, wire it as a door
    /// target (on) and a second node as an off-target, set node pool/color/anchor/
    /// glyph/threshold, set pedestal per-axis volume + center, save, reload, and
    /// assert EVERY newly-editable field survives `to_lvl_text` ↔ parser. This is
    /// the proof that the editor's exhaustive object settings round-trip.
    #[test]
    fn exhaustive_object_edits_save_reload_preserve_every_field() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(10, 10, 4);

        // Two nodes: n0 (the door's ON target) and n1 (the OFF target).
        state.place = Some(EntityKind::Node);
        state.cursor = (2, 2);
        state.z = 0;
        entity::place_at_cursor(&mut state); // n0 selected
        let n0_name = state.level.nodes[0].name.clone();
        state.cursor = (3, 2);
        entity::place_at_cursor(&mut state); // n1 selected
        let n1_name = state.level.nodes[1].name.clone();

        // Edit n0 exhaustively: glyph, color, io=output, threshold, anchor, pool,
        // pool_alt, and move its pos.
        state.selected_node = Some(0);
        state.entity_field = EntityField::NodeGlyph;
        entity::cycle_property(&mut state); // N -> next glyph
        let want_glyph = state.level.nodes[0].glyph;
        state.entity_field = EntityField::NodeColor;
        entity::adjust_property(&mut state, 1);
        let want_color = state.level.nodes[0].color_rgb;
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither -> input
        entity::cycle_property(&mut state); // input -> output
        state.entity_field = EntityField::NodeThreshold;
        entity::adjust_property(&mut state, 3);
        state.entity_field = EntityField::NodeAnchor;
        entity::adjust_property(&mut state, 2);
        state.entity_field = EntityField::NodePool;
        entity::adjust_property(&mut state, 1); // +1 word
        entity::adjust_property(&mut state, 1); // +1 word (2 total)
        state.entity_field = EntityField::NodePoolAlt;
        entity::adjust_property(&mut state, 1); // +1 alt word
        state.entity_field = EntityField::NodePosX;
        entity::adjust_property(&mut state, 1); // x 2 -> 3? clamps in room
        let want_pos = (
            state.level.nodes[0].x,
            state.level.nodes[0].y,
            state.level.nodes[0].z,
        );
        let want_thr = state.level.nodes[0].threshold;
        let want_anchor = state.level.nodes[0].anchor;
        let want_pool = state.level.nodes[0].pool.clone();
        let want_pool_alt = state.level.nodes[0].pool_alt.clone();

        // Door: place it, wire n0 as an ON target and n1 as an OFF target.
        state.place = Some(EntityKind::Door);
        state.cursor = (5, 5);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        // Cursor on n0 (index 0), toggle ON.
        state.door_target_cursor = 0;
        state.entity_field = EntityField::DoorTargetOn;
        entity::cycle_property(&mut state); // add n0 to target_parts
                                            // Cursor on n1 (index 1), toggle OFF.
        state.door_target_cursor = 1;
        state.entity_field = EntityField::DoorTargetOff;
        entity::cycle_property(&mut state); // add n1 to target_off

        // Pedestal: place + per-axis volume + center nudge.
        state.place = Some(EntityKind::Pedestal);
        state.cursor = (6, 6);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        state.entity_field = EntityField::PedVolW;
        entity::adjust_property(&mut state, 2); // w 1 -> 3
        state.entity_field = EntityField::PedVolH;
        entity::adjust_property(&mut state, 1); // h 1 -> 2
        state.entity_field = EntityField::PedVolD;
        entity::adjust_property(&mut state, 3); // d 1 -> 4
        state.entity_field = EntityField::PedCenterX;
        entity::adjust_property(&mut state, 1);
        let want_ped_vol = (
            state.level.pedestal.w,
            state.level.pedestal.h,
            state.level.pedestal.d,
        );
        let want_ped_center = (
            state.level.pedestal.cx,
            state.level.pedestal.cy,
            state.level.pedestal.cz,
        );

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "exhaustive-rt.lvl").expect("reparse");

        // Node n0: every edited field survived.
        let i0 = reloaded.node_index(&n0_name).expect("n0 name round-trips");
        let n0 = &reloaded.nodes[i0];
        assert_eq!(n0.glyph, want_glyph, "node glyph round-trips");
        assert_eq!(
            n0.color_rgb & 0xFF_FFFF,
            want_color & 0xFF_FFFF,
            "node color"
        );
        assert!(n0.is_output && !n0.is_input, "node io role round-trips");
        assert_eq!(n0.threshold, want_thr, "node threshold round-trips");
        assert_eq!(n0.anchor, want_anchor, "node anchor round-trips");
        assert_eq!(n0.pool, want_pool, "node pool round-trips");
        assert_eq!(n0.pool_alt, want_pool_alt, "node pool_alt round-trips");
        assert_eq!((n0.x, n0.y, n0.z), want_pos, "node pos round-trips");

        // Door targets resolved back to the SAME node indices (by name).
        let on_idx = reloaded.node_index(&n0_name).unwrap();
        let off_idx = reloaded.node_index(&n1_name).unwrap();
        assert!(
            reloaded.door.target_parts.contains(&on_idx),
            "door target_parts (on) round-trips: {:?}",
            reloaded.door.target_parts
        );
        assert!(
            reloaded.door.target_off.contains(&off_idx),
            "door target_off round-trips: {:?}",
            reloaded.door.target_off
        );

        // Pedestal per-axis volume + center survived.
        assert_eq!(
            (
                reloaded.pedestal.w,
                reloaded.pedestal.h,
                reloaded.pedestal.d
            ),
            want_ped_vol,
            "pedestal per-axis volume round-trips"
        );
        assert_eq!(
            (
                reloaded.pedestal.cx,
                reloaded.pedestal.cy,
                reloaded.pedestal.cz
            ),
            want_ped_center,
            "pedestal center round-trips"
        );
    }

    /// Paint a wedge + a ramp, save, reload, rebuild: the 45° geometry survives
    /// the full editor save path end-to-end.
    #[test]
    fn paint_save_reload_preserves_shapes() {
        let mut state = EditorState::new_empty(5, 5, 2);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.map.set_shape(1, 1, 0, Shape::WedgeXY(DiagXY::NW));
        state.map.set(3, 2, 1, CellKind::WallSolid);
        state.map.set_shape(3, 2, 1, Shape::RampZ(RampDir::NegY));

        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("wedgexy nw"),
            "readable wedge token emitted:\n{text}"
        );
        assert!(
            text.contains("rampz negy"),
            "readable ramp token emitted:\n{text}"
        );

        let reloaded = re_level::load_string(&text, "shapes-rt.lvl").expect("reparse");
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.shape_at(1, 1, 0), Shape::WedgeXY(DiagXY::NW));
        assert_eq!(rebuilt.shape_at(3, 2, 1), Shape::RampZ(RampDir::NegY));
        assert_eq!(
            rebuilt.shape_at(0, 0, 0),
            Shape::Full,
            "unshaped cells stay Full"
        );
    }

    /// Emitter round-trip: place an emitter, tweak its rate + spread through
    /// the focused-field editor, save, reload, and assert the emitter + its edited
    /// params survive. Proves `to_lvl_text`'s new `[emitter]` writer is the exact
    /// inverse of `re_level`'s parser.
    #[test]
    fn place_emitter_edit_params_save_reload_preserves_them() {
        use crate::entity;
        use crate::state::{EmitterField, EntityKind};
        use re_world::EmitterOrigin;

        let mut state = EditorState::new_empty(8, 8, 4);
        // Some geometry so the cells path is exercised alongside the emitter.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Place an emitter at (3,4,1).
        state.place = Some(EntityKind::Emitter);
        state.cursor = (3, 4);
        state.z = 1;
        entity::place_at_cursor(&mut state);

        // Tweak rate (+3) and spread (+2 steps of 5° = +10°) via the focus editor.
        state.emitter_field = EmitterField::Rate;
        entity::adjust_property(&mut state, 1);
        entity::adjust_property(&mut state, 1);
        entity::adjust_property(&mut state, 1);
        state.emitter_field = EmitterField::SpreadDeg;
        entity::adjust_property(&mut state, 1);
        entity::adjust_property(&mut state, 1);
        // Cycle an enum too (color_mode RampOverAge → RandomPick).
        state.emitter_field = EmitterField::ColorMode;
        entity::cycle_property(&mut state);

        let want_rate = state.level.emitters[0].rate;
        let want_spread = state.level.emitters[0].spread_deg;
        let want_mode = state.level.emitters[0].color_mode;

        // Save (no disk; path is None) and reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[emitter"),
            "save emits an [emitter] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "emitter-rt.lvl").expect("reparse");

        assert_eq!(reloaded.emitters.len(), 1, "one emitter round-trips");
        let e = &reloaded.emitters[0];
        // Origin (cell centre 3.5,4.5,1.5) survives.
        match e.origin {
            EmitterOrigin::World(p) => {
                assert!((p.x - 3.5).abs() < 1e-4, "origin x round-trips");
                assert!((p.y - 4.5).abs() < 1e-4, "origin y round-trips");
                assert!((p.z - 1.5).abs() < 1e-4, "origin z round-trips");
            }
            _ => panic!("origin must be World"),
        }
        // Edited params survive.
        assert!((e.rate - want_rate).abs() < 1e-4, "rate round-trips");
        assert!(
            (e.spread_deg - want_spread).abs() < 1e-4,
            "spread round-trips"
        );
        assert_eq!(e.color_mode, want_mode, "color_mode round-trips");

        // The painted floor still round-trips (emitter didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// Material round-trip: create a material, edit its glyph + fg + recipe
    /// through the editor's material path, paint it onto a cell, save, reload, and
    /// assert BOTH the material-def edits AND the per-cell material id survive.
    /// Materials persist via `[material]`; per-cell ids via `[cells] material_ids`.
    #[test]
    fn material_edit_and_paint_save_reload_preserves_them() {
        use crate::material;
        use crate::material::MaterialField;

        let mut state = EditorState::new_empty(6, 6, 4);
        // Geometry so the cells path is exercised alongside the material.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Create a material and edit its appearance: glyph → '#', fg → a palette
        // swatch, recipe → brick.
        material::new_material(&mut state); // mat0, selected
        let want_name = state.level.materials[0].name.clone();

        state.material_field = MaterialField::Glyph;
        material::adjust_field(&mut state, 1); // None → '+'
        let want_glyph = state.level.materials[0].glyph;
        assert_eq!(want_glyph, Some('+' as u32));

        state.material_field = MaterialField::Fg;
        material::adjust_field(&mut state, 1); // step to a known palette swatch
        let want_fg = state.level.materials[0].fg;

        state.material_field = MaterialField::Recipe;
        material::adjust_field(&mut state, 1); // None → brick
        let want_recipe = state.level.materials[0].recipe;
        assert!(want_recipe.is_some());

        // Paint the material's id (1) onto a distinctive cell.
        state.cursor = (3, 4);
        state.z = 2;
        material::paint_id(&mut state);
        assert_eq!(state.map.side_data_at(3, 4, 2), 1, "id 1 stamped");

        // Save (no disk; path is None) and reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[material"),
            "save emits a [material] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "material-rt.lvl").expect("reparse");

        // The material DEFINITION edits survived.
        let midx = reloaded
            .material_index(&want_name)
            .expect("material name round-trips");
        let m = &reloaded.materials[midx];
        assert_eq!(m.glyph, want_glyph, "glyph edit round-trips");
        assert_eq!(m.fg & 0xFF_FFFF, want_fg & 0xFF_FFFF, "fg edit round-trips");
        assert_eq!(
            super::recipe_label_for_test(m.recipe),
            super::recipe_label_for_test(want_recipe),
            "recipe edit round-trips"
        );

        // The per-cell material ID survived (via [cells] material_ids).
        let rebuilt = reloaded.build_map();
        assert_eq!(
            rebuilt.side_data_at(3, 4, 2),
            1,
            "painted material id round-trips into the cell's side_data"
        );
        // And the unrelated floor cell is intact.
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// Vox round-trip: place a `.vox` stamp through the editor's vox path,
    /// set its scale + material, save, reload, and assert the `[vox]` entry
    /// (file / offset / material / scale) survives. This guards the `write_vox`
    /// writer in `re_voximport::to_lvl_text` (a section can be readable on
    /// load and silently dropped on save if no writer emits it — the same gap
    /// `[emitter]` once had).
    #[test]
    fn place_vox_set_params_save_reload_preserves_them() {
        use crate::vox;
        use re_level::VoxMaterial;

        let mut state = EditorState::new_empty(16, 16, 6);
        // A real committed `.vox` path so build_map can actually read + stamp it
        // (the editor cwd at test time is the crate dir; reach the workspace).
        // For the round-trip we only need the path STRING to survive — but we use
        // a real file so the rebuild stamps geometry too. `house01` is one of the
        // procedurally-generated models (`re_voxscene::genmodels`), self-made so
        // every asset under `authoring/` stays free of licensing constraints.
        state.vox_files = vec!["authoring/voxelmodels/generated/house01.vox".into()];
        state.vox_file_sel = 0;
        state.vox_mode = true;

        // Some painted geometry so the cells path is exercised alongside the vox.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Place the model at (4,5,1).
        state.cursor = (4, 5);
        state.z = 1;
        vox::place_at_cursor(&mut state);
        assert_eq!(state.level.voxes.len(), 1, "one vox placed");
        assert_eq!(state.selected_vox, Some(0));

        // Set material → trace, scale → 3 (both retarget the selected stamp).
        vox::cycle_material(&mut state); // Solid → Trace
        assert_eq!(state.level.voxes[0].material, VoxMaterial::Trace);
        vox::adjust_scale(&mut state, 2); // 1 → 3
        assert_eq!(state.level.voxes[0].scale, 3);

        // Save (no disk; path is None) and reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(text.contains("[vox"), "save emits a [vox] section:\n{text}");
        let reloaded = re_level::load_string(&text, "vox-rt.lvl").expect("reparse");

        // The vox entry survived with its file / offset / material / scale.
        assert_eq!(reloaded.voxes.len(), 1, "one vox round-trips");
        let v = &reloaded.voxes[0];
        assert_eq!(
            v.file, "authoring/voxelmodels/generated/house01.vox",
            "file round-trips"
        );
        assert_eq!((v.off_x, v.off_y, v.off_z), (4, 5, 1), "offset round-trips");
        assert_eq!(v.material, VoxMaterial::Trace, "material round-trips");
        assert_eq!(v.scale, 3, "scale round-trips");

        // The painted floor still round-trips (vox didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// EXHAUSTIVE emitter round-trip: set a representative value for EVERY
    /// newly-exposed emitter field (speed_spread, lifetime_spread, path_jitter,
    /// swirl, accel, max_particles, burst, seed, colour stops, glyph_set, blend,
    /// origin nudges), save, reload, and assert each survives. Proves the
    /// `[emitter]` writer is a complete inverse of the parser for these fields.
    #[test]
    fn exhaustive_emitter_fields_round_trip() {
        use re_world::{EmitterBlend, EmitterOrigin, EmitterSpec};

        let mut state = EditorState::new_empty(8, 8, 4);
        let mut spec = EmitterSpec::default()
            .with_origin(EmitterOrigin::World(glam::Vec3::new(3.7, 4.2, 1.3)));
        spec.colors = [0x112233, 0x445566, 0x778899];
        spec.speed_spread = 0.37;
        spec.lifetime_spread = 0.41;
        spec.path_jitter = 0.23;
        spec.swirl = -0.55;
        spec.accel = glam::Vec3::new(0.12, -0.34, 0.56);
        spec.max_particles = 333;
        spec.burst = Some(42);
        spec.seed = 0xDEAD_BEEF_1234;
        spec.glyph_set = Some(vec!['a', 'b', 'c']);
        spec.blend = EmitterBlend::Opaque;
        spec.rate = 7.0;
        spec.validate();
        state.level.emitters.push(spec.clone());

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "emitter-all-rt.lvl").expect("reparse");
        assert_eq!(reloaded.emitters.len(), 1);
        let e = &reloaded.emitters[0];

        match e.origin {
            EmitterOrigin::World(p) => {
                assert!(
                    (p.x - 3.7).abs() < 1e-4
                        && (p.y - 4.2).abs() < 1e-4
                        && (p.z - 1.3).abs() < 1e-4
                );
            }
            _ => panic!("origin must be World"),
        }
        assert_eq!(
            e.colors,
            [0x112233, 0x445566, 0x778899],
            "colour stops round-trip"
        );
        assert!((e.speed_spread - 0.37).abs() < 1e-4);
        assert!((e.lifetime_spread - 0.41).abs() < 1e-4);
        assert!((e.path_jitter - 0.23).abs() < 1e-4);
        assert!((e.swirl + 0.55).abs() < 1e-4);
        assert!((e.accel.x - 0.12).abs() < 1e-4);
        assert!((e.accel.y + 0.34).abs() < 1e-4);
        assert!((e.accel.z - 0.56).abs() < 1e-4);
        assert_eq!(e.max_particles, 333);
        assert_eq!(e.burst, Some(42));
        assert_eq!(e.seed, 0xDEAD_BEEF_1234);
        assert_eq!(e.glyph_set, Some(vec!['a', 'b', 'c']));
        assert_eq!(e.blend, EmitterBlend::Opaque);
    }

    /// EXHAUSTIVE material round-trip: set every newly-exposed material field —
    /// recipe variant + its params, animation (kind/speed/amplitude), rank,
    /// surface binding, reflective + clarity, preset — save, reload, assert each
    /// survives. Proves the writer fix (reflective/clarity/preset were dropped)
    /// closes the gap and that recipe params + anim + rank + surface round-trip.
    #[test]
    fn exhaustive_material_fields_round_trip() {
        use re_world::{MaterialAnim, MaterialAnimKind, MaterialRecipe};

        let mut state = EditorState::new_empty(6, 6, 4);
        let idx = state.level.push_material();
        let name = state.level.materials[idx].name.clone();
        {
            let m = &mut state.level.materials[idx];
            m.fg = 0x00_20FF80;
            m.bg = 0x00_101820;
            m.glyph = Some('+' as u32);
            m.recipe = Some(MaterialRecipe::Brick {
                aspect: 3.5,
                mortar_w: 0.22,
                grain: 0.9,
            });
            m.anim = MaterialAnim {
                kind: MaterialAnimKind::Pulse,
                speed: 2.5,
                amplitude: 0.6,
            };
            m.rank = 42;
            m.reflective = true;
            m.clarity = 0.4;
            m.preset = Some("metal".to_string());
        }
        // Bind it to a surface via the editor's material-mode enum cycle (none →
        // wall_solid → wall_trace), so the binding goes through the real code path.
        state.material_mode = true;
        state.material_sel = idx;
        state.material_field = crate::material::MaterialField::Surface;
        crate::material::cycle_enum(&mut state, 1); // none → wall_solid
        crate::material::cycle_enum(&mut state, 1); // wall_solid → wall_trace

        let text = capture_and_serialize(&mut state);
        // The writer fix must emit the previously-dropped keys.
        assert!(
            text.contains("reflective = true"),
            "reflective written:\n{text}"
        );
        assert!(text.contains("clarity ="), "clarity written");
        assert!(text.contains("preset = metal"), "preset written");
        assert!(text.contains("rank = 42"), "rank written");
        assert!(text.contains("animation = pulse"), "animation written");
        assert!(text.contains("surface = wall_trace"), "surface written");

        let reloaded = re_level::load_string(&text, "material-all-rt.lvl").expect("reparse");
        let midx = reloaded.material_index(&name).expect("name round-trips");
        let m = &reloaded.materials[midx];
        assert_eq!(m.fg & 0xFF_FFFF, 0x20FF80);
        assert_eq!(m.glyph, Some('+' as u32));
        match m.recipe {
            Some(MaterialRecipe::Brick {
                aspect,
                mortar_w,
                grain,
            }) => {
                assert!((aspect - 3.5).abs() < 1e-4, "brick aspect round-trips");
                assert!((mortar_w - 0.22).abs() < 1e-4, "brick mortar_w round-trips");
                assert!((grain - 0.9).abs() < 1e-4, "brick grain round-trips");
            }
            other => panic!("recipe variant + params must round-trip, got {other:?}"),
        }
        assert_eq!(m.anim.kind, MaterialAnimKind::Pulse);
        assert!((m.anim.speed - 2.5).abs() < 1e-4, "anim speed round-trips");
        assert!(
            (m.anim.amplitude - 0.6).abs() < 1e-4,
            "anim amplitude round-trips"
        );
        assert_eq!(m.rank, 42, "rank round-trips");
        assert!(m.reflective, "reflective round-trips");
        assert!((m.clarity - 0.4).abs() < 1e-4, "clarity round-trips");
        assert_eq!(m.preset.as_deref(), Some("metal"), "preset round-trips");
        // Surface binding round-trips.
        assert_eq!(
            reloaded.surface_materials.wall_trace.as_deref(),
            Some(name.as_str()),
            "surface binding round-trips"
        );
    }

    /// TRIGGER round-trip: place a trigger through the object path, set its region
    /// corners + action (ActivateNode to a node) + once, save, reload through the
    /// real parser, and assert the trigger survives with the right region / action /
    /// once. Triggers serialize via `[trigger]` (`re_voximport::write_trigger`); the
    /// `ActivateNode` name resolves back to a node index in `re_level::finalize`.
    #[test]
    fn place_trigger_set_region_action_once_save_reload_preserves_it() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(12, 8, 4);
        // Some geometry so the cells path is exercised alongside the trigger.
        state.map.set(0, 0, 0, CellKind::Floor);

        // A node the trigger will drive (must be an INPUT — re_level::finalize
        // enforces ActivateNode targets an input node).
        state.place = Some(EntityKind::Node);
        state.cursor = (8, 4);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither -> input
        let node_name = state.level.nodes[0].name.clone();

        // Place a trigger at (3,3,1); it defaults to ActivateNode(node0).
        state.place = Some(EntityKind::Trigger);
        state.cursor = (3, 3);
        state.z = 1;
        entity::place_at_cursor(&mut state);

        // Grow the region into a 3x2x1 box: max X +2, max Y +1.
        state.entity_field = EntityField::TrigMaxX;
        entity::adjust_property(&mut state, 2); // x 3 -> 5
        state.entity_field = EntityField::TrigMaxY;
        entity::adjust_property(&mut state, 1); // y 3 -> 4

        // once = false (toggle off the default true).
        state.entity_field = EntityField::TrigOnce;
        entity::cycle_property(&mut state);
        assert!(!state.level.triggers[0].once);

        let want_region = state.level.triggers[0].region.clone();

        // Save (no disk; path is None) and reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[trigger]"),
            "save emits a [trigger] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "trigger-rt.lvl").expect("reparse");

        assert_eq!(reloaded.triggers.len(), 1, "one trigger round-trips");
        let t = &reloaded.triggers[0];
        // Region survived exactly (normalised box corners).
        assert_eq!(t.region, want_region, "trigger region round-trips");
        // once survived.
        assert!(!t.once, "trigger once = false round-trips");
        // Action: ActivateNode resolved back to the SAME node (by name + index).
        let want_idx = reloaded.node_index(&node_name).expect("node round-trips");
        match &t.action {
            re_core::TriggerAction::ActivateNode { name, node_index } => {
                assert_eq!(name, &node_name, "trigger target name round-trips");
                assert_eq!(*node_index, want_idx, "trigger target index resolves");
            }
            other => panic!("expected ActivateNode, got {other:?}"),
        }

        // The painted floor still round-trips (the trigger didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// ZONE INPUT round-trip: place an input node, place a zone input through the
    /// object path (it auto-links the first input node), grow its region + link the
    /// input node via the focused-field editor, save, reload through the real
    /// parser, and assert the zone survives with the right name / linked node /
    /// region. Zone inputs serialize via `[zone_input]`
    /// (`re_voximport::write_zone_input`); `node` resolves back to a node INDEX in
    /// `re_level::finalize` (which requires the named node to be an INPUT node — so
    /// the test must link an existing input node for a lossless round-trip).
    #[test]
    fn place_zone_input_set_region_node_save_reload_preserves_it() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(12, 10, 4);
        // Some geometry so the cells path is exercised alongside the zone.
        state.map.set(0, 0, 0, CellKind::Floor);

        // An INPUT node the zone will drive (re_level::finalize enforces the
        // zone's `node` names an input node).
        state.place = Some(EntityKind::Node);
        state.cursor = (9, 5);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither -> input
        let node_name = state.level.nodes[0].name.clone();
        assert!(state.level.nodes[0].is_input, "node is an input");

        // Place a zone input at (2,2,1); it auto-links the first (only) input node.
        state.place = Some(EntityKind::ZoneInput);
        state.cursor = (2, 2);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.zone_inputs.len(), 1);
        let zone_name = state.level.zone_inputs[0].name.clone();
        assert_eq!(state.level.zone_inputs[0].node, node_name, "auto-linked");

        // Grow the region into a 3x2x1 box: max X +2, max Y +1.
        state.entity_field = EntityField::ZoneMaxX;
        entity::adjust_property(&mut state, 2); // x 2 -> 4
        state.entity_field = EntityField::ZoneMaxY;
        entity::adjust_property(&mut state, 1); // y 2 -> 3
        let want_corners = {
            let zi = &state.level.zone_inputs[0];
            (zi.x0, zi.y0, zi.z0, zi.x1, zi.y1, zi.z1)
        };
        assert_eq!(want_corners, (2, 2, 1, 4, 3, 1));

        // Re-affirm the link via the ZoneNode field (cycles through input nodes).
        state.entity_field = EntityField::ZoneNode;
        entity::cycle_property(&mut state); // single input node -> stays linked

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[zone_input"),
            "save emits a [zone_input] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "zone-rt.lvl").expect("reparse");

        assert_eq!(reloaded.zone_inputs.len(), 1, "one zone input round-trips");
        let zi = &reloaded.zone_inputs[0];
        assert_eq!(zi.name, zone_name, "zone name round-trips");
        assert_eq!(zi.node, node_name, "zone linked-node name round-trips");
        // node_index re-resolves on load to the named INPUT node.
        assert_eq!(
            zi.node_index,
            reloaded.node_index(&node_name).expect("node round-trips"),
            "zone node_index re-resolves"
        );
        assert_eq!(
            (zi.x0, zi.y0, zi.z0, zi.x1, zi.y1, zi.z1),
            want_corners,
            "zone region round-trips"
        );

        // The painted floor still round-trips (the zone didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// WALL DECAL round-trip: place a wall decal through the object path, set its
    /// surface (cycle to `solid`), grow its region, set density, and step both
    /// colours via the focused-field editor, then assert IN MEMORY that every field
    /// took. Save + reload through the real parser and assert that ALL fields —
    /// surface, region, density, color_head, color_tail — SURVIVE the round-trip.
    ///
    /// The earlier `re_voximport::to_lvl_text` gap (it used to emit only
    /// `surface` + `region`, dropping `density`/`color_head`/`color_tail`) is now
    /// CLOSED: the writer emits those fields when non-default, so they round-trip.
    /// This test asserts all of them post-reload.
    #[test]
    fn place_wall_decal_set_fields_save_reload_preserves_surface_and_region() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};
        use re_world::CellKind as CK;

        let mut state = EditorState::new_empty(12, 10, 4);
        // Some geometry so the cells path is exercised alongside the decal.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Place a wall decal at (2,2,1) via the object path.
        state.place = Some(EntityKind::WallDecal);
        state.cursor = (2, 2);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.decals.len(), 1);
        assert_eq!(state.selected_decal, Some(0));
        let decal_name = state.level.decals[0].name.clone();
        // A fresh decal is a `trace` surface, 1×1×1, density 1.0.
        assert_eq!(state.level.decals[0].surface_kind, CK::WallTrace);

        // Surface: cycle trace -> matrix -> solid (two cycles).
        state.entity_field = EntityField::DecalSurface;
        entity::cycle_property(&mut state); // trace -> matrix
        entity::cycle_property(&mut state); // matrix -> solid
        assert_eq!(state.level.decals[0].surface_kind, CK::WallSolid);

        // Grow the region into a 3×2×1 box: max X +2, max Y +1.
        state.entity_field = EntityField::DecalMaxX;
        entity::adjust_property(&mut state, 2); // x 2 -> 4
        state.entity_field = EntityField::DecalMaxY;
        entity::adjust_property(&mut state, 1); // y 2 -> 3
        let want_corners = {
            let d = &state.level.decals[0];
            (d.x0, d.y0, d.z0, d.x1, d.y1, d.z1)
        };
        assert_eq!(want_corners, (2, 2, 1, 4, 3, 1));

        // Density: step down from 1.0 by 0.3 (three -1 steps of 0.1).
        state.entity_field = EntityField::DecalDensity;
        entity::adjust_property(&mut state, -1);
        entity::adjust_property(&mut state, -1);
        entity::adjust_property(&mut state, -1);
        assert!(
            (state.level.decals[0].density - 0.7).abs() < 1e-6,
            "density stepped to 0.7 in memory"
        );

        // Colours: the placement defaults are OFF the shared palette, so the first
        // palette step snaps each to a palette swatch. Step head once and tail twice
        // so they land on DIFFERENT swatches, proving the IN-MEMORY colour edit path.
        let def_head = state.level.decals[0].color_head;
        let def_tail = state.level.decals[0].color_tail;
        state.entity_field = EntityField::DecalColorHead;
        entity::adjust_property(&mut state, 1);
        let want_head = state.level.decals[0].color_head;
        state.entity_field = EntityField::DecalColorTail;
        entity::adjust_property(&mut state, 1);
        entity::adjust_property(&mut state, 1);
        let want_tail = state.level.decals[0].color_tail;
        // Each colour moved off its placement default (the edit took effect) …
        assert_ne!(want_head, def_head, "color head changed from default");
        assert_ne!(want_tail, def_tail, "color tail changed from default");
        // … and they landed on different swatches (stepped different distances).
        assert_ne!(want_head, want_tail, "head and tail differ after stepping");

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[wall_decal"),
            "save emits a [wall_decal] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "decal-rt.lvl").expect("reparse");

        assert_eq!(reloaded.decals.len(), 1, "one wall decal round-trips");
        let d = &reloaded.decals[0];
        assert_eq!(d.name, decal_name, "decal name round-trips");
        // surface SURVIVES (the serializer writes it).
        assert_eq!(d.surface_kind, CK::WallSolid, "decal surface round-trips");
        // region SURVIVES (the serializer writes it).
        assert_eq!(
            (d.x0, d.y0, d.z0, d.x1, d.y1, d.z1),
            want_corners,
            "decal region round-trips"
        );
        // density + colours now SURVIVE reload (the re_voximport writer gap was
        // closed: it emits density/color_head/color_tail when non-default).
        assert!(
            (d.density - 0.7).abs() < 1e-6,
            "decal density round-trips (was {})",
            d.density
        );
        assert_eq!(d.color_head, want_head, "decal color_head round-trips");
        assert_eq!(d.color_tail, want_tail, "decal color_tail round-trips");

        // The painted floor still round-trips (the decal didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// NOTE round-trip: place a note through the object path, set its kind, facing,
    /// and position via the focused-field editor, type a multi-word body through the
    /// key dispatcher (proving the note-text routing), save, reload through the real
    /// parser, and assert the note survives with the right kind / facing / pos /
    /// text. Notes serialize via `[note ...]` (the note-writing loop in
    /// `re_voximport::to_lvl_text`); the body is captured verbatim.
    #[test]
    fn place_note_set_fields_type_body_save_reload_preserves_it() {
        use crate::edit::apply_key;
        use crate::entity;
        use crate::state::{EntityField, EntityKind};
        use re_present::{EditorKey, EditorKeyCode as E};

        let mut state = EditorState::new_empty(10, 10, 4);
        // Some geometry so the cells path is exercised alongside the note.
        state.map.set(0, 0, 0, CellKind::Floor);

        // Place a note at (4,5,2) via the object path.
        state.place = Some(EntityKind::Note);
        state.cursor = (4, 5);
        state.z = 2;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.notes.len(), 1);
        assert_eq!(state.selected_note, Some(0));

        // Kind: Researcher -> SystemLog (cycle once).
        state.entity_field = EntityField::NoteKindField;
        entity::cycle_property(&mut state);
        let want_kind = state.level.notes[0].kind;
        assert_eq!(want_kind, re_level::NoteKind::SystemLog);

        // Facing: +y -> +x (cycle once).
        state.entity_field = EntityField::NoteFacing;
        entity::cycle_property(&mut state);
        let want_facing = state.level.notes[0].facing;
        assert_ne!(want_facing, 0, "facing changed off the default");

        // Pos: nudge x +1 (4 -> 5).
        state.entity_field = EntityField::NotePosX;
        entity::adjust_property(&mut state, 1);
        let want_pos = {
            let n = &state.level.notes[0];
            (n.x, n.y, n.z)
        };
        assert_eq!(want_pos, (5, 5, 2));

        // Body: focus the text field, clear the default, then TYPE a body through
        // the real key dispatcher (proving printable keys route to the note body).
        state.entity_field = EntityField::NoteText;
        // Clear the default "note" body (4 backspaces).
        let key = |code| EditorKey { code, shift: false };
        for _ in 0..4 {
            apply_key(&mut state, key(E::Backspace));
        }
        assert!(
            state.level.notes[0].text.is_empty(),
            "default body cleared by typing"
        );
        for c in "hi there".chars() {
            apply_key(&mut state, key(E::Char(c)));
        }
        let want_text = state.level.notes[0].text.clone();
        assert_eq!(
            want_text, "hi there",
            "typed body (with a space) is captured"
        );

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[note "),
            "save emits a [note] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "note-rt.lvl").expect("reparse");

        assert_eq!(reloaded.notes.len(), 1, "one note round-trips");
        let n = &reloaded.notes[0];
        assert_eq!(n.kind, want_kind, "note kind round-trips");
        assert_eq!(n.facing, want_facing, "note facing round-trips");
        assert_eq!((n.x, n.y, n.z), want_pos, "note pos round-trips");
        // The note body is captured verbatim up to TRAILING whitespace: the writer
        // appends a blank separator line that the verbatim-body parser re-absorbs on
        // the first reparse (`"hi there"` -> `"hi there\n"`). The writer's `trim_end`
        // makes this a stable fixed point on the next save, so the SECOND serialize
        // is idempotent — that is the format's documented behaviour.
        assert_eq!(
            n.text.trim_end(),
            want_text,
            "note body text round-trips (sans trailing ws)"
        );
        let out2 = re_voximport::to_lvl_text(&reloaded);
        let reloaded2 = re_level::load_string(&out2, "note-rt2.lvl").expect("reparse2");
        assert_eq!(
            reloaded2.notes[0].text, reloaded.notes[0].text,
            "note body reaches an idempotent fixed point on the second round-trip"
        );

        // The painted floor still round-trips (the note didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// PART + EMIT round-trip: place a node + a part through the object path, edit
    /// the part's size (3×2×2) / offset / colour via the focused-field editor, carve
    /// a NON-TRIVIAL mask (some 0s, some 1s) through the `part.rs` toggle helpers,
    /// then link the node→part via the Node `NodeEmits` `c` toggle. Save, reload
    /// through the real parser, and assert the part name / size / offset / colour /
    /// the FULL mask bits and the node→part emit mapping all survive. Proves the
    /// editor's `[part]` + `[emit]` authoring round-trips losslessly through
    /// `re_voximport::to_lvl_text` ↔ `re_level`.
    #[test]
    fn place_part_edit_mask_and_emit_save_reload_preserves_them() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(12, 12, 4);
        // Some geometry so the cells path is exercised alongside the part/emit.
        state.map.set(0, 0, 0, CellKind::Floor);

        // A node that will light the part (place it first so it exists at index 0).
        state.place = Some(EntityKind::Node);
        state.cursor = (2, 2);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        let node_name = state.level.nodes[0].name.clone();

        // Place a part at anchor (5, 6, 1) via the object path.
        state.place = Some(EntityKind::Part);
        state.cursor = (5, 6);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.parts.len(), 1);
        assert_eq!(state.selected_part, Some(0));
        let part_name = state.level.parts[0].name.clone();

        // Size: default 2×2×2 → 3×2×2 (grow w by 1). The size edit resizes the mask.
        state.entity_field = EntityField::PartSizeW;
        entity::adjust_property(&mut state, 1);
        assert_eq!(
            (
                state.level.parts[0].w,
                state.level.parts[0].h,
                state.level.parts[0].d
            ),
            (3, 2, 2),
            "part is 3x2x2 after the size edit"
        );

        // Offset: nudge offX +1 (5 -> 6); colour: step it off white.
        state.entity_field = EntityField::PartOffX;
        entity::adjust_property(&mut state, 1);
        state.entity_field = EntityField::PartColor;
        entity::adjust_property(&mut state, 1);
        let want_off = {
            let p = &state.level.parts[0];
            (p.off_x, p.off_y, p.off_z)
        };
        assert_eq!(want_off, (6, 6, 1));
        let want_color = state.level.parts[0].color_rgb & 0xFF_FFFF;

        // NON-TRIVIAL mask: start all-solid (12 cells), carve a few holes via the
        // SSOT `part.rs` toggle helpers so the mask is a deliberate mix of 0s + 1s.
        crate::part::set_mask(&mut state.level.parts[0], 0, 0, 0, false);
        crate::part::set_mask(&mut state.level.parts[0], 2, 1, 0, false);
        crate::part::set_mask(&mut state.level.parts[0], 1, 0, 1, false);
        let want_mask = state.level.parts[0].mask.clone();
        // Sanity: a genuine mix (not all-solid, not all-empty).
        assert!(want_mask.contains(&0), "mask has empty cells");
        assert!(want_mask.contains(&1), "mask has solid cells");

        // EMIT: focus the Node's emits field, point the emit cursor at part 0, and
        // toggle the clause ON via `c` (the SAME path the keyboard uses).
        state.place = Some(EntityKind::Node);
        state.selected_node = Some(0);
        state.entity_field = EntityField::NodeEmits;
        state.emit_part_cursor = 0;
        entity::cycle_property(&mut state); // add node -> part clause
        assert_eq!(state.level.emits.len(), 1, "one emit clause added");

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[part "),
            "save emits a [part] section:\n{text}"
        );
        assert!(
            text.contains("[emit "),
            "save emits an [emit] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "part-rt.lvl").expect("reparse");

        // The part survives with name / size / offset / colour intact.
        assert_eq!(reloaded.parts.len(), 1, "one part round-trips");
        let pi = reloaded
            .parts
            .iter()
            .position(|p| p.name == part_name)
            .expect("part name round-trips");
        let p = &reloaded.parts[pi];
        assert_eq!((p.w, p.h, p.d), (3, 2, 2), "part size round-trips");
        assert_eq!(
            (p.off_x, p.off_y, p.off_z),
            want_off,
            "part offset round-trips"
        );
        assert_eq!(
            p.color_rgb & 0xFF_FFFF,
            want_color,
            "part colour round-trips"
        );
        // The FULL mask survives bit-for-bit.
        assert_eq!(p.mask, want_mask, "part mask round-trips bit-for-bit");

        // The node→part emit mapping survives.
        let ni = reloaded
            .node_index(&node_name)
            .expect("node name round-trips");
        assert!(
            reloaded
                .emits
                .iter()
                .any(|e| e.source_node == ni && e.part == pi),
            "node->part emit clause round-trips: emits={:?}",
            reloaded.emits
        );

        // The painted floor still round-trips (part/emit didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// STAIRCASE STAMP TOOL round-trip: select the tool through the keyboard (`8`),
    /// stamp a straight staircase with `space`, save, reload through the real parser,
    /// and assert the generator's RampZ treads + landing survive the editor's `[cells]`
    /// save path cell-for-cell. Proves the staircase tool writes through the
    /// same lossless geometry path every brush uses.
    #[test]
    fn stamp_straight_staircase_save_reload_preserves_ramp_cells() {
        use crate::edit::apply_key;
        use re_present::{EditorKey, EditorKeyCode as E};

        let h = 4usize; // matches the default storey height a fresh level reports.
        let mut state = EditorState::new_empty(h + 4, 6, h + 4);
        // Select the staircase tool (straight is the default sub-mode), dir +X.
        let key = |code| EditorKey { code, shift: false };
        apply_key(&mut state, key(E::Char('8')));
        assert!(state.stair_mode, "8 selects the staircase tool");
        state.stair_dir = re_world::RampDir::PosX;
        state.cursor = (0, 2);
        state.z = 0;
        // Stamp the whole staircase in one undoable action.
        apply_key(&mut state, key(E::Char(' ')));
        assert!(state.dirty, "stamping marks dirty");

        // Treads laid as the generator proves climbable: (n,2,n) RampZ(+X).
        for n in 1..=h as i32 {
            assert_eq!(state.map.cell_at(n, 2, n), CellKind::WallSolid);
            assert_eq!(state.map.shape_at(n, 2, n), Shape::RampZ(RampDir::PosX));
        }

        // Full editor save path → reload → rebuild; every stamped cell survives.
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "stair-rt.lvl").expect("reparse");
        let rebuilt = reloaded.build_map();
        for n in 1..=h as i32 {
            assert_eq!(
                rebuilt.cell_at(n, 2, n),
                CellKind::WallSolid,
                "tread {n} kind round-trips"
            );
            assert_eq!(
                rebuilt.shape_at(n, 2, n),
                Shape::RampZ(RampDir::PosX),
                "tread {n} ramp shape round-trips"
            );
        }
        // Landing one step past the top tread, flat solid on the upper floor.
        assert_eq!(
            rebuilt.cell_at(h as i32 + 1, 2, h as i32),
            CellKind::WallSolid
        );
        assert_eq!(
            rebuilt.shape_at(h as i32 + 1, 2, h as i32),
            Shape::Full,
            "landing flat"
        );
    }

    /// The staircase stamp is a SINGLE undoable edit: after stamping, one `u`
    /// reverts every cell the generator wrote (proves the tool routes through the
    /// shared snapshot choke point, not a bypass).
    #[test]
    fn stamp_staircase_is_one_undoable_edit() {
        use crate::edit::apply_key;
        use re_present::{EditorKey, EditorKeyCode as E};

        let mut state = EditorState::new_empty(8, 6, 8);
        let key = |code| EditorKey { code, shift: false };
        apply_key(&mut state, key(E::Char('8')));
        state.stair_dir = re_world::RampDir::PosX;
        state.cursor = (0, 2);
        state.z = 0;
        apply_key(&mut state, key(E::Char(' '))); // stamp
        assert_eq!(
            state.history.undo_len(),
            1,
            "the whole stamp is ONE undo step"
        );
        assert_eq!(
            state.map.cell_at(1, 2, 1),
            CellKind::WallSolid,
            "tread present"
        );

        apply_key(&mut state, key(E::Char('u'))); // one undo reverts the WHOLE stamp
        for n in 1..=4i32 {
            assert_eq!(
                state.map.cell_at(n, 2, n),
                CellKind::Empty,
                "tread {n} reverted"
            );
            assert_eq!(
                state.map.shape_at(n, 2, n),
                Shape::Full,
                "tread {n} shape reverted"
            );
        }
        // Redo re-lays it.
        apply_key(&mut state, key(E::Char('y')));
        assert_eq!(
            state.map.cell_at(1, 2, 1),
            CellKind::WallSolid,
            "redo re-stamps"
        );
    }

    /// A placed vox stamp's geometry shows up in `state.map` after the rebuild
    /// (so it is visible in the top-down + preview), AND it survives even with
    /// NO real file (the path string round-trips; the stamp is a best-effort
    /// skip at build_map time). This proves placement + rebuild don't require a
    /// readable model to keep editing.
    #[test]
    fn place_vox_with_unknown_path_still_records_and_round_trips() {
        use crate::vox;

        let mut state = EditorState::new_empty(8, 8, 4);
        state.vox_files = vec!["levels/does-not-exist.vox".into()];
        state.vox_file_sel = 0;
        state.vox_mode = true;
        state.cursor = (2, 2);
        state.z = 0;
        vox::place_at_cursor(&mut state); // best-effort stamp; no panic

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "vox-missing-rt.lvl").expect("reparse");
        assert_eq!(
            reloaded.voxes.len(),
            1,
            "stamp recorded even for a missing file"
        );
        assert_eq!(reloaded.voxes[0].file, "levels/does-not-exist.vox");
    }

    /// B1 round-trip: deleting an INPUT node whose initial bit is set must also drop
    /// its `[inputs]` entry (keyed by NAME, which `remove_node` can't touch). Without
    /// the cleanup the serializer emits `[inputs] <name> = 1` for a node that no
    /// longer exists and `re_level` finalize rejects the reload. Build through the
    /// editor's place + settings + delete paths, serialize, and assert (a) no
    /// dangling `[inputs]` entry and (b) the level reloads.
    #[test]
    fn delete_input_node_drops_dangling_inputs_entry_and_reloads() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(8, 8, 4);
        // Place a node and cycle it to INPUT via the focused-field IO cycle.
        state.place = Some(EntityKind::Node);
        state.cursor = (2, 2);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither → input
        assert!(state.level.nodes[0].is_input);
        let node_name = state.level.nodes[0].name.clone();

        // Set its initial bit ON through the settings form (writes `[inputs]`).
        crate::settings::open(&mut state);
        {
            let s = state.settings.as_mut().unwrap();
            s.toggle_input(); // single input row, cursor 0
        }
        crate::settings::apply(&mut state);
        assert!(
            state
                .level
                .inputs
                .iter()
                .any(|(n, v)| n == &node_name && *v == 1),
            "input bit recorded"
        );

        // Delete the node at its cell (the focused-IO delete path).
        state.place = Some(EntityKind::Node);
        state.cursor = (2, 2);
        state.z = 0;
        entity::delete_at_cursor(&mut state);
        assert!(state.level.nodes.is_empty(), "node removed");
        assert!(
            !state.level.inputs.iter().any(|(n, _)| n == &node_name),
            "B1: dangling [inputs] entry removed"
        );

        let text = capture_and_serialize(&mut state);
        assert!(
            !text.contains(&format!("{node_name} = 1")),
            "serialized [inputs] has no dangling entry"
        );
        re_level::load_string(&text, "b1-delete-input.lvl")
            .expect("B1: level reloads after deleting an input node");
    }

    /// B2 round-trip: cycling a node OUT of `is_input` while a `[zone_input]` links it
    /// by name must NOT produce an unreloadable level. The editor REFUSES the cycle
    /// (least destructive — keeps the node an input and posts a status). Assert the
    /// node stays an input, the zone still links a real input node, and the level
    /// reloads.
    #[test]
    fn cycle_io_refused_when_zone_input_depends_and_level_reloads() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(8, 8, 4);
        // Place a node and make it an INPUT.
        state.place = Some(EntityKind::Node);
        state.cursor = (3, 3);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state); // neither → input
        assert!(state.level.nodes[0].is_input);
        let node_name = state.level.nodes[0].name.clone();

        // Place a ZoneInput on a different cell — it auto-links the first input node.
        state.place = Some(EntityKind::ZoneInput);
        state.cursor = (5, 5);
        state.z = 0;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.zone_inputs.len(), 1);
        assert_eq!(
            state.level.zone_inputs[0].node, node_name,
            "zone links the input"
        );

        // Re-select the node and attempt to cycle its IO off input → REFUSED.
        state.place = Some(EntityKind::Node);
        state.cursor = (3, 3);
        state.z = 0;
        entity::place_at_cursor(&mut state); // re-selects node 0
        state.entity_field = EntityField::NodeIo;
        entity::cycle_property(&mut state);
        assert!(
            state.level.nodes[0].is_input && !state.level.nodes[0].is_output,
            "B2: cycle refused — node stays an input"
        );
        assert!(
            state.status.contains("zone_input"),
            "status explains the refusal"
        );

        // The zone still points at a genuine input node → finalize accepts the reload.
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "b2-cycle-io.lvl")
            .expect("B2: level reloads (zone still links an input node)");
        let zi = &reloaded.zone_inputs[0];
        assert!(
            reloaded.nodes[zi.node_index].is_input,
            "zone_input resolves to an input node after reload"
        );
    }

    /// LEVEL SETTINGS round-trip: open the settings form, author meta/room/input/
    /// sequence through its helpers, apply, serialize, reload through the real
    /// parser, and assert every authored field survives. Proves the editor's
    /// settings modal reaches the saved `.lvl` losslessly (and that the authored
    /// sequence satisfies `re_level` finalize — distinct steps, observe subset).
    #[test]
    fn settings_roundtrip_preserves_meta_room_inputs_sequence() {
        let mut state = EditorState::new_empty(8, 8, 4);

        // One INPUT node + two OUTPUT nodes for the observe window. push_node
        // auto-names n0/n1/...; overwrite to known unique names.
        let i0 = state.level.push_node(1, 1, 0);
        state.level.nodes[i0].is_input = true;
        state.level.nodes[i0].name = "drive".into();
        let a = state.level.push_node(2, 2, 0);
        state.level.nodes[a].is_output = true;
        state.level.nodes[a].name = "outA".into();
        let b = state.level.push_node(3, 3, 0);
        state.level.nodes[b].is_output = true;
        state.level.nodes[b].name = "outB".into();
        let input_name = state.level.nodes[i0].name.clone();

        crate::settings::open(&mut state);
        {
            let s = state.settings.as_mut().unwrap();
            // Meta: name != title. (open seeds name="untitled"; pop it first.)
            while !s.name.is_empty() {
                s.name_pop();
            }
            for c in "vault".chars() {
                s.name_push(c);
            }
            while !s.title.is_empty() {
                s.title_pop();
            }
            for c in "The Vault".chars() {
                s.title_push(c);
            }
            // Room: matrix ambient, storey 6, both flags on. Cycle ambient to
            // "matrix" (index 2 in AMBIENT_CYCLE) from the default.
            while s.ambient() != "matrix" {
                s.ambient_cycle(1);
            }
            s.storey_height = 1;
            s.storey_adjust(5); // -> 6
            s.toggle_cycle_keep();
            s.toggle_render_pipes();
            // Input: toggle the single input ON (cursor starts at 0).
            s.toggle_input();
            // Sequence: built PURELY through the keyboard helpers (no direct field
            // assignment) — proves the authoring keystrokes reach the saved .lvl.
            // `add_observe` now prefers OUTPUT nodes, so two adds give [outA, outB]
            // (the input "drive" is never auto-picked for the observe window).
            s.focus = crate::settings::SettingsField::Observe;
            s.add_observe(); // outA
            s.add_observe(); // outB
                             // Steps = [[outA], [outB]] (distinct), built via add_step/add_step_name.
            s.add_step();
            s.add_step_name(); // outA
            s.add_step();
            s.add_step_name(); // outA
            s.add_step_name(); // outB
            s.name_cursor = 0;
            s.remove_step_name(); // drop outA → step 2 = [outB]
        }
        crate::settings::apply(&mut state);

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "settings-rt.lvl").expect("reparse");

        assert_eq!(reloaded.name, "vault");
        assert_eq!(reloaded.title, "The Vault");
        assert_ne!(reloaded.name, reloaded.title);
        assert_eq!(reloaded.ambient, "matrix");
        assert!(reloaded.render_pipes);
        assert!(reloaded.cycle_keep);
        assert_eq!(reloaded.storey_height(), 6, "storey height round-trips");
        assert!(
            reloaded
                .inputs
                .iter()
                .any(|(n, v)| n == &input_name && *v == 1),
            "input initial-on bit round-trips"
        );
        assert_eq!(
            reloaded.sequence.observe_names,
            vec!["outA".to_string(), "outB".to_string()]
        );
        assert_eq!(
            reloaded.sequence.step_names,
            vec![vec!["outA".to_string()], vec!["outB".to_string()]]
        );
    }

    /// ATMOSPHERE round-trip: author the `[effects]` / `[palette]` (fog) / `[sky]`
    /// sections through the settings form's helpers, apply, serialize, reload through
    /// the real parser, and assert every value survives. Proves the editor's new
    /// atmosphere authoring reaches the saved `.lvl` losslessly.
    #[test]
    fn settings_roundtrip_preserves_effects_fog_sky() {
        use crate::settings::SettingsField as F;

        let mut state = EditorState::new_empty(8, 8, 4);
        crate::settings::open(&mut state);
        {
            let s = state.settings.as_mut().unwrap();
            // [effects]: snow 0.5, steam on, matrix 0.3, glitch 0.2 — set via helpers.
            s.focus = F::Snow;
            for _ in 0..5 {
                s.snow_adjust(1);
            }
            s.toggle_steam();
            for _ in 0..3 {
                s.matrix_adjust(1);
            }
            for _ in 0..2 {
                s.glitch_adjust(1);
            }
            // [palette] fog: a hex colour + a distance pair (near 2.0, far 9.0).
            for c in "#102030".chars() {
                s.fog_color_push(c);
            }
            for _ in 0..4 {
                s.fog_near_adjust(1); // +0.5 each → 2.0
            }
            for _ in 0..18 {
                s.fog_far_adjust(1); // +0.5 each → 9.0
            }
            // [sky]: three image paths.
            s.sky_push(F::SkyCylinder, 'a'); // build "tex/cyl.png" by typing
            s.sky_pop(F::SkyCylinder);
            s.sky_cylinder = "tex/cyl.png".into();
            s.sky_cap_top = "tex/top.png".into();
            s.sky_cap_bottom = "tex/bot.png".into();
        }
        crate::settings::apply(&mut state);

        let text = capture_and_serialize(&mut state);
        assert!(text.contains("[effects]"), "save emits [effects]:\n{text}");
        assert!(text.contains("[palette]"), "save emits [palette]:\n{text}");
        assert!(text.contains("[sky]"), "save emits [sky]:\n{text}");
        let reloaded = re_level::load_string(&text, "atmo-rt.lvl").expect("reparse");

        assert!(
            (reloaded.effects.snow - 0.5).abs() < 1e-4,
            "snow round-trips"
        );
        assert!(reloaded.effects.steam, "steam round-trips");
        assert!(
            (reloaded.effects.matrix - 0.3).abs() < 1e-4,
            "matrix round-trips"
        );
        assert!(
            (reloaded.effects.glitch - 0.2).abs() < 1e-4,
            "glitch round-trips"
        );

        assert!(reloaded.palette.has_fog, "fog tint authored");
        assert_eq!(
            reloaded.palette.fog_color & 0xFF_FFFF,
            0x10_2030,
            "fog_color round-trips"
        );
        assert!(reloaded.palette.has_fog_dist, "fog distance authored");
        assert!(
            (reloaded.palette.fog_near - 2.0).abs() < 1e-4,
            "fog_near round-trips"
        );
        assert!(
            (reloaded.palette.fog_far - 9.0).abs() < 1e-4,
            "fog_far round-trips"
        );

        assert_eq!(
            reloaded.sky.cylinder.as_deref(),
            Some("tex/cyl.png"),
            "sky cylinder round-trips"
        );
        assert_eq!(
            reloaded.sky.cap_top.as_deref(),
            Some("tex/top.png"),
            "sky cap_top round-trips"
        );
        assert_eq!(
            reloaded.sky.cap_bottom.as_deref(),
            Some("tex/bot.png"),
            "sky cap_bottom round-trips"
        );
    }

    /// `[room] power_budget` round-trips through the SETTINGS FORM,
    /// end to end, on a level that already sets it.
    ///
    /// This closes a real coverage gap: the campaign's shipped levels
    /// set the pipe-power cap, `re_game` shows it in the HUD and enforces it,
    /// and the editor had no field for it at all — it was preserved on save but
    /// silently unauthorable. This drives the REAL key dispatch
    /// (`edit::apply_key`, the same path a keystroke takes), reaches the field
    /// by TABBING to it, TYPES a new value digit by digit, then goes to disk
    /// and back through the real parser.
    ///
    /// The level is an inline crate-local fixture (not a campaign level — the
    /// editor is engine-repo content and the campaign ships only in the game
    /// repo, so this crate must not depend on campaign level files), carrying
    /// the same `power_budget = 5` shape a shipped level would.
    #[test]
    fn settings_form_roundtrips_power_budget_on_a_shipped_level() {
        const FIXTURE: &str = "\
[meta]
name = power-budget-form-fixture
[room]
size = 10 6 3
power_budget = 5
[node A]
pos = 2,2,1
is_input = yes
[node B]
pos = 7,2,1
is_output = yes
[inputs]
A = 1
[pipe A->B]
state = OFF
path = (2,2,1) (7,2,1)
switch_at = (4,2,1)
[pedestal]
center = 7 2 1
volume = 1 1 1
[door]
pos = 1 2 1
target_parts = B
[player]
spawn = 5 4 1
facing = -y
";

        use re_present::{EditorKey, EditorKeyCode as K};
        let press = |state: &mut EditorState, code: K| {
            crate::edit::apply_key(state, EditorKey { code, shift: false });
        };

        let orig = FIXTURE.to_string();
        assert!(
            orig.contains("power_budget = 5"),
            "fixture level must still set power_budget = 5"
        );
        let path = scratch_path("power-budget-form.lvl");
        std::fs::write(&path, &orig).expect("write scratch copy");

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut state, &path),
            "the editor opens the fixture"
        );
        assert_eq!(state.level.power_budget, 5, "the budget came off disk");

        crate::settings::open(&mut state);
        assert_eq!(
            state.settings.as_ref().unwrap().power_budget,
            "5",
            "open seeds the digit buffer from the level"
        );

        // Reach the field the way a user does — Tab, no direct focus assignment.
        let order = crate::settings::SettingsField::ORDER;
        for _ in 0..order.len() {
            if state.settings.as_ref().unwrap().focus == crate::settings::SettingsField::PowerBudget
            {
                break;
            }
            press(&mut state, K::Tab);
        }
        assert_eq!(
            state.settings.as_ref().unwrap().focus,
            crate::settings::SettingsField::PowerBudget,
            "Tab reaches the power_budget field"
        );

        // TYPE the new value. Typed entry is the whole point: reaching 12 from 5
        // by pressing `+` seven times is not a practical authoring method.
        for code in [K::Backspace, K::Char('1'), K::Char('2')] {
            press(&mut state, code);
        }
        assert_eq!(
            state.settings.as_ref().unwrap().power_budget,
            "12",
            "digits typed"
        );
        // A non-digit must NOT enter an integer field.
        press(&mut state, K::Char('q'));
        assert_eq!(
            state.settings.as_ref().unwrap().power_budget,
            "12",
            "a non-digit is rejected, not appended"
        );
        // `-`/`+` still step the same buffer — the field has BOTH input methods.
        press(&mut state, K::Char('+'));
        assert_eq!(
            state.settings.as_ref().unwrap().power_budget,
            "13",
            "+ steps up"
        );
        press(&mut state, K::Char('-'));
        assert_eq!(
            state.settings.as_ref().unwrap().power_budget,
            "12",
            "- steps back"
        );

        // Enter applies the whole form and closes it.
        press(&mut state, K::Enter);
        assert!(state.settings.is_none(), "Enter applied + closed the form");
        assert_eq!(state.level.power_budget, 12, "apply wrote the level field");

        // …and it survives a real save → reparse.
        let v = save_lvl_text(&mut state);
        assert!(
            !v.starts_with("REFUSED") && !v.starts_with("write failed") && v != "no path",
            "save did not succeed: {v}"
        );
        let saved = std::fs::read_to_string(&path).expect("read after save");
        let reloaded = re_level::load_string(&saved, "power-budget-form.lvl")
            .expect("the saved file reparses");
        assert!(
            saved.contains("power_budget = 12"),
            "the saved .lvl carries the new cap"
        );
        assert_eq!(
            reloaded.power_budget, 12,
            "power_budget round-trips through the settings form"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.bak"));
    }

    /// The four remaining INERT metadata keys (`[meta] version`,
    /// `[meta] spec_rev`, `[room] floor`, `[room] ceiling`) plus the per-level
    /// `[door] target_model` are authorable from the form and reach the file.
    ///
    /// Each is free TEXT: `re_level` keeps them VERBATIM and nothing resolves them
    /// against a vocabulary (`[room] floor` is a STYLE tag — the floor's look comes
    /// from a `[material] surface = floor` binding authored on the material), so
    /// there is nothing to validate a value against and nothing to cycle.
    #[test]
    fn settings_form_roundtrips_the_inert_metadata_keys() {
        use crate::settings::SettingsField as F;
        use re_present::{EditorKey, EditorKeyCode as K};

        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        crate::settings::open(&mut state);
        // Type each value through the REAL dispatch, focusing the field first.
        for (field, text) in [
            (F::Version, "2"),
            (F::SpecRev, "2026-08-22"),
            (F::RoomFloor, "grid"),
            (F::RoomCeiling, "dim"),
            (F::DoorTargetModel, "atrium"),
        ] {
            state.settings.as_mut().unwrap().focus = field;
            for c in text.chars() {
                crate::edit::apply_key(
                    &mut state,
                    EditorKey {
                        code: K::Char(c),
                        shift: false,
                    },
                );
            }
        }
        // Backspace pops from the focused text field (still DoorTargetModel).
        crate::edit::apply_key(
            &mut state,
            EditorKey {
                code: K::Backspace,
                shift: false,
            },
        );
        for c in "m".chars() {
            crate::edit::apply_key(
                &mut state,
                EditorKey {
                    code: K::Char(c),
                    shift: false,
                },
            );
        }
        crate::settings::apply(&mut state);

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "inert-rt.lvl").expect("reparse");
        assert_eq!(reloaded.version, "2", "[meta] version round-trips");
        assert_eq!(
            reloaded.spec_rev, "2026-08-22",
            "[meta] spec_rev round-trips"
        );
        assert_eq!(reloaded.room_floor, "grid", "[room] floor round-trips");
        assert_eq!(reloaded.room_ceiling, "dim", "[room] ceiling round-trips");
        assert_eq!(
            reloaded.door.target_model, "atrium",
            "[door] target_model round-trips"
        );
    }

    /// EMIT-ON-CHANGE for the new fields: a level with none of them, opened and
    /// applied through the form WITHOUT touching them, gains no `version` /
    /// `spec_rev` / `floor` / `ceiling` / `power_budget` / `target_model` line.
    /// This is the property `every_shipped_level_is_byte_stable_through_the_editor`
    /// depends on, asserted directly.
    #[test]
    fn settings_apply_adds_no_metadata_lines_when_untouched() {
        let mut baseline = EditorState::new_empty(6, 6, 4);
        baseline.map.set(1, 1, 0, CellKind::WallSolid);
        let before = capture_and_serialize(&mut baseline);

        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        crate::settings::open(&mut state);
        crate::settings::apply(&mut state);
        let after = capture_and_serialize(&mut state);

        assert_eq!(before, after, "an untouched open→apply is byte-identical");
        for key in ["version", "spec_rev", "power_budget", "target_model"] {
            assert!(!after.contains(key), "no spurious `{key}` line:\n{after}");
        }
    }

    /// EMIT-ON-CHANGE byte-identity: a level with NO atmosphere, opened + applied
    /// through the settings form WITHOUT touching any atmosphere field, must NOT gain
    /// a spurious `[effects]`/`[palette]`/`[sky]` section — the serialized text is
    /// identical to a straight serialize of the same level. Guards the
    /// emit-on-change contract documented on `settings::apply`.
    #[test]
    fn settings_apply_with_no_atmosphere_stays_byte_identical() {
        // Build a level with one painted cell (so there's real content), serialize it
        // straight as the baseline.
        let mut baseline = EditorState::new_empty(6, 6, 4);
        baseline.map.set(1, 1, 0, CellKind::WallSolid);
        let before = capture_and_serialize(&mut baseline);
        assert!(!before.contains("[effects]"), "baseline has no [effects]");
        assert!(!before.contains("[palette]"), "baseline has no [palette]");
        assert!(!before.contains("[sky]"), "baseline has no [sky]");

        // A second identical level, run through open → apply with NO atmosphere edits.
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(1, 1, 0, CellKind::WallSolid);
        crate::settings::open(&mut state);
        crate::settings::apply(&mut state);
        let after = capture_and_serialize(&mut state);

        assert!(
            !after.contains("[effects]"),
            "no spurious [effects]:\n{after}"
        );
        assert!(
            !after.contains("[palette]"),
            "no spurious [palette]:\n{after}"
        );
        assert!(!after.contains("[sky]"), "no spurious [sky]:\n{after}");
        assert_eq!(after, before, "no-atmosphere apply stays byte-identical");
    }

    /// HOLOGRAM CELL round-trip: paint a `CellKind::Hologram` cell (glyph `H` in the
    /// `[cells]` section), serialize, reload through the real parser, and assert the
    /// reloaded map carries `Hologram` at that cell. Proves the new Hologram cell
    /// brush reaches the saved file losslessly.
    #[test]
    fn paint_hologram_cell_save_reload_preserves_it() {
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(2, 3, 1, CellKind::Hologram);
        // Some other geometry so the cells path is exercised broadly too.
        state.map.set(0, 0, 0, CellKind::Floor);

        let text = capture_and_serialize(&mut state);
        assert!(state.level.cells.is_some(), "capture must set [cells]");

        let reloaded = re_level::load_string(&text, "holo-cell-rt.lvl").expect("reparse");
        let rebuilt = reloaded.build_map();
        assert_eq!(
            rebuilt.cell_at(2, 3, 1),
            CellKind::Hologram,
            "painted Hologram cell survives save->reload"
        );
    }

    /// `[holo]` OBJECT round-trip: build a level carrying a `LevelHolo` with a
    /// NON-DEFAULT shape (Pyramid), a non-default region, and a non-default style,
    /// serialize via `to_lvl_text`, reload through the real parser, and assert the
    /// shape, all six region coords, and the style all survive. Proves the editor's
    /// `[holo]` placement round-trips losslessly (re_voximport already serializes it).
    #[test]
    fn holo_object_save_reload_preserves_shape_region_style() {
        use re_world::HoloShape;

        let mut state = EditorState::new_empty(8, 8, 4);
        state.level.holos.push(re_level::LevelHolo {
            name: "holo1".to_string(),
            shape: HoloShape::Pyramid,
            x0: 1,
            y0: 2,
            z0: 0,
            x1: 4,
            y1: 5,
            z1: 2,
            style: 3,
        });

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "holo-obj-rt.lvl").expect("reparse");

        assert_eq!(reloaded.holos.len(), 1, "the [holo] section round-trips");
        let h = &reloaded.holos[0];
        assert_eq!(h.shape, HoloShape::Pyramid, "shape round-trips");
        assert_eq!(
            (h.x0, h.y0, h.z0, h.x1, h.y1, h.z1),
            (1, 2, 0, 4, 5, 2),
            "all six region coords round-trip"
        );
        assert_eq!(h.style, 3, "style round-trips");
    }

    /// WARP-PORTAL CELL round-trip: paint a `CellKind::Portal` cell (glyph `@` in
    /// the `[cells]` section), serialize, reload through the real parser, and assert
    /// the rebuilt map carries `Portal` at that cell. Proves the new `0` portal cell
    /// brush reaches the saved file losslessly.
    #[test]
    fn paint_portal_cell_save_reload_preserves_it() {
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map.set(2, 3, 1, CellKind::Portal);
        // Some other geometry so the cells path is exercised broadly too.
        state.map.set(0, 0, 0, CellKind::Floor);

        let text = capture_and_serialize(&mut state);
        assert!(state.level.cells.is_some(), "capture must set [cells]");

        let reloaded = re_level::load_string(&text, "portal-cell-rt.lvl").expect("reparse");
        let rebuilt = reloaded.build_map();
        assert_eq!(
            rebuilt.cell_at(2, 3, 1),
            CellKind::Portal,
            "painted Portal cell survives save->reload"
        );
    }

    /// `[portal]` OBJECT round-trip: place a portal via the editor's placement path,
    /// set Region A + Region B to distinct boxes, set a non-zero `rotate` (45°), and
    /// set `skin` to a material name that exists in the level. Serialize, reload, and
    /// assert exactly ONE portal survives with all twelve region coords, the rotate
    /// (epsilon compare to stay clippy-clean), and the skin name intact. Proves the
    /// editor's paired-region `[portal]` placement round-trips losslessly.
    #[test]
    fn portal_object_save_reload_preserves_both_regions_rotate_skin() {
        use crate::state::{EntityKind, ToolMode};

        let mut state = EditorState::new_empty(10, 10, 4);
        // A material the skin can name (re_voximport only writes `skin` when nonempty,
        // and the parser carries it verbatim — it need not resolve in re_world). Built
        // via the level's own constructor (LevelMaterial has no Default), then renamed.
        let mi = state.level.push_material();
        state.level.materials[mi].name = "glass".to_string();

        // Place a portal at the cursor via the real placement path.
        state.enter_tool_mode(ToolMode::Object);
        state.place = Some(EntityKind::Portal);
        state.cursor = (1, 1);
        state.z = 0;
        crate::entity::place_at_cursor(&mut state);
        assert_eq!(state.level.portals.len(), 1, "placement created one portal");

        // Set both regions to distinct boxes + a non-default rotate + a skin name.
        // The parser requires each region to be a FLAT wall face sitting on a room
        // BOUNDARY, with both faces sharing the same opening shape: Region A is the
        // WEST wall (x=0) and Region B the EAST wall (x=sx-1=9), each a 1×3×2 opening
        // (y spans 2..=4, z spans 0..=1), so they are distinct yet valid + matched.
        let p = &mut state.level.portals[0];
        p.ax0 = 0;
        p.ay0 = 2;
        p.az0 = 0;
        p.ax1 = 0;
        p.ay1 = 4;
        p.az1 = 1;
        p.bx0 = 9;
        p.by0 = 2;
        p.bz0 = 0;
        p.bx1 = 9;
        p.by1 = 4;
        p.bz1 = 1;
        p.rotate = 45.0;
        p.skin = "glass".to_string();

        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "portal-obj-rt.lvl").expect("reparse");

        assert_eq!(
            reloaded.portals.len(),
            1,
            "the [portal] section round-trips"
        );
        let p = &reloaded.portals[0];
        assert_eq!(
            (p.ax0, p.ay0, p.az0, p.ax1, p.ay1, p.az1),
            (0, 2, 0, 0, 4, 1),
            "Region A round-trips"
        );
        assert_eq!(
            (p.bx0, p.by0, p.bz0, p.bx1, p.by1, p.bz1),
            (9, 2, 0, 9, 4, 1),
            "Region B round-trips"
        );
        assert!(
            (p.rotate - 45.0).abs() < 1e-4,
            "rotate round-trips (got {})",
            p.rotate
        );
        assert_eq!(p.skin, "glass", "skin round-trips");
    }

    /// REGRESSION: a portal placed via the REAL default path (no manual
    /// region fix-up) must reload through `re_level::load_string` WITHOUT error.
    /// A default that seeds two 1×1×1 regions — THREE flat axes each — is
    /// rejected by `re_level::finalize` ("got 3 flat axes"), making the saved
    /// .lvl unreloadable by the editor OR the game, so the default must seed
    /// two valid wall faces instead; this test guards that the DEFAULT alone
    /// (nothing hand-edited) is reloadable, which the existing portal test does
    /// NOT cover (it hand-fixes the regions before saving).
    #[test]
    fn default_placed_portal_reloads_without_fixup() {
        use crate::state::{EntityKind, ToolMode};

        let mut state = EditorState::new_empty(10, 10, 4);
        state.enter_tool_mode(ToolMode::Object);
        state.place = Some(EntityKind::Portal);
        state.cursor = (4, 4);
        state.z = 1;
        // Place via the real default path — DO NOT touch the regions afterwards.
        crate::entity::place_at_cursor(&mut state);
        assert_eq!(state.level.portals.len(), 1, "placement created one portal");

        // Save + reload through the SAME parser the game/editor use. The default
        // must survive finalize with no manual region fix-up.
        let text = capture_and_serialize(&mut state);
        let reloaded = re_level::load_string(&text, "portal-default-rt.lvl")
            .expect("a default-placed portal must reload without error");
        assert_eq!(
            reloaded.portals.len(),
            1,
            "the default portal round-trips as exactly one portal"
        );
    }

    /// SCALABLE MATERIAL OVERLAY — ZONE round-trip: place a zone + an inset through
    /// the object path, define a material (so the zone's binding is valid), grow the
    /// zone region, bind the material, set a SelfSimilar scale (arms + twist + warp),
    /// and reference the inset by name — then save, reload through the real parser,
    /// and assert region / material / scale params / inset reference all survive.
    /// NOTE the cross-ref: a zone naming an inset only reloads cleanly when the inset
    /// ALSO exists, so the test creates BOTH (mirrors the zone_input ↔ input-node
    /// cross-ref discipline). The serializer SKIPS default scale + unbound material,
    /// so every asserted field is NON-default and therefore actually serialized.
    #[test]
    fn overlay_zone_save_reload_preserves_region_material_scale_insets() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(16, 12, 4);
        // A bound material so `validate_zones` accepts the zone's binding on reload.
        let mi = state.level.push_material();
        let mat_name = "ore".to_string();
        state.level.materials[mi].name = mat_name.clone();

        // Place an INSET first (the zone will reference it by name) and give it a
        // distinct region so the cross-ref + region both round-trip.
        state.place = Some(EntityKind::Inset);
        state.cursor = (10, 8);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.overlay.insets.len(), 1);
        let inset_name = state.level.overlay.insets[0].name.clone();
        // Give the inset a real child path so it serializes a non-empty reference
        // (typed through the caret-based text router).
        state.entity_field = EntityField::InsetChild;
        for c in "child.lvl".chars() {
            assert!(entity::text_entry_key(
                &mut state,
                re_present::EditorKey {
                    code: re_present::EditorKeyCode::Char(c),
                    shift: false
                }
            ));
        }

        // Place a ZONE at (2,2,1) via the object path.
        state.place = Some(EntityKind::Zone);
        state.cursor = (2, 2);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.overlay.zones.len(), 1);
        let zone_name = state.level.overlay.zones[0].name.clone();

        // Grow the region into a 4×3×1 box: max X +3, max Y +2.
        state.entity_field = EntityField::ZoneRegMaxX;
        entity::adjust_property(&mut state, 3); // x 2 -> 5
        state.entity_field = EntityField::ZoneRegMaxY;
        entity::adjust_property(&mut state, 2); // y 2 -> 4
        let want_region = state.level.overlay.zones[0].region.clone();

        // Bind the material via the `c` cycle (steps `(none)` -> "ore").
        state.entity_field = EntityField::ZoneMaterial;
        entity::cycle_property(&mut state);
        assert_eq!(
            state.level.overlay.zones[0].material.0, mat_name,
            "material bound"
        );

        // Scale → SelfSimilar (None -> Crisp -> SelfSimilar = two cycles), then set
        // arms + twist + warp via the dedicated numeric sub-fields.
        state.entity_field = EntityField::ZoneScale;
        entity::cycle_property(&mut state); // None -> Crisp
        entity::cycle_property(&mut state); // Crisp -> SelfSimilar
        state.entity_field = EntityField::ZoneArms;
        entity::adjust_property(&mut state, 5); // arms 0 -> 5
        state.entity_field = EntityField::ZoneTwist;
        entity::adjust_property(&mut state, 3); // twist 0 -> 0.3
        state.entity_field = EntityField::ZoneWarp;
        entity::adjust_property(&mut state, 2); // warp 0 -> 0.10
        let want_scale = state.level.overlay.zones[0].scale.clone();
        assert!(matches!(
            want_scale,
            re_level::OverlayScaleRule::SelfSimilar(_)
        ));

        // Reference the inset: cursor on it (only one), `c` toggles membership ON.
        state.entity_field = EntityField::ZoneInsets;
        state.zone_inset_cursor = 0;
        entity::cycle_property(&mut state);
        assert_eq!(
            state.level.overlay.zones[0].insets.len(),
            1,
            "inset referenced"
        );

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[zone "),
            "save emits a [zone] section:\n{text}"
        );
        assert!(
            text.contains("scale = self_similar"),
            "scale emitted:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "overlay-zone-rt.lvl").expect("reparse");

        assert_eq!(reloaded.overlay.zones.len(), 1, "one zone round-trips");
        assert_eq!(
            reloaded.overlay.insets.len(),
            1,
            "the referenced inset round-trips"
        );
        let z = &reloaded.overlay.zones[0];
        assert_eq!(z.name, zone_name, "zone name round-trips");
        assert_eq!(z.region, want_region, "zone region round-trips");
        assert_eq!(z.material.0, mat_name, "zone material binding round-trips");
        // Scale params (arms / twist / warp) survive.
        match (&z.scale, &want_scale) {
            (
                re_level::OverlayScaleRule::SelfSimilar(got),
                re_level::OverlayScaleRule::SelfSimilar(want),
            ) => {
                assert_eq!(got.arms, want.arms, "arms round-trips");
                assert_eq!(got.arms, 5, "arms value survived");
                assert!((got.twist - want.twist).abs() < 1e-4, "twist round-trips");
                assert!(
                    (got.octave_warp - want.octave_warp).abs() < 1e-4,
                    "warp round-trips"
                );
            }
            other => panic!("scale must round-trip as SelfSimilar, got {other:?}"),
        }
        // The inset reference survives by name.
        assert_eq!(z.insets.len(), 1, "inset reference round-trips");
        assert_eq!(
            z.insets[0].0, inset_name,
            "referenced inset name round-trips"
        );

        // The reload did not reject (validate_zones accepted the bound material) — the
        // build also resolves cleanly.
        let _ = reloaded.build_map();
    }

    /// SCALABLE MATERIAL OVERLAY — LINE round-trip: place a 3-point line via the
    /// waypoint flow (`space` appends each), bind a line-material, set a Crisp scale,
    /// save, reload, and assert the path points / material / scale survive. Lines
    /// serialize via `[line]` (`path` / `material` / `scale`); `none` scale is skipped
    /// so `crisp` is used to guarantee the field actually serializes.
    #[test]
    fn overlay_line_save_reload_preserves_path_material_scale() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(16, 12, 4);
        let mi = state.level.push_material();
        let mat_name = "trace_mat".to_string();
        state.level.materials[mi].name = mat_name.clone();

        // Place a 3-point polyline via the waypoint append flow.
        state.place = Some(EntityKind::Line);
        state.z = 0;
        for &(x, y) in &[(1usize, 2usize), (4, 2), (4, 6)] {
            state.cursor = (x, y);
            entity::place_at_cursor(&mut state); // appends a waypoint
        }
        assert_eq!(state.level.overlay.lines.len(), 1, "one line created");
        assert_eq!(
            state.level.overlay.lines[0].path.points.len(),
            3,
            "3 waypoints"
        );
        let want_points = state.level.overlay.lines[0].path.points.clone();

        // Bind the line-material via `c` (steps `(none)` -> the material name).
        state.entity_field = EntityField::LineMaterial;
        entity::cycle_property(&mut state);
        assert_eq!(state.level.overlay.lines[0].material.0, mat_name);

        // Scale → Crisp (None -> Crisp = one cycle).
        state.entity_field = EntityField::LineScale;
        entity::cycle_property(&mut state);
        assert_eq!(
            state.level.overlay.lines[0].scale,
            re_level::OverlayScaleRule::Crisp
        );

        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[line "),
            "save emits a [line] section:\n{text}"
        );
        assert!(
            text.contains("scale = crisp"),
            "crisp scale emitted:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "overlay-line-rt.lvl").expect("reparse");

        assert_eq!(reloaded.overlay.lines.len(), 1, "one line round-trips");
        let l = &reloaded.overlay.lines[0];
        // Path points survive (epsilon compare each component).
        assert_eq!(l.path.points.len(), 3, "3 waypoints round-trip");
        for (got, want) in l.path.points.iter().zip(want_points.iter()) {
            assert!((got.x - want.x).abs() < 1e-4, "point x round-trips");
            assert!((got.y - want.y).abs() < 1e-4, "point y round-trips");
            assert!((got.z - want.z).abs() < 1e-4, "point z round-trips");
        }
        assert_eq!(l.material.0, mat_name, "line material round-trips");
        assert_eq!(
            l.scale,
            re_level::OverlayScaleRule::Crisp,
            "line scale round-trips"
        );
    }

    /// SCALABLE MATERIAL OVERLAY — INSET round-trip: place an inset via the object
    /// path, set an octave RANGE (lo != hi), grow the region, type a child-scene path,
    /// save, reload, and assert at_octave / region / child all survive. Insets
    /// serialize via `[inset]` (`at_octave` as `N-M` when lo != hi, `region`, `child`).
    #[test]
    fn overlay_inset_save_reload_preserves_octave_region_child() {
        use crate::entity;
        use crate::state::{EntityField, EntityKind};

        let mut state = EditorState::new_empty(14, 10, 4);

        // Place an INSET at (3,3,1) via the object path.
        state.place = Some(EntityKind::Inset);
        state.cursor = (3, 3);
        state.z = 1;
        entity::place_at_cursor(&mut state);
        assert_eq!(state.level.overlay.insets.len(), 1);
        let inset_name = state.level.overlay.insets[0].name.clone();

        // Octave range: lo 1, hi 4 → a real range (lo != hi) so the writer emits `N-M`.
        // (A negative LO would serialize to `-1-3`, which the parser's `-`-split treats
        // as ambiguous — a documented format constraint, not exercised here.)
        state.entity_field = EntityField::InsetOctaveHi;
        entity::adjust_property(&mut state, 4); // hi 0 -> 4 first (lo stays 0 ≤ hi)
        state.entity_field = EntityField::InsetOctaveLo;
        entity::adjust_property(&mut state, 1); // lo 0 -> 1
        let want_octave = state.level.overlay.insets[0].at_octave;
        assert_eq!(want_octave, (1, 4), "octave range set (lo != hi)");

        // Grow the region: max X +2, max Y +1.
        state.entity_field = EntityField::InsetRegMaxX;
        entity::adjust_property(&mut state, 2);
        state.entity_field = EntityField::InsetRegMaxY;
        entity::adjust_property(&mut state, 1);
        let want_region = state.level.overlay.insets[0].region.clone();

        // Type a child-scene path (via the caret-based text router).
        state.entity_field = EntityField::InsetChild;
        for c in "rooms/inner.lvl".chars() {
            assert!(entity::text_entry_key(
                &mut state,
                re_present::EditorKey {
                    code: re_present::EditorKeyCode::Char(c),
                    shift: false
                }
            ));
        }
        let re_level::OverlayInsetContent::ChildScene(want_child) =
            state.level.overlay.insets[0].content.clone();
        assert_eq!(want_child, "rooms/inner.lvl");

        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[inset "),
            "save emits an [inset] section:\n{text}"
        );
        assert!(
            text.contains("at_octave = 1-4"),
            "octave RANGE emitted:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "overlay-inset-rt.lvl").expect("reparse");

        assert_eq!(reloaded.overlay.insets.len(), 1, "one inset round-trips");
        let ins = &reloaded.overlay.insets[0];
        assert_eq!(ins.name, inset_name, "inset name round-trips");
        assert_eq!(ins.at_octave, want_octave, "octave range round-trips");
        assert_eq!(ins.region, want_region, "inset region round-trips");
        let re_level::OverlayInsetContent::ChildScene(child) = &ins.content;
        assert_eq!(child, &want_child, "inset child path round-trips");
    }

    /// [texture] round-trip: add a `[texture NAME] file = …` through the Level
    /// Settings form, save, reload through the real parser, and assert the named
    /// image resource survives. Proves the editor can author a level-global texture
    /// that `re_voximport::to_lvl_text` ↔ `re_level` carry losslessly.
    #[test]
    fn settings_add_texture_save_reload_preserves_it() {
        use crate::settings;
        use crate::settings::SettingsField;

        let mut state = EditorState::new_empty(8, 8, 4);
        state.map.set(0, 0, 0, CellKind::Floor);

        // Author a texture entry through the settings form (open → add → type → apply).
        settings::open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            st.focus = SettingsField::Textures;
            st.add_texture();
            for c in "wall".chars() {
                st.texture_push(c);
            }
            st.move_texture_col(1);
            for c in "assets/texture/wall.webp".chars() {
                st.texture_push(c);
            }
        }
        settings::apply(&mut state);
        assert_eq!(state.level.textures.len(), 1, "texture authored in memory");

        // Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[texture wall]"),
            "save emits a [texture] section:\n{text}"
        );
        let reloaded = re_level::load_string(&text, "texture-rt.lvl").expect("reparse");

        assert_eq!(reloaded.textures.len(), 1, "one texture round-trips");
        assert_eq!(
            reloaded.textures[0].name, "wall",
            "texture name round-trips"
        );
        assert_eq!(
            reloaded.textures[0].file, "assets/texture/wall.webp",
            "texture file path round-trips"
        );

        // The painted floor still round-trips (the texture didn't break cells).
        let rebuilt = reloaded.build_map();
        assert_eq!(rebuilt.cell_at(0, 0, 0), CellKind::Floor);
    }

    /// IMAGE MATERIAL round-trip: author a `[texture]` in Settings, then create an
    /// `recipe = image` material bound to it via the material editor with a
    /// non-default subcell / fit / span, save, reload, and assert the recipe variant,
    /// the texture binding, subcell, fit, AND the world-anchored span all survive.
    /// Proves the editor can author an image-textured surface end-to-end and that the
    /// `re_voximport` writer ↔ `re_level` parser round-trip every image key.
    #[test]
    fn create_image_material_bound_to_texture_save_reload_preserves_every_field() {
        use crate::material;
        use crate::material::MaterialField;
        use crate::settings;
        use crate::settings::SettingsField;
        use re_world::{Fit, MaterialRecipe, SubcellMode};

        let mut state = EditorState::new_empty(8, 8, 4);
        state.map.set(0, 0, 0, CellKind::Floor);

        // 1. Author the texture in Settings.
        settings::open(&mut state);
        {
            let st = state.settings.as_mut().unwrap();
            st.focus = SettingsField::Textures;
            st.add_texture();
            for c in "wall".chars() {
                st.texture_push(c);
            }
            st.move_texture_col(1);
            for c in "assets/texture/wall.webp".chars() {
                st.texture_push(c);
            }
        }
        settings::apply(&mut state);

        // 2. Create an image material bound to that texture.
        material::new_material(&mut state); // mat0, selected
        let want_name = state.level.materials[0].name.clone();

        // Recipe → Image (wrap backward; Image is the last RECIPES entry).
        state.material_field = MaterialField::Recipe;
        material::adjust_field(&mut state, -1);
        assert!(matches!(
            state.level.materials[0].recipe,
            Some(MaterialRecipe::Image { .. })
        ));

        // Texture binding → wall (cycle from none).
        state.material_field = MaterialField::Texture;
        material::cycle_enum(&mut state, 1);
        assert_eq!(state.level.materials[0].texture.as_deref(), Some("wall"));

        // Subcell → braille. `SUBCELL_MODES` walks the sub-cell resolution
        // LADDER: off → half → quad → sextant → braille.
        state.material_field = MaterialField::Subcell;
        for _ in 0..4 {
            material::cycle_enum(&mut state, 1);
        }
        let want_subcell = state.level.materials[0].image_subcell;
        assert_eq!(want_subcell, SubcellMode::Braille);

        // Fit → stretch (contain → stretch, one step backward).
        state.material_field = MaterialField::Fit;
        material::cycle_enum(&mut state, -1);
        let want_fit = state.level.materials[0].image_fit;
        assert_eq!(want_fit, Fit::Stretch);

        // Span → author a non-default world-anchored box.
        state.material_field = MaterialField::SpanAlongMin;
        material::adjust_field(&mut state, 2); // 0.0 + 2*0.5 = 1.0
        state.material_field = MaterialField::SpanAlongSize;
        material::adjust_field(&mut state, 4); // 1.0 + 4*0.5 = 3.0
        state.material_field = MaterialField::SpanUpTop;
        material::adjust_field(&mut state, 6); // 1.0 + 6*0.5 = 4.0
        state.material_field = MaterialField::SpanUpSize;
        material::adjust_field(&mut state, 2); // 1.0 + 2*0.5 = 2.0
        let want_span = state.level.materials[0].image_span.expect("span authored");

        // Paint it onto a cell so the material is referenced (id 1).
        state.cursor = (3, 4);
        state.z = 2;
        material::paint_id(&mut state);

        // 3. Save + reload through the real parser.
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("recipe = image"),
            "image recipe written:\n{text}"
        );
        assert!(text.contains("texture = wall"), "texture binding written");
        assert!(text.contains("subcell = braille"), "subcell written");
        assert!(text.contains("fit = stretch"), "fit written");
        assert!(text.contains("span = "), "span written");
        let reloaded = re_level::load_string(&text, "image-mat-rt.lvl").expect("reparse");

        let midx = reloaded
            .material_index(&want_name)
            .expect("material name round-trips");
        let m = &reloaded.materials[midx];
        // Recipe VARIANT is Image (the parser re-derives it from `recipe = image`).
        assert!(
            matches!(m.recipe, Some(MaterialRecipe::Image { .. })),
            "recipe = image round-trips to the Image variant"
        );
        // Texture binding (by name) survives.
        assert_eq!(
            m.texture.as_deref(),
            Some("wall"),
            "texture binding round-trips"
        );
        // Subcell + fit survive.
        assert_eq!(m.image_subcell, want_subcell, "subcell round-trips");
        assert_eq!(m.image_fit, want_fit, "fit round-trips");
        // Span survives (every component).
        let span = m.image_span.expect("span round-trips");
        assert!(
            (span.along_min - want_span.along_min).abs() < 1e-4,
            "span along_min"
        );
        assert!(
            (span.along_size - want_span.along_size).abs() < 1e-4,
            "span along_size"
        );
        assert!((span.up_top - want_span.up_top).abs() < 1e-4, "span up_top");
        assert!(
            (span.up_size - want_span.up_size).abs() < 1e-4,
            "span up_size"
        );

        // The texture the material binds to is also present in the reloaded level.
        assert_eq!(reloaded.textures.len(), 1, "the bound texture survives");
        assert_eq!(reloaded.textures[0].name, "wall");
    }

    /// A small procedural level with two props (a solid shelving block and a
    /// notice board that adopts — i.e. MOVES — its note).
    const PROP_LEVEL: &str = "[meta]
name = prop-rt
title = prop rt

[room]
size = 16 16 8
ambient = solid

[node R]
pos = 2,14,2
glyph = R
color = #FF4040
is_input = yes

[node output]
pos = 13,14,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1

[pipe R->output]
state = OFF
path = (2,14,2) (2,12,2) (13,12,2) (13,14,2)
switch_at = (7,12,2)

[inputs]
R = 1

[pedestal]
center = 11 14 2
volume = 1 1 1

[door]
pos = 14 1 1
target_parts = output

[player]
spawn = 8 9 1
facing = -y

[prop stacks]
kind = shelving
at = 3 5 1
facing = +y
size = 4 1 3

[prop board]
kind = notice_board
at = 9 1 1
facing = +y
size = 2 1 2
note = system_log

[note system_log pos=12,6,2 facing=+y]
    NOTICE.
";

    /// Open → save a prop level: the `[prop]` sections are written back verbatim,
    /// the adopted note keeps its AUTHORED position (not the board position the
    /// kit moved it to), and an unedited procedural level is not baked at all.
    /// Then force a bake (edit one cell): `[cells]` is written but carries NO prop
    /// cell, so the reload's base room is the authored room and the props expand
    /// exactly once.
    #[test]
    fn props_survive_save_and_are_never_baked_into_cells() {
        let fx = crate::testutil::scratch_path("props-rt.lvl");
        std::fs::write(&fx, PROP_LEVEL).expect("write fixture");
        let original = re_level::load(&fx).expect("fixture loads");
        let orig_map = original.build_map();
        let (exp, errs) = re_level::props::expand_level(&original, &original.build_map_base());
        assert!(errs.is_empty(), "{errs:?}");
        assert!(!exp.cells.is_empty());

        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &fx));
        let text = capture_and_serialize(&mut state);
        assert!(
            text.contains("[prop stacks]") && text.contains("kind = shelving"),
            "{text}"
        );
        assert!(
            text.contains("[prop board]") && text.contains("note = system_log"),
            "{text}"
        );
        assert!(
            text.contains("pos=12,6,2"),
            "the note keeps its authored position:\n{text}"
        );
        assert!(
            !text.contains("[cells]"),
            "an unedited procedural level is not baked:\n{text}"
        );
        let again = re_level::load_string(&text, "rt").expect("saved text reloads");
        assert_eq!(again.props.len(), 2);
        assert_eq!(
            re_level::CellGrid::from_map(&again.build_map()),
            re_level::CellGrid::from_map(&orig_map),
            "an unedited save reproduces the built map exactly"
        );

        // Force a bake: paint one far cell.
        let mut state = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut state, &fx));
        state.map.set(14, 9, 1, re_world::CellKind::WallSolid);
        let text = capture_and_serialize(&mut state);
        assert!(text.contains("[cells]"), "an edited level bakes its cells");
        assert!(text.contains("[prop stacks]") && text.contains("[prop board]"));
        let baked = re_level::load_string(&text, "rt-baked").expect("baked text reloads");
        let base = baked.build_map_base();
        for c in &exp.cells {
            if c.mode == re_level::props::CellMode::Build {
                assert_eq!(
                    base.cell_at(c.pos.0, c.pos.1, c.pos.2),
                    CellKind::Empty,
                    "prop cell {:?} was baked into [cells]",
                    c.pos
                );
            }
        }
        assert_eq!(
            base.cell_at(14, 9, 1),
            CellKind::WallSolid,
            "the real edit is kept"
        );
        // The re-expanded map equals the original plus the one edit.
        let mut want = re_level::CellGrid::from_map(&orig_map);
        let mut got = re_level::CellGrid::from_map(&baked.build_map());
        let i = |g: &re_level::CellGrid| 14 + g.size.0 * (9 + g.size.1);
        let (wi, gi) = (i(&want), i(&got));
        want.kinds[wi] = CellKind::WallSolid;
        got.kinds[gi] = CellKind::WallSolid;
        assert_eq!(
            got.kinds, want.kinds,
            "props expand exactly once after a baked save"
        );
        let _ = std::fs::remove_file(&fx);
    }
}
