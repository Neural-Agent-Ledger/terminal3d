//! In-editor OPEN / LOAD: scan `levels/*.lvl`, present a modal picker overlay,
//! and load a selected level IN-PLACE (no restart). The same [`load_path`]
//! routine backs both the CLI `--dump`/arg startup load and the overlay's Enter,
//! so there is one cleanly-rebuilding load path.
//!
//! Pure functions over [`EditorState`] except [`load_path`], which reads the
//! `.lvl` file from disk; no terminal. A bad file is a defined no-op with a
//! status hint — the editor keeps the current document.
use crate::state::{
    CommentAnchor, EditorState, OpenState, PreservedText, TrailKey, TrailingComment,
};

/// Scan `levels/` for `*.lvl` files, returning sorted `levels/<name>.lvl` paths
/// (mirrors [`crate::vox::scan_vox_files`]). A missing/unreadable directory
/// yields an empty list. Relative paths keep the scan engine-pure.
pub fn scan_lvl_files() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir("levels") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("lvl") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    out.push(format!("levels/{name}"));
                }
            }
        }
    }
    out.sort();
    out
}

/// Scan raw `.lvl` text for `[pipe FROM->TO]` blocks authored with a `spline = …`
/// key (rather than `path = …`) and return their `(FROM, TO)` node-name pairs.
/// This re-derives the editor's spline-mode set on load, which `re_level` cannot
/// preserve (it voxelizes `spline` into `path`). A pipe with `path` (or neither)
/// is omitted, so a plain level yields an empty set.
fn scan_spline_pipes(raw: &str) -> std::collections::HashSet<(String, String)> {
    let mut out = std::collections::HashSet::new();
    // The `(from, to)` of the pipe block currently being scanned, if any.
    let mut current: Option<(String, String)> = None;
    for line in raw.lines() {
        let trimmed = line.trim_start();
        if let Some(inner) = trimmed
            .strip_prefix("[pipe ")
            .and_then(|s| s.strip_suffix(']'))
        {
            // Entering a new pipe block: remember its endpoints. Accept BOTH the
            // ASCII `->` (what the editor emits) and the unicode `→` that
            // `re_level` also parses, so a hand-authored arrow header is
            // recovered too.
            current = inner
                .split_once("->")
                .or_else(|| inner.split_once('→'))
                .map(|(f, t)| (f.trim().to_string(), t.trim().to_string()));
        } else if trimmed.starts_with('[') {
            // Any other section header closes the pipe block.
            current = None;
        } else if let Some((from, to)) = current.clone() {
            // A key line inside a pipe block: a `spline =` key flags this pipe.
            if let Some(rest) = trimmed.strip_prefix("spline") {
                if rest.trim_start().starts_with('=') {
                    out.insert((from, to));
                }
            }
        }
    }
    out
}

/// Which kind of body the capture scanner is currently inside. Mirrors the
/// `re_level` parser's section semantics so the capture NEVER misreads content.
#[derive(PartialEq)]
enum CaptureBody {
    /// Before the first section header, or inside an ordinary `key = value`
    /// section: full-line comments + trailing comments are captured.
    Normal,
    /// Inside `[cells]` / `[part]` / `[note]`: the body is GEOMETRY glyph rows
    /// (`'#'`→`WallSolid`; part-mask `#` rows) or verbatim note text — NOTHING is
    /// captured here. A `#`-leading line here is data, not comment.
    Verbatim,
    /// Inside `[map]` / `[win_when]` (`re_level` `Section::Skip`): every line is
    /// appended verbatim to the active raw block.
    Raw,
}

/// Classify a section by the first token of its header interior.
fn body_kind(header_interior: &str) -> CaptureBody {
    match header_interior.split_whitespace().next().unwrap_or("") {
        "cells" | "part" | "note" => CaptureBody::Verbatim,
        "map" | "win_when" => CaptureBody::Raw,
        _ => CaptureBody::Normal,
    }
}

/// Split a NON-full-line-comment line into `(code, Some(trailing_suffix))`,
/// mirroring `re_level`'s `strip_comment` exactly: a ` #` (or tab-`#`) starts a
/// comment UNLESS the 6 chars after `#` are hex digits (`color = #FF0000` is a
/// value, not a comment). The suffix includes the whole whitespace run before
/// `#`, so `code + suffix == line` byte-for-byte.
fn split_trailing_comment(line: &str) -> (&str, Option<&str>) {
    let b = line.as_bytes();
    let mut i = 0;
    while i + 1 < b.len() {
        if (b[i] == b' ' || b[i] == b'\t') && b[i + 1] == b'#' {
            let after = &line[i + 2..];
            let hex_run = after
                .chars()
                .take(6)
                .take_while(char::is_ascii_hexdigit)
                .count();
            if hex_run < 6 {
                // The lone-`#` VALUE exception, shared with the parser so the two
                // can never drift: `glyph = #` is data, not an empty comment.
                // Duplicating this rule instead of calling `re_level` is what made
                // the editor capture a phantom ` #` suffix and re-emit
                // `glyph = # #` on five shipped levels.
                if re_level::hash_at_is_value(line, i + 1) {
                    return (line, None);
                }
                // Back over the whole whitespace run so the suffix carries it.
                let mut j = i;
                while j > 0 && (b[j - 1] == b' ' || b[j - 1] == b'\t') {
                    j -= 1;
                }
                return (&line[..j], Some(&line[j..]));
            }
            // `#RRGGBB` color value — keep scanning (parser parity).
        }
        i += 1;
    }
    (line, None)
}

/// True if `line`, sitting inside a [`CaptureBody::Verbatim`] body, is a
/// LEVEL-FILE COMMENT rather than body content.
///
/// Two conditions, both required, and both mirror `re_level`:
///
/// * **Column 0.** `re_level`'s `[note]` discriminator is column-0 `#` only, and
///   `re_voximport`'s writer ESCAPES an authored body line that starts with `#` by
///   prefixing one space. So a leading-space ` # …` is note TEXT, never a comment,
///   and testing the trimmed line here would silently steal it out of the body.
/// * **Not a geometry row.** `[cells]` and `[part]` mask rows are runs of glyph
///   characters and `#` is one of them (`WallSolid`), so `####` is GEOMETRY. A row
///   made entirely of `re_level::glyph_to_cell_kind` characters is therefore never
///   treated as a comment — the rule this whole capture is built around.
fn verbatim_line_is_comment(line: &str) -> bool {
    if !line.starts_with('#') {
        return false;
    }
    !line
        .chars()
        .all(|c| re_level::glyph_to_cell_kind(c).is_some())
}

/// Move the TRAILING run of level-file comments out of a just-ended
/// [`CaptureBody::Verbatim`] body into `pending`, so it re-anchors to the section
/// that follows (or, at EOF, to the file tail).
///
/// ## Why this exists (a silent-deletion bug in its own right)
///
/// [`reattach_preserved`](crate::io) re-emits a header-anchored comment block on the
/// line immediately above its header. When the PRECEDING emitted section has a
/// verbatim body — and after an editor save it usually does, because every save
/// bakes a `[cells]` block and `[note]`s are written last — that position is
/// *inside* the previous body. The old capture skipped verbatim bodies wholesale,
/// so the very next save deleted the comment: save 1 looked fine and save 2 quietly
/// lost hundreds of bytes of authored commentary across 12 shipped levels.
///
/// Only the trailing run is recovered, and only its comment lines plus the blanks
/// INTERLEAVED with them — a leading blank belongs to the writer's own formatting,
/// exactly as in the `Normal` path. Body content is never touched: the run stops at
/// the first line that is not a comment or a blank.
fn recover_verbatim_tail(buf: &mut Vec<String>, pending: &mut Vec<String>) {
    let mut start = buf.len();
    let mut saw_comment = false;
    for i in (0..buf.len()).rev() {
        let line = &buf[i];
        if line.trim().is_empty() {
            start = i;
        } else if verbatim_line_is_comment(line) {
            start = i;
            saw_comment = true;
        } else {
            break;
        }
    }
    if saw_comment {
        let mut run = buf.split_off(start);
        // Drop the leading blanks — the writer's separator, not the block's.
        while run.first().is_some_and(|l| l.trim().is_empty()) {
            run.remove(0);
        }
        pending.append(&mut run);
    }
    buf.clear();
}

/// The anchor key for a body line: the `key` of `key = value`, or the full
/// trimmed code text for key-less lines (`[hierarchy]` links, shape rows).
fn line_key(code_trimmed: &str) -> String {
    code_trimmed
        .split_once('=')
        .map_or_else(|| code_trimmed.to_string(), |(k, _)| k.trim().to_string())
}

/// Capture everything the typed `Level` model drops from `raw` — full-line `#`
/// comment blocks (with anchors), trailing inline comments, and the verbatim
/// `[map]`/`[win_when]` blocks — into a [`PreservedText`].
///
/// Section-aware: `[cells]`/`[part]`/`[note]` bodies are skipped
/// entirely (their `#` is geometry / verbatim text). Blank lines interleaved
/// with a pending comment block are kept verbatim so a canonical file
/// round-trips byte-identically; blank lines with no pending comments belong to
/// the writer's own formatting and are ignored. Comment blocks are anchored to
/// the FOLLOWING header (leading block) or the FOLLOWING body line (interior
/// block); comments at EOF go to `tail`. A comment block immediately above a
/// `[map]`/`[win_when]` header travels inside that raw block instead (the
/// writer never emits those headers, so a header anchor could never re-attach).
///
/// NOTE(rename): anchors are keyed by the section-header string — see the
/// [`CommentAnchor`] docs; [`crate::rename::rename_object`] must rewrite those keys.
pub fn capture_preserved(raw: &str) -> PreservedText {
    // Close the active raw block: trim trailing blank lines, store.
    fn flush_raw(raw_buf: &mut Vec<String>, p: &mut PreservedText) {
        while raw_buf.last().is_some_and(|l| l.trim().is_empty()) {
            raw_buf.pop();
        }
        if !raw_buf.is_empty() {
            p.raw_blocks.push(raw_buf.join("\n"));
            raw_buf.clear();
        }
    }

    let mut p = PreservedText::default();
    let mut body = CaptureBody::Normal;
    // The TRIMMED header line of the section being scanned ("" before the first
    // header — an anchor keyed "" can never match an emitted header, so a
    // stray pre-header key line safely routes its pending block to leftovers).
    let mut cur_header = String::new();
    let mut pending: Vec<String> = Vec::new();
    let mut raw_buf: Vec<String> = Vec::new();
    // The lines of the verbatim body currently being scanned. Only its TRAILING
    // comment run is ever kept — see `recover_verbatim_tail`.
    let mut verbatim_buf: Vec<String> = Vec::new();
    // Are we inside a `[note]` body? Note bodies alone take the parser's column-0
    // header rule (see `is_header` below).
    let mut in_note_body = false;

    for line in raw.lines() {
        // CRLF tolerance, exactly like the parser.
        let line = line.strip_suffix('\r').unwrap_or(line);
        let trimmed = line.trim();

        // A section header ends EVERY body kind (parser parity: headers are
        // recognized even inside [cells]/[part]/[note]/[map] bodies) — with ONE
        // exception, also parser parity: inside a `[note]` body a bracketed line
        // is a header only in COLUMN 0 (`re_level::note_body_line_is_header`). An
        // INDENTED bracketed line there is placard TEXT — `L7-hidden-depths.lvl`'s
        // `    [ STEWARDSHIP - DO NOT POWER DOWN ]`. Without mirroring the rule the
        // capture would end the note early and start comment-scanning real placard
        // prose, which is exactly the hazard this function exists to avoid.
        let is_header = trimmed.len() >= 2 && (!in_note_body || line.starts_with('[')) && {
            let (code, _) = split_trailing_comment(line);
            let ct = code.trim();
            ct.starts_with('[') && ct.ends_with(']')
        };

        if body == CaptureBody::Raw && !is_header {
            raw_buf.push(line.to_string());
            continue;
        }
        if body == CaptureBody::Verbatim && !is_header {
            // Never comment-scan geometry / note bodies IN PLACE. The lines
            // are buffered only so the trailing comment run can be recovered when
            // the body closes; nothing else here is ever captured.
            verbatim_buf.push(line.to_string());
            continue;
        }

        if is_header {
            // The previous verbatim body just ended — rescue any comments that sat
            // between its last row and this header.
            recover_verbatim_tail(&mut verbatim_buf, &mut pending);
            let (code, suffix) = split_trailing_comment(line);
            let header = code.trim().to_string();
            let interior = header[1..header.len() - 1].trim();
            let new_body = body_kind(interior);
            flush_raw(&mut raw_buf, &mut p);
            if new_body == CaptureBody::Raw {
                // The leading comment block travels INSIDE the raw block, and
                // the header line itself is kept verbatim (suffix and all).
                raw_buf.append(&mut pending);
                raw_buf.push(line.to_string());
            } else {
                if !pending.is_empty() {
                    p.anchors.push(CommentAnchor {
                        header: header.clone(),
                        key: None,
                        lines: std::mem::take(&mut pending),
                    });
                }
                if let Some(sfx) = suffix {
                    p.trailing.push(TrailingComment {
                        header: header.clone(),
                        key: TrailKey::Header,
                        suffix: sfx.to_string(),
                    });
                }
            }
            in_note_body = interior.split_whitespace().next() == Some("note");
            cur_header = header;
            body = new_body;
            continue;
        }

        // Normal body (or pre-header) line.
        if trimmed.is_empty() {
            // Blanks are captured only as part of a pending comment block.
            if !pending.is_empty() {
                pending.push(line.to_string());
            }
            continue;
        }
        if trimmed.starts_with('#') {
            pending.push(line.to_string());
            continue;
        }
        // A code line: flush any pending block as an interior anchor and
        // record a trailing comment if present.
        let (code, suffix) = split_trailing_comment(line);
        let key = line_key(code.trim());
        if !pending.is_empty() {
            p.anchors.push(CommentAnchor {
                header: cur_header.clone(),
                key: Some(key.clone()),
                lines: std::mem::take(&mut pending),
            });
        }
        if let Some(sfx) = suffix {
            p.trailing.push(TrailingComment {
                header: cur_header.clone(),
                key: TrailKey::Key(key),
                suffix: sfx.to_string(),
            });
        }
    }
    flush_raw(&mut raw_buf, &mut p);
    // A file that ENDS inside a verbatim body (the writer emits `[note]`s last, so
    // this is the common shape) still owes its trailing comment run to the tail.
    recover_verbatim_tail(&mut verbatim_buf, &mut pending);
    // Comments (and their interleaved blanks) left at EOF.
    while pending.last().is_some_and(|l| l.trim().is_empty()) {
        pending.pop();
    }
    p.tail = pending;
    p
}

/// Load the `.lvl` at `path` into `state`, REBUILDING the document cleanly (no
/// restart): read + parse, then swap in the new map/camera/level, reset the
/// cursor/z into bounds, and clear the dirty flag, history, and all transient
/// selections / modes so the editor lands in a clean "just opened" state.
///
/// Returns `true` on success. On a read/parse failure it leaves the current
/// document untouched, sets a status hint, and returns `false`.
pub fn load_path(state: &mut EditorState, path: &str) -> bool {
    let Ok(raw) = std::fs::read_to_string(path) else {
        state.status = format!("load failed: {path}");
        return false;
    };
    let level = match re_level::load_string(&raw, path) {
        Ok(l) => l,
        Err(_) => {
            state.status = format!("load failed: {path}");
            return false;
        }
    };
    // Recover which pipes were authored as a SPLINE: `re_level` voxelizes a
    // `[pipe] spline = …` curve into `path` and discards the fact, so scan the
    // RAW text for pipe blocks whose key was `spline` and re-flag them. (Done
    // before the level swap so the scan reads exactly the loaded file.)
    state.spline_pipes = scan_spline_pipes(&raw);
    // Preserve what the typed model drops — `#` comments (with
    // anchors) and the verbatim `[map]`/`[win_when]` blocks — for re-emission by
    // the io.rs save post-pass. Captured from the RAW text, like the spline scan.
    state.preserved = capture_preserved(&raw);
    // Remember whether THIS file already carried an explicit
    // `[cells]` section, so the rebuild/bake work can refuse to
    // silently flatten a procedural level on save.
    state.had_cells_at_load = level.cells.is_some();
    // Keep the AS-LOADED `[cells]` provenance for rebuild_map — the save
    // path overwrites `level.cells` with a live capture, which a rebuild must
    // not treat as authored geometry.
    state.loaded_cells = level.cells.clone();
    state.map = level.build_map();
    // At load the live map IS the procedural build — record it as the
    // baseline for rebuild_map's diff merge (hand edits = later divergences).
    state.preview_baseline = Some(state.map.clone());
    state.cam = level.spawn_camera();
    state.level = level;
    state.path = Some(path.to_string());
    // Reseed the off-list cycle vocabulary from what THIS file authors, so every
    // authored glyph / preset / direction / family is offered by `c` and by the
    // enum popup from the first keypress rather than only after someone has stepped
    // off it — and so the previous document's vocabulary does not leak into this
    // one. Lap identity does not depend on the seed (every cycle also remembers the
    // value it steps off); this is the discoverability half. See `crate::vocab`.
    state.vocab = crate::vocab::Vocab::default();
    state.vocab.harvest(&state.level);

    // Land in a clean, predictable editing state.
    state.cursor = (0, 0);
    state.z = 0;
    state.clamp_cursor();
    state.dirty = false;
    state.quit_armed = false;
    state.history = crate::history::History::default();
    // Drop every transient selection / placement mode so stale indices from the
    // previous level can't point into the new one.
    state.place = None;
    state.selected_node = None;
    state.selected_emitter = None;
    state.selected_trigger = None;
    state.selected_pipe = None;
    state.pipe_pending = None;
    state.selected_vox = None;
    state.shape_mode = false;
    state.stair_mode = false;
    state.material_mode = false;
    state.vox_mode = false;
    state.fill_anchor = None;
    // Also reset the transient INTERACTION state a fresh "just opened"
    // document expects, so stale UI focus from the previous level can't leak in.
    // Release any preview capture — the pure layer can't call the OS-grab
    // deactivate directly, so it requests it (a Bevy system drains the request).
    state.preview_captured = false;
    state.capture_request = Some(false);
    state.entity_field = crate::state::EntityField::DoorPosX;
    state.material_field = crate::material::MaterialField::Fg;
    state.noclip = true; // the editor default (free-fly)
    state.open_armed = false; // defensive: a fresh load is never mid-confirm
                              // A fresh document has no pending debounced
                              // rebuild, no open rename prompt, and no armed playtest confirm.
    state.preview_rebuild_delay = None;
    state.rename = None;
    state.playtest_armed = false;
    state.status = format!("opened {path}");
    // CRASH RECOVERY: after a successful load, offer an autosave sidecar
    // that is strictly NEWER than the file just opened (a stale/equal/missing
    // sidecar is silently ignored — clearing any offer left from a previous
    // document). Answered by the pre-chain `y`/`n` prompt in `edit::apply_key`.
    // Loading the sidecar ITSELF (via `restore_autosave`) can never re-offer:
    // its sidecar path resolves to the same file, whose mtime is never strictly
    // newer than itself.
    state.autosave_offer = None;
    let sidecar = crate::io::autosave_path_for(Some(path));
    if autosave_is_newer(file_mtime(&sidecar), file_mtime(path)) {
        state.autosave_offer = Some(sidecar);
        state.status = "AUTOSAVE found (newer) — Y restore / N discard".into();
    }
    true
}

/// The filesystem mtime of `path`, or `None` when the file is missing or the
/// metadata is unreadable. The I/O half feeding [`autosave_is_newer`].
fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// The PURE crash-recovery decision: should a found autosave sidecar be
/// OFFERED over the just-opened file? `true` only when BOTH mtimes are known and
/// the sidecar's is STRICTLY newer — an older, equal-aged, or missing sidecar
/// (or an unreadable original mtime) is never offered.
pub fn autosave_is_newer(
    sidecar_mtime: Option<std::time::SystemTime>,
    original_mtime: Option<std::time::SystemTime>,
) -> bool {
    matches!((sidecar_mtime, original_mtime), (Some(s), Some(o)) if s > o)
}

/// ACCEPT the pending crash-recovery offer: load the autosave sidecar's
/// content INTO the editor via the one true [`load_path`] routine, then re-point
/// `state.path` back at the ORIGINAL document path and mark the document DIRTY —
/// the restored content lives only in memory until the user explicitly saves it
/// over the original. The offer is cleared either way; on a failed sidecar load
/// the current document is left untouched (`load_path`'s no-op contract) with
/// its "load failed" status hint.
pub fn restore_autosave(state: &mut EditorState) {
    let Some(sidecar) = state.autosave_offer.take() else {
        return;
    };
    let original = state.path.clone();
    if load_path(state, &sidecar) {
        state.path = original;
        state.dirty = true;
        state.autosave_offer = None;
        state.status = "AUTOSAVE restored (unsaved) — s to keep it".into();
    } else {
        state.autosave_offer = None;
    }
}

/// The status line a REJECTED crash-recovery offer leaves behind. Named so the
/// keyboard `n` arm (`edit::apply_key`) and the overlay's `[ Discard ]` button
/// (`mouse::apply_modal_click`) speak with one voice; `discard_status_is_the_same_
/// through_the_key_and_the_button` pins the two together.
pub const AUTOSAVE_DISCARDED_STATUS: &str = "autosave discarded (sidecar kept on disk)";

/// REJECT the pending crash-recovery offer: drop it and leave the document
/// exactly as it was opened. The sidecar FILE is deliberately left on disk — a
/// discard is a "not now", not a delete, so the recovered work stays recoverable
/// by hand. This is the `n` answer; [`dismiss_autosave_offer`] is the Esc answer
/// (same effect, different status wording).
pub fn discard_autosave_offer(state: &mut EditorState) {
    state.autosave_offer = None;
    state.status = AUTOSAVE_DISCARDED_STATUS.into();
}

/// Open the file-picker overlay: scan `levels/` and show the modal list. The
/// cursor starts on the currently-loaded file if it is in the list, else 0.
pub fn open_overlay(state: &mut EditorState) {
    let files = scan_lvl_files();
    let cursor = state
        .path
        .as_deref()
        .and_then(|p| files.iter().position(|f| f == p))
        .unwrap_or(0);
    if files.is_empty() {
        state.status = "no .lvl files in levels/".into();
    } else {
        state.status = "open: Up/Dn  Enter load  Esc cancel".into();
    }
    state.open = Some(OpenState { files, cursor });
}

/// Close the overlay without loading.
pub fn close_overlay(state: &mut EditorState) {
    state.open = None;
    state.open_armed = false; // cancelling disarms the dirty-open guard
    state.status = "open cancelled".into();
}

/// Move the overlay cursor by `delta` (clamped to the list).
pub fn navigate(state: &mut EditorState, delta: i32) {
    if let Some(o) = state.open.as_mut() {
        if o.files.is_empty() {
            return;
        }
        let n = o.files.len() as i32;
        o.cursor = (o.cursor as i32 + delta).clamp(0, n - 1) as usize;
    }
}

/// Load the highlighted file and close the overlay.
///
/// Dirty-open guard: if the document has unsaved edits, the FIRST pick
/// does NOT discard — it arms the guard (`open_armed`), keeps the overlay open on
/// the same highlighted row, and asks for a confirm, mirroring the `quit_armed`
/// two-step. A SECOND pick (while armed) — or a pick on a clean document —
/// proceeds with the load (which resets `open_armed`). A no-op (closes) when the
/// list is empty or the highlighted index is out of range.
pub fn pick(state: &mut EditorState) {
    // Read the selection WITHOUT taking the overlay yet, so re-arming keeps the
    // same highlighted file visible.
    let Some(o) = state.open.as_ref() else { return };
    let Some(path) = o.files.get(o.cursor).cloned() else {
        state.status = "nothing to open".into();
        return;
    };
    // Dirty + not yet armed → require a confirm; keep the overlay open.
    if state.dirty && !state.open_armed {
        state.open_armed = true;
        state.status = "UNSAVED — Enter again to discard & open".into();
        return;
    }
    // Clean, or armed → load. Now take the overlay and load (load_path clears
    // `open_armed`).
    state.open = None;
    load_path(state, &path);
}

/// Open `path` through the SAME dirty guard the overlay's Enter uses.
///
/// The `:` command palette's `open level <file>` row used to call [`load_path`]
/// directly, which replaces map/level, clears `dirty`, and WIPES history — one
/// fuzzy hit on a dirty document destroyed the session with no undo. This stages
/// the pick in the overlay (the `levels/` scan, `path` highlighted) and delegates
/// to [`pick`]: a clean document loads immediately (the overlay closes again), a
/// dirty one ARMS `open_armed` and leaves the overlay up on that row, so the
/// user's second Enter is the explicit confirm.
pub fn pick_path(state: &mut EditorState, path: &str) {
    let mut files = scan_lvl_files();
    let cursor = files.iter().position(|f| f == path).unwrap_or_else(|| {
        // Not in the scan (a level renamed/removed since the palette opened, or
        // a path outside `levels/`): stage it alone so the guard still applies
        // to the real target rather than to some unrelated highlighted row.
        files = vec![path.to_string()];
        0
    });
    state.open = Some(OpenState { files, cursor });
    pick(state);
}

/// DISMISS a pending crash-recovery offer, returning whether there was
/// one. The offer is MODAL — only `y`/`n` (in `edit::apply_key`) and Esc (via
/// `main::quit_on_esc`, which calls this) answer it. Esc means "leave the
/// document as opened"; the sidecar file itself is kept on disk.
pub fn dismiss_autosave_offer(state: &mut EditorState) -> bool {
    if state.autosave_offer.take().is_none() {
        return false;
    }
    state.status = "autosave offer dismissed (sidecar kept on disk)".into();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::ScratchFile;
    use re_world::CellKind;

    #[test]
    fn scan_returns_sorted_relative_lvl_paths() {
        let files = scan_lvl_files();
        assert!(files.iter().all(|f| f.ends_with(".lvl")), "only .lvl paths");
        assert!(
            files.iter().all(|f| f.starts_with("levels/")),
            "relative levels/ paths"
        );
        let mut sorted = files.clone();
        sorted.sort();
        assert_eq!(files, sorted, "deterministic / sorted");
    }

    /// Loading a different level over a dirty, edited state swaps the document and
    /// clears dirty + history. We use a real shipped level via the workspace path
    /// (the test cwd is the crate dir, like io.rs's round-trip test).
    #[test]
    fn load_path_swaps_document_and_clears_dirty_and_history() {
        let mut s = EditorState::new_empty(4, 4, 2);
        // Make the state look "used": paint, dirty, history, a selection.
        s.brush = CellKind::WallSolid;
        s.cursor = (1, 1);
        crate::edit::apply_tool_at_cursor(&mut s); // paints + snapshots undo
        s.place = Some(crate::state::EntityKind::Node);
        assert!(s.dirty);
        assert!(s.history.undo_len() > 0);
        let old_size = s.map.size();

        let ok = load_path(&mut s, "../../levels/01.lvl");
        assert!(ok, "shipped level 01 loads");
        assert_eq!(s.path.as_deref(), Some("../../levels/01.lvl"));
        assert_ne!(s.map.size(), old_size, "map swapped to the loaded level");
        assert!(!s.dirty, "load clears dirty");
        assert_eq!(s.history.undo_len(), 0, "load clears undo history");
        assert!(s.place.is_none(), "load clears placement mode");
        assert_eq!(s.cursor, (0, 0), "cursor reset");
    }

    #[test]
    fn load_path_missing_file_keeps_document() {
        let mut s = EditorState::new_empty(5, 5, 2);
        let before = s.map.size();
        let ok = load_path(&mut s, "levels/does-not-exist-xyz.lvl");
        assert!(!ok);
        assert_eq!(s.map.size(), before, "document untouched on a bad load");
        assert!(s.status.starts_with("load failed"));
    }

    /// Opening a different level over a DIRTY document requires a confirm.
    /// The first `pick` arms the guard (no load); the second loads.
    #[test]
    fn open_with_dirty_requires_confirm() {
        let mut s = EditorState::new_empty(4, 4, 2);
        // Make the document dirty via a real edit.
        s.brush = CellKind::WallSolid;
        s.cursor = (1, 1);
        crate::edit::apply_tool_at_cursor(&mut s);
        assert!(s.dirty, "edit made it dirty");
        let old_path = s.path.clone();

        // Open the overlay onto a real shipped level (workspace-relative, like the
        // load round-trip test).
        open_overlay(&mut s);
        s.open = Some(OpenState {
            files: vec!["../../levels/01.lvl".into()],
            cursor: 0,
        });

        // First pick: ARM, do not load.
        pick(&mut s);
        assert!(s.open.is_some(), "overlay stays open while armed");
        assert!(s.open_armed, "first pick arms the dirty-open guard");
        assert_eq!(s.path, old_path, "no load on the first (arming) pick");
        assert!(s.dirty, "still dirty — nothing was loaded");

        // Second pick: load.
        pick(&mut s);
        assert!(s.open.is_none(), "overlay closed after the load");
        assert_eq!(s.path.as_deref(), Some("../../levels/01.lvl"), "loaded");
        assert!(!s.dirty, "load clears dirty");
        assert!(!s.open_armed, "load disarms the guard");
    }

    /// Loading a level resets transient interaction state — preview
    /// capture, the focused entity field, the editor noclip default.
    #[test]
    fn load_path_clears_preview_captured() {
        use crate::state::EntityField;
        let mut s = EditorState::new_empty(5, 5, 2);
        s.preview_captured = true;
        s.noclip = false;
        s.entity_field = EntityField::NodeThreshold; // a non-default field

        let ok = load_path(&mut s, "../../levels/01.lvl");
        assert!(ok, "shipped level 01 loads");
        assert!(!s.preview_captured, "load releases preview capture");
        assert_eq!(
            s.capture_request,
            Some(false),
            "load requests OS-grab deactivate"
        );
        assert!(s.noclip, "load restores the editor noclip default");
        assert_eq!(
            s.entity_field,
            EntityField::DoorPosX,
            "load resets the focused entity field"
        );
    }

    /// `split_trailing_comment` mirrors `re_level::strip_comment` —
    /// ` #` starts a comment unless it is a `#RRGGBB` color value; the suffix
    /// carries the whitespace run so `code + suffix == line`.
    #[test]
    fn split_trailing_comment_mirrors_parser_rules() {
        use super::split_trailing_comment as stc;
        assert_eq!(stc("speed = 2   # fast"), ("speed = 2", Some("   # fast")));
        assert_eq!(stc("a = b\t# c"), ("a = b", Some("\t# c")));
        // A #RRGGBB color VALUE is not a comment (parser parity)…
        assert_eq!(stc("color = #FF0000"), ("color = #FF0000", None));
        // …but a later real comment after a color still is.
        assert_eq!(
            stc("color = #FF0000 # warm"),
            ("color = #FF0000", Some(" # warm"))
        );
        assert_eq!(stc("pos = 1,2,3"), ("pos = 1,2,3", None));
    }

    /// The capture anchors comment blocks and trailing
    /// comments, hoards `[map]`/`[win_when]` verbatim, and NEVER touches the
    /// bodies of `[cells]`/`[part]`/`[note]` — `#` there is geometry/text.
    #[test]
    fn capture_is_section_aware_and_anchors_blocks() {
        let raw = "\
# file header A
# file header B

[meta]
name = t # inline
title = T

# leading block for room
[room]
size = 4,4,2

# above map
[map]
# inside map
XY

[win_when]
all_on = yes

[cells]
size = 4,4,2
####
#..#
#..#
####

[part p]
w = 2
h = 1
d = 2
##
##

[note wall 1,1,1]
 # this is note BODY, not a comment

[door]
pos = 1,1,0
# tail comment
";
        let p = capture_preserved(raw);

        // Leading blocks: the file header (anchored to the first section) and
        // the [room] block. NOTHING for [cells]/[part]/[note] bodies.
        //
        // NOTE on the `[note]` line above: it carries ONE leading space, which is
        // exactly the form `re_voximport` writes. `re_level`'s note-body comment
        // discriminator is COLUMN-0 `#` only (`note_body_line_is_comment`), and the
        // writer escapes an authored body line that would hit it by prefixing that
        // space. So ` # …` is body text and must not be captured — which is what
        // this asserts. An UNescaped column-0 `#` in a note body is the other case:
        // `re_level` discards it outright, so it is a comment by definition and
        // `recover_verbatim_tail` now rescues it (see
        // `trailing_comments_after_a_verbatim_body_survive_the_next_save`).
        assert_eq!(p.anchors.len(), 2, "anchors: {:?}", p.anchors);
        assert_eq!(p.anchors[0].header, "[meta]");
        assert_eq!(p.anchors[0].key, None);
        assert_eq!(
            p.anchors[0].lines,
            vec![
                "# file header A".to_string(),
                "# file header B".into(),
                String::new()
            ],
            "file header block keeps its trailing blank line"
        );
        assert_eq!(p.anchors[1].header, "[room]");
        assert_eq!(
            p.anchors[1].lines,
            vec!["# leading block for room".to_string()]
        );

        // Trailing comment keyed to ([meta], name).
        assert_eq!(p.trailing.len(), 1, "trailing: {:?}", p.trailing);
        assert_eq!(p.trailing[0].header, "[meta]");
        assert_eq!(p.trailing[0].key, TrailKey::Key("name".into()));
        assert_eq!(p.trailing[0].suffix, " # inline");

        // Raw blocks: [map] (with the comment above AND inside it) + [win_when].
        assert_eq!(p.raw_blocks.len(), 2, "raw blocks: {:?}", p.raw_blocks);
        assert_eq!(p.raw_blocks[0], "# above map\n[map]\n# inside map\nXY");
        assert_eq!(p.raw_blocks[1], "[win_when]\nall_on = yes");

        // Tail comment after the last section's body.
        assert_eq!(p.tail, vec!["# tail comment".to_string()]);

        // THE POINT: no geometry row was mistaken for a comment.
        let all_captured: Vec<&String> = p
            .anchors
            .iter()
            .flat_map(|a| a.lines.iter())
            .chain(p.tail.iter())
            .collect();
        assert!(
            all_captured
                .iter()
                .all(|l| !l.contains("####") && l.trim() != "##"),
            "no [cells]/[part] glyph row captured as a comment: {all_captured:?}"
        );
    }

    /// A comment sitting between the END of a verbatim body and the next header
    /// (or EOF) must be CAPTURED, not eaten.
    ///
    /// This is where `reattach_preserved` puts a header-anchored block, and after
    /// an editor save the preceding section is usually verbatim — every save bakes
    /// a `[cells]` block, and `[note]`s are written last. The old capture skipped
    /// verbatim bodies wholesale, so save 1 re-emitted the comment into that spot
    /// and save 2 silently deleted it (12 shipped levels, hundreds of bytes).
    ///
    /// Geometry is still untouchable: the `####` / `##` rows and the escaped
    /// ` # …` note line stay in their bodies.
    #[test]
    fn trailing_comments_after_a_verbatim_body_survive_the_next_save() {
        let raw = "\
[cells]
size = 4,4,1
####
#..#
#..#
####

# anchored to the pedestal, but parked inside the [cells] body
[pedestal]
center = 1,1,0

[part p]
w = 2
h = 1
d = 1
##

# anchored to the door, parked inside the [part] body
[door]
pos = 1,1,0

[note wall 1,1,1]
 # escaped: this is BODY
# but THIS one re_level discards, so it is a comment
";
        let p = capture_preserved(raw);

        let pedestal = p
            .anchors
            .iter()
            .find(|a| a.header == "[pedestal]" && a.key.is_none())
            .expect("the comment after the [cells] body anchors to [pedestal]");
        assert_eq!(
            pedestal.lines,
            vec!["# anchored to the pedestal, but parked inside the [cells] body".to_string()]
        );

        let door = p
            .anchors
            .iter()
            .find(|a| a.header == "[door]" && a.key.is_none())
            .expect("the comment after the [part] body anchors to [door]");
        assert_eq!(
            door.lines,
            vec!["# anchored to the door, parked inside the [part] body".to_string()]
        );

        // EOF inside a `[note]` body — the shape every saved file ends with.
        assert_eq!(
            p.tail,
            vec!["# but THIS one re_level discards, so it is a comment".to_string()]
        );

        // Geometry and escaped body text were NOT stolen.
        let all: Vec<&String> = p
            .anchors
            .iter()
            .flat_map(|a| a.lines.iter())
            .chain(p.tail.iter())
            .collect();
        assert!(
            all.iter()
                .all(|l| !l.contains("####") && l.trim() != "##" && !l.contains("escaped")),
            "geometry rows and escaped note text stay in their bodies: {all:?}"
        );
    }

    /// 1.4-prep: `had_cells_at_load` reflects the LOADED file (true for a
    /// `[cells]`-bearing level, false for a procedural one and for a fresh
    /// document), and `load_path` re-captures preserved text per file.
    #[test]
    fn had_cells_at_load_tracks_cells_section() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(!s.had_cells_at_load, "fresh document = false");
        assert!(s.preserved.is_empty(), "fresh document preserves nothing");

        assert!(load_path(&mut s, "../../levels/08-material-demo.lvl"));
        assert!(s.had_cells_at_load, "[cells]-bearing level = true");
        assert!(
            !s.preserved.is_empty(),
            "08's hand-restored header comments are captured"
        );

        assert!(load_path(&mut s, "../../levels/01.lvl"));
        assert!(!s.had_cells_at_load, "procedural level = false");
    }

    /// The pure crash-recovery decision — ONLY a strictly-newer sidecar
    /// mtime (both mtimes known) yields an offer.
    #[test]
    fn autosave_is_newer_decision() {
        use std::time::{Duration, SystemTime};
        let t = SystemTime::now();
        let older = t - Duration::from_mins(1);
        assert!(
            autosave_is_newer(Some(t), Some(older)),
            "newer sidecar → offer"
        );
        assert!(
            !autosave_is_newer(Some(older), Some(t)),
            "older sidecar → no offer"
        );
        assert!(
            !autosave_is_newer(Some(t), Some(t)),
            "equal mtimes → no offer"
        );
        assert!(
            !autosave_is_newer(None, Some(t)),
            "missing sidecar → no offer"
        );
        assert!(
            !autosave_is_newer(Some(t), None),
            "unreadable original → no offer"
        );
        assert!(!autosave_is_newer(None, None), "nothing known → no offer");
    }

    /// `load_path` OFFERS a strictly-newer autosave sidecar (at the
    /// production `levels/.autosave/<stem>.lvl` path) and stays silent when the
    /// sidecar is absent.
    #[test]
    fn load_path_offers_newer_autosave_sidecar() {
        // The original document, on disk, backdated 2 minutes so the sidecar
        // written below is strictly newer (no mtime-granularity flake). The path
        // is process- and call-unique (`testutil`), so a concurrent test process
        // cannot overwrite or delete this fixture mid-assert, and the guard
        // removes it even if an assertion below panics.
        let mut a = EditorState::new_empty(4, 4, 2);
        a.map.set(1, 1, 0, CellKind::WallSolid);
        let orig_file = ScratchFile::write(
            crate::testutil::scratch_path("w24-offer.lvl"),
            &crate::io::capture_and_serialize(&mut a),
        );
        let orig = orig_file.path().to_string();
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&orig)
            .expect("open orig");
        f.set_modified(std::time::SystemTime::now() - std::time::Duration::from_mins(2))
            .expect("backdate orig");

        // No sidecar yet: a load makes NO offer. The sidecar path is keyed on the
        // ORIGINAL's canonical path (`io::autosave_path_for`), so a unique
        // original buys a unique sidecar for free.
        let sidecar = crate::io::autosave_path_for(Some(&orig));
        let _ = std::fs::remove_file(&sidecar);
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(load_path(&mut s, &orig));
        assert_eq!(s.autosave_offer, None, "no sidecar → no offer");

        // A newer sidecar: the load OFFERS it and says so in the status.
        std::fs::create_dir_all("levels/.autosave").expect("sidecar dir");
        let mut b = EditorState::new_empty(4, 4, 2);
        b.map.set(2, 2, 0, CellKind::WallSolid);
        let sidecar_file =
            ScratchFile::write(sidecar.clone(), &crate::io::capture_and_serialize(&mut b));
        let mut s2 = EditorState::new_empty(4, 4, 2);
        assert!(load_path(&mut s2, &orig));
        assert_eq!(
            s2.autosave_offer.as_deref(),
            Some(sidecar.as_str()),
            "offered"
        );
        assert!(
            s2.status.contains("AUTOSAVE"),
            "status announces it: {}",
            s2.status
        );
        assert!(
            !s2.dirty,
            "offering does not dirty the freshly-loaded document"
        );

        drop(sidecar_file);
        drop(orig_file);
        // (No shared-dir cleanup: removing `levels/.autosave` here races the
        // parallel tests — and the parallel PROCESSES — writing their own
        // sidecars into it. An empty leftover dir is invisible to git and harmless.)
    }

    /// Fixture for the accept/reject tests: an ORIGINAL document on disk (wall
    /// at 1,1,0), a SIDECAR with different content (wall at 2,2,0), and a state
    /// that loaded the original with the offer armed. Both files come back as
    /// self-deleting guards.
    ///
    /// `tag` alone was NOT enough uniqueness: it separates the three tests from
    /// each other inside one process, but every process picked the same three
    /// paths, so two concurrent runs deleted each other's fixtures mid-assert.
    /// `testutil::scratch_path` adds the pid and a per-call ordinal, and
    /// [`ScratchFile`] removes the files on EVERY exit path — including a
    /// panicking assertion, which a trailing `remove_file` never reaches.
    fn offered_state(tag: &str) -> (EditorState, ScratchFile, ScratchFile) {
        let mut a = EditorState::new_empty(4, 4, 2);
        a.map.set(1, 1, 0, CellKind::WallSolid);
        let orig = ScratchFile::write(
            crate::testutil::scratch_path(&format!("w24-{tag}-orig.lvl")),
            &crate::io::capture_and_serialize(&mut a),
        );
        let mut b = EditorState::new_empty(4, 4, 2);
        b.map.set(2, 2, 0, CellKind::WallSolid);
        let side = ScratchFile::write(
            crate::testutil::scratch_path(&format!("w24-{tag}-side.lvl")),
            &crate::io::capture_and_serialize(&mut b),
        );
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(load_path(&mut s, orig.path()), "original loads");
        // Arm the offer directly (mtime-independent), as load_path would have.
        s.autosave_offer = Some(side.path().to_string());
        (s, orig, side)
    }

    /// ACCEPT: `y` on the pending offer swaps the DOCUMENT to the sidecar
    /// content while `path` stays the ORIGINAL file, marks it dirty (an explicit
    /// save is required to keep it), and clears the offer.
    #[test]
    fn autosave_offer_accept_restores_sidecar_content() {
        use re_present::{EditorKey, EditorKeyCode as E};
        let (mut s, orig, side) = offered_state("accept");
        crate::edit::apply_key(
            &mut s,
            EditorKey {
                code: E::Char('y'),
                shift: false,
            },
        );
        assert_eq!(
            s.map.cell_at(2, 2, 0),
            CellKind::WallSolid,
            "document content replaced by the sidecar's"
        );
        assert_ne!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "the original document's cell is gone"
        );
        assert_eq!(
            s.path.as_deref(),
            Some(orig.path()),
            "path stays the ORIGINAL"
        );
        assert!(
            s.dirty,
            "restored content is unsaved until an explicit save"
        );
        assert_eq!(s.autosave_offer, None, "offer cleared");
        drop(side);
    }

    /// REJECT: `n` clears the offer, touches nothing else, and deletes no
    /// file (the sidecar stays on disk).
    #[test]
    fn autosave_offer_reject_keeps_document_and_sidecar() {
        use re_present::{EditorKey, EditorKeyCode as E};
        let (mut s, orig, side) = offered_state("reject");
        crate::edit::apply_key(
            &mut s,
            EditorKey {
                code: E::Char('n'),
                shift: false,
            },
        );
        assert_eq!(s.autosave_offer, None, "offer cleared");
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "document untouched"
        );
        assert!(!s.dirty, "reject leaves the clean document clean");
        assert!(
            std::path::Path::new(side.path()).exists(),
            "sidecar NOT deleted"
        );
        drop(orig);
    }

    /// SSOT for the REJECT wording: the keyboard `n` arm (in `edit.rs`) and the
    /// overlay's `[ Discard ]` button (which calls [`discard_autosave_offer`])
    /// must leave the SAME status behind, or the two answers to one question
    /// would report differently. `edit.rs`'s arm still spells the literal out;
    /// this test is what stops the two from drifting apart.
    #[test]
    fn discard_status_is_the_same_through_the_key_and_the_button() {
        use re_present::{EditorKey, EditorKeyCode as E};
        let (mut by_key, _o1, _s1) = offered_state("discard-key");
        crate::edit::apply_key(
            &mut by_key,
            EditorKey {
                code: E::Char('n'),
                shift: false,
            },
        );

        let (mut by_button, _o2, _s2) = offered_state("discard-btn");
        discard_autosave_offer(&mut by_button);

        assert_eq!(
            by_key.status, by_button.status,
            "the `n` key and the [ Discard ] button must say the same thing"
        );
        assert_eq!(by_key.status, AUTOSAVE_DISCARDED_STATUS);
        assert_eq!(by_key.autosave_offer, None);
        assert_eq!(by_button.autosave_offer, None);
    }

    /// THE FIX: the offer is MODAL — any other key is SWALLOWED, leaving
    /// the offer standing and performing NO normal action. It used to be a
    /// transient prompt that any keystroke discarded, so pressing `s` after a
    /// crash-restart both dismissed the recovery AND deleted the sidecar (the
    /// save path removes it), losing the recovered work permanently.
    #[test]
    fn autosave_offer_is_modal_and_survives_a_stray_key() {
        use re_present::{EditorKey, EditorKeyCode as E};
        let (mut s, _orig, side) = offered_state("fall");
        assert_eq!(s.cursor, (0, 0));

        // An arrow does NOT move the edit cursor while the offer is up…
        crate::edit::apply_key(
            &mut s,
            EditorKey {
                code: E::Right,
                shift: false,
            },
        );
        assert_eq!(
            s.autosave_offer.as_deref(),
            Some(side.path()),
            "offer stands"
        );
        assert_eq!(s.cursor, (0, 0), "the key performed no normal action");
        assert!(
            s.status.contains("AUTOSAVE"),
            "the question is re-stated: {}",
            s.status
        );

        // …and neither does `s`, the keystroke that used to destroy the sidecar.
        crate::edit::apply_key(
            &mut s,
            EditorKey {
                code: E::Char('s'),
                shift: false,
            },
        );
        assert_eq!(
            s.autosave_offer.as_deref(),
            Some(side.path()),
            "`s` cannot dismiss it"
        );
        assert_ne!(s.status, "SAVE", "`s` did not reach the save path");
        assert!(
            std::path::Path::new(side.path()).exists(),
            "the sidecar is still on disk"
        );

        // Only y / n / Esc answer it. Esc dismisses without touching the sidecar.
        assert!(dismiss_autosave_offer(&mut s), "Esc answers the offer");
        assert_eq!(s.autosave_offer, None, "offer cleared");
        assert!(
            std::path::Path::new(side.path()).exists(),
            "Esc keeps the sidecar file"
        );
        assert!(!dismiss_autosave_offer(&mut s), "no offer left to dismiss");

        // With the offer gone the chain works normally again.
        crate::edit::apply_key(
            &mut s,
            EditorKey {
                code: E::Right,
                shift: false,
            },
        );
        assert_eq!(s.cursor, (1, 0), "normal editing resumes once answered");
    }

    /// Open the `:` palette carrying `target` as its only level file and park the
    /// selection ON that row, so `execute_selected` runs the OpenLevel action
    /// (row 0 of an unfiltered palette is a keymap binding, not a file).
    fn select_open_level_row(state: &mut EditorState, target: &str) {
        use crate::keymap::PaletteAction;
        state.cmd_palette = Some(crate::state::CmdPaletteState {
            query: String::new(),
            sel: 0,
            files: vec![target.to_string()],
        });
        let sel = crate::keymap::palette_matches(state)
            .iter()
            .position(|e| matches!(&e.action, Some(PaletteAction::OpenLevel(p)) if p == target))
            .expect("the open-level row is a palette candidate");
        state.cmd_palette.as_mut().expect("palette open").sel = sel;
    }

    /// THE FIX: the `:` command palette's `open level …` row goes through
    /// the SAME dirty guard as the overlay. On a dirty document the first Enter
    /// only ARMS (nothing loads, history intact); a second confirm loads. It used
    /// to call `load_path` directly — one fuzzy hit destroyed the session.
    #[test]
    fn palette_open_on_dirty_document_requires_a_second_confirm() {
        let mut s = EditorState::new_empty(4, 4, 2);
        // Make the document dirty via a real edit (which also fills history).
        s.brush = CellKind::WallSolid;
        s.cursor = (1, 1);
        crate::edit::apply_tool_at_cursor(&mut s);
        assert!(s.dirty, "edit made it dirty");
        let old_path = s.path.clone();
        let old_size = s.map.size();
        let undo_before = s.history.undo_len();
        assert!(undo_before > 0, "there is undo history to lose");

        // Run the palette's `open level <target>` row exactly as Enter does.
        let target = "../../levels/01.lvl";
        select_open_level_row(&mut s, target);
        crate::keymap::execute_selected(&mut s);

        // ARMED, not loaded.
        assert!(s.open_armed, "the palette open arms the dirty guard");
        assert!(s.open.is_some(), "the overlay stays up on the picked row");
        assert_eq!(s.path, old_path, "NOTHING was loaded");
        assert_eq!(s.map.size(), old_size, "the document is untouched");
        assert!(s.dirty, "still dirty");
        assert_eq!(s.history.undo_len(), undo_before, "undo history intact");
        assert!(
            s.status.contains("UNSAVED"),
            "the guard says so: {}",
            s.status
        );

        // The second confirm (the overlay's Enter) loads.
        pick(&mut s);
        assert_eq!(s.path.as_deref(), Some(target), "the confirm loads");
        assert!(!s.dirty, "load clears dirty");
        assert!(s.open.is_none(), "overlay closed");
    }

    /// A CLEAN document still opens from the palette in ONE step — the
    /// guard costs the user nothing when there is nothing to lose.
    #[test]
    fn palette_open_on_clean_document_loads_immediately() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(!s.dirty, "a fresh document is clean");
        let target = "../../levels/01.lvl";
        select_open_level_row(&mut s, target);
        crate::keymap::execute_selected(&mut s);
        assert_eq!(
            s.path.as_deref(),
            Some(target),
            "clean ⇒ loads on the first Enter"
        );
        assert!(s.open.is_none(), "no overlay left behind");
        assert!(!s.open_armed, "nothing armed");
    }

    /// `scan_lvl_files` NEVER returns a path containing `.autosave` — the
    /// scan is non-recursive over `levels/` and the `.autosave` dir entry itself
    /// has no `.lvl` extension, so sidecars can never appear in the Open picker.
    #[test]
    fn scan_lvl_files_ignores_autosave_dir() {
        // Process-unique probe name: the CWD-relative `levels/.autosave/` is
        // shared by every concurrent test process, so a fixed name meant one run
        // deleting the probe another run was still asserting against — and a
        // leaked probe would sit in the worktree forever.
        let probe = ScratchFile::write(
            format!(
                "levels/.autosave/{}.lvl",
                crate::testutil::unique_name("w24-scan-probe")
            ),
            "x",
        );
        let files = scan_lvl_files();
        assert!(
            files.iter().all(|f| !f.contains(".autosave")),
            "no scanned path mentions .autosave: {files:?}"
        );
        drop(probe);
        // (No shared-dir cleanup: removing `levels/.autosave` here races the
        // parallel tests that are writing their own sidecars into it — an empty
        // leftover dir is invisible to git and harmless.)
    }

    #[test]
    fn overlay_navigation_clamps() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.open = Some(OpenState {
            files: vec!["a".into(), "b".into(), "c".into()],
            cursor: 0,
        });
        navigate(&mut s, -1); // clamps at 0
        assert_eq!(s.open.as_ref().unwrap().cursor, 0);
        navigate(&mut s, 2);
        assert_eq!(s.open.as_ref().unwrap().cursor, 2);
        navigate(&mut s, 5); // clamps at last
        assert_eq!(s.open.as_ref().unwrap().cursor, 2);
    }
}
