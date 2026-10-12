//! SAVE-AS: a small text-entry prompt to name a never-saved (or to-be-renamed)
//! level, then route the actual write through the existing save flow.
//!
//! Pure functions over [`EditorState`] (no terminal, no I/O). [`open_saveas`]
//! shows the prompt seeded from the current path / level name; the user types a
//! bare name; [`confirm`] normalizes it to a `levels/<name>.lvl` path, sets
//! `state.path`, and closes the prompt WITHOUT writing — the caller then triggers
//! the single save choke point (the Bevy `save_system`) by setting the SAVE flag.
use crate::state::{EditorState, SaveAsState};

/// Open the SAVE-AS prompt, seeding the buffer with a bare editable name. The
/// seed is, in order of preference: the current `state.path` basename, then the
/// level name, then `"untitled"`. Any `levels/` prefix and `.lvl` suffix is
/// stripped so the user edits just the name.
pub fn open_saveas(state: &mut EditorState) {
    let seed = state
        .path
        .as_deref()
        .map(strip_name)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let n = state.level.name.trim();
            if n.is_empty() {
                None
            } else {
                Some(strip_name(n))
            }
        })
        .unwrap_or_else(|| "untitled".to_string());
    state.saveas = Some(SaveAsState { buf: seed });
    state.saveas_armed = false; // a freshly opened prompt is never mid-confirm
    state.status = "save as: type a name  Enter save  Esc cancel".into();
}

/// Close the SAVE-AS prompt without saving.
pub fn close_saveas(state: &mut EditorState) {
    state.saveas = None;
    state.saveas_armed = false; // cancelling disarms the overwrite guard
    state.status = "save-as cancelled".into();
}

/// Append a printable filename character to the buffer. Control chars are
/// ignored; only letters, digits, `-`, and `_` are accepted (the same set
/// [`normalize_path`] keeps), so the visible buffer matches the final name.
pub fn push_char(state: &mut EditorState, c: char) {
    if let Some(sa) = state.saveas.as_mut() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            sa.buf.push(c);
            // The armed target is the name that was confirmed; editing the name
            // re-opens the question.
            state.saveas_armed = false;
        }
    }
}

/// Clear the SAVE-AS buffer (the `[Clr]` button).
pub fn clear(state: &mut EditorState) {
    if let Some(sa) = state.saveas.as_mut() {
        sa.buf.clear();
        state.saveas_armed = false; // the name changed — re-check on confirm
    }
}

/// Backspace: pop the last character of the filename buffer. No-op (status
/// unchanged) when the buffer is already empty; updates the status when it removes
/// a character so the prompt reflects the edit.
pub fn pop_char(state: &mut EditorState) {
    if let Some(sa) = state.saveas.as_mut() {
        if sa.buf.pop().is_some() {
            state.saveas_armed = false; // the name changed — re-check on confirm
            state.status = "save as: type a name  Enter save  Esc cancel".into();
        }
    }
}

/// Strip a `levels/` (or `levels\`) prefix and a `.lvl` suffix from `s`, leaving
/// the bare name. Used to seed the buffer from an existing path or level name.
fn strip_name(s: &str) -> String {
    let s = s.trim();
    let s = s
        .strip_prefix("levels/")
        .or_else(|| s.strip_prefix("levels\\"))
        .unwrap_or(s);
    let s = s.strip_suffix(".lvl").unwrap_or(s);
    s.to_string()
}

/// PURE: normalize an entered filename to a canonical `levels/<name>.lvl` path.
///
/// Trims whitespace; returns `None` for an empty buffer. Strips a leading
/// `levels/` (or `levels\`) and a trailing `.lvl`, then keeps only filename-safe
/// characters (ASCII alphanumerics, `-`, `_`), dropping anything else. If nothing
/// safe remains, returns `None`. Otherwise returns `Some("levels/<name>.lvl")`,
/// guaranteeing the file lands under `levels/` with a `.lvl` extension.
pub fn normalize_path(buf: &str) -> Option<String> {
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return None;
    }
    let bare = strip_name(trimmed);
    let safe: String = bare
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if safe.is_empty() {
        return None;
    }
    Some(format!("levels/{safe}.lvl"))
}

/// Confirm the SAVE-AS prompt. Computes the normalized path from the buffer; if
/// it is empty/invalid, sets a hint and returns `false` WITHOUT closing or
/// writing. Otherwise sets `state.path`, closes the prompt, and returns `true`
/// WITHOUT writing — the caller triggers the real write via the SAVE flag so the
/// single save choke point performs (and dirty-clears on) the actual disk write.
///
/// Overwrite guard: when the normalized path names a DIFFERENT file
/// that already exists on disk, the first confirm does NOT commit — it arms
/// `saveas_armed`, names the file in the status, keeps the prompt open, and
/// returns `false`. A second Enter (still armed, same name) proceeds. Re-saving
/// over the document's OWN path is not an overwrite and never arms. This
/// mirrors the `quit_armed`/`open_armed` two-step: the prompt is seeded with the
/// current file's basename, so one mistyped character used to clobber a shipped
/// campaign level silently.
pub fn confirm(state: &mut EditorState) -> bool {
    let buf = state
        .saveas
        .as_ref()
        .map(|sa| sa.buf.clone())
        .unwrap_or_default();
    match normalize_path(&buf) {
        Some(path) if would_clobber(state, &path) && !state.saveas_armed => {
            state.saveas_armed = true;
            state.status = format!("{path} EXISTS — Enter again to overwrite");
            false
        }
        Some(path) => {
            // A document with NO path was never loaded from disk (startup-fresh
            // or wizard-new): it has no load provenance, so drop any stale
            // preserved comments / `[cells]`-at-load flag inherited from a
            // previously-opened file BEFORE this save names the document —
            // otherwise the old file's comments would re-emit into the new one.
            // A loaded document doing Save-As (path already set) keeps both.
            if state.path.is_none() {
                state.reset_load_provenance();
            }
            state.path = Some(path);
            state.saveas = None;
            state.saveas_armed = false; // the confirm is consumed
            true
        }
        None => {
            state.status = "enter a filename".into();
            false
        }
    }
}

/// Would saving to `dest` destroy an EXISTING, DIFFERENT file? `true` only when
/// `dest` exists on disk and is not the document's own path — re-saving over the
/// file you already have open is the expected outcome of Save-As, not a clobber.
/// The existence check is cwd-relative, exactly like the `io::save_lvl_text`
/// write it guards.
fn would_clobber(state: &EditorState, dest: &str) -> bool {
    state.path.as_deref() != Some(dest) && std::path::Path::new(dest).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::ScratchFile;

    #[test]
    fn normalize_path_handles_the_canonical_cases() {
        assert_eq!(normalize_path(""), None, "empty → None");
        assert_eq!(normalize_path("   "), None, "whitespace-only → None");
        assert_eq!(normalize_path("foo").as_deref(), Some("levels/foo.lvl"));
        assert_eq!(
            normalize_path("levels/foo.lvl").as_deref(),
            Some("levels/foo.lvl"),
            "already-qualified path normalizes idempotently"
        );
        assert_eq!(
            normalize_path("foo.lvl").as_deref(),
            Some("levels/foo.lvl"),
            "bare name + extension"
        );
        // Sanitization drops unsafe chars (space, slash) and keeps the rest.
        assert_eq!(normalize_path("a b/c").as_deref(), Some("levels/abc.lvl"));
        assert_eq!(
            normalize_path("///").as_deref(),
            None,
            "no safe chars → None"
        );
    }

    #[test]
    fn open_saveas_seeds_from_path_basename() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.path = Some("levels/22-tower.lvl".into());
        open_saveas(&mut s);
        assert_eq!(
            s.saveas.as_ref().unwrap().buf,
            "22-tower",
            "seed strips levels/ + .lvl"
        );
    }

    #[test]
    fn open_saveas_seeds_from_level_name_when_no_path() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.path = None;
        s.level.name = "vault".into();
        open_saveas(&mut s);
        assert_eq!(s.saveas.as_ref().unwrap().buf, "vault");
    }

    #[test]
    fn pop_char_removes_last_and_is_noop_when_empty() {
        let mut s = EditorState::new_empty(4, 4, 2);
        open_saveas(&mut s);
        clear(&mut s);
        push_char(&mut s, 'a');
        push_char(&mut s, 'b');
        assert_eq!(s.saveas.as_ref().unwrap().buf, "ab");
        pop_char(&mut s);
        assert_eq!(
            s.saveas.as_ref().unwrap().buf,
            "a",
            "Backspace pops the last char"
        );
        pop_char(&mut s);
        assert_eq!(s.saveas.as_ref().unwrap().buf, "");
        // Empty buffer: no panic, stays empty.
        pop_char(&mut s);
        assert_eq!(s.saveas.as_ref().unwrap().buf, "");
    }

    #[test]
    fn push_char_skips_unsafe_and_keeps_safe() {
        let mut s = EditorState::new_empty(4, 4, 2);
        open_saveas(&mut s);
        clear(&mut s);
        push_char(&mut s, 'a');
        push_char(&mut s, '/'); // skipped
        push_char(&mut s, '1');
        push_char(&mut s, '_');
        push_char(&mut s, ' '); // skipped
        assert_eq!(s.saveas.as_ref().unwrap().buf, "a1_");
    }

    /// The policy `save_system` applies for a path-less SAVE — open the
    /// Save-As prompt and leave the work dirty, never reporting "saved". We drive
    /// the SAME branch `save_system` uses (path is None ⇒ open the prompt).
    #[test]
    fn save_with_no_path_opens_saveas_and_keeps_dirty() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.dirty = true;
        s.path = None;
        // This is exactly what `save_system` does when status == "SAVE" and the
        // document has no path.
        assert!(s.path.is_none());
        open_saveas(&mut s);
        assert!(s.saveas.is_some(), "no-path save opens the Save-As prompt");
        assert!(s.dirty, "work stays dirty — nothing was saved");
        assert!(
            !s.status.starts_with("saved"),
            "status must not claim a save happened: {}",
            s.status
        );
    }

    #[test]
    fn confirm_with_empty_buffer_keeps_open_and_no_path() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.path = None;
        open_saveas(&mut s);
        clear(&mut s);
        assert!(!confirm(&mut s), "empty buffer ⇒ confirm fails");
        assert!(s.saveas.is_some(), "prompt stays open on a failed confirm");
        assert!(s.path.is_none(), "no path set on a failed confirm");
        assert_eq!(s.status, "enter a filename");
    }

    /// 1.4-prep: naming a PATH-LESS document (startup-fresh or wizard-new — it
    /// was never loaded from disk) resets the load provenance, so a previously
    /// opened file's preserved comments / `[cells]` flag can't leak into the
    /// new file. A LOADED document doing Save-As keeps both.
    #[test]
    fn confirm_on_pathless_doc_resets_load_provenance() {
        // Path-less (fresh/wizard) document with stale provenance.
        let mut s = EditorState::new_empty(4, 4, 2);
        s.had_cells_at_load = true;
        s.preserved.tail.push("# stale".into());
        s.path = None;
        open_saveas(&mut s);
        clear(&mut s);
        for c in "freshdoc".chars() {
            push_char(&mut s, c);
        }
        assert!(confirm(&mut s));
        assert!(!s.had_cells_at_load, "fresh doc: [cells] flag reset");
        assert!(s.preserved.is_empty(), "fresh doc: no stale comments leak");

        // Loaded document (path set) doing Save-As keeps its provenance.
        let mut s2 = EditorState::new_empty(4, 4, 2);
        s2.had_cells_at_load = true;
        s2.preserved.tail.push("# mine".into());
        s2.path = Some("levels/original.lvl".into());
        open_saveas(&mut s2);
        clear(&mut s2);
        for c in "copyname".chars() {
            push_char(&mut s2, c);
        }
        assert!(confirm(&mut s2));
        assert!(s2.had_cells_at_load, "loaded doc keeps the [cells] flag");
        assert!(!s2.preserved.is_empty(), "loaded doc keeps its comments");
    }

    /// `confirm` sets the path + closes the modal but does NOT itself write or
    /// clear dirty — only the real disk-writer does that.
    #[test]
    fn confirm_sets_path_closes_modal_but_does_not_clear_dirty() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.dirty = true;
        s.path = None;
        open_saveas(&mut s);
        clear(&mut s);
        for c in "scratchtest".chars() {
            push_char(&mut s, c);
        }
        assert!(confirm(&mut s), "non-empty buffer ⇒ confirm succeeds");
        assert_eq!(s.path.as_deref(), Some("levels/scratchtest.lvl"));
        assert!(s.saveas.is_none(), "prompt closed on success");
        assert!(s.dirty, "confirm does NOT write, so dirty stays set");
    }

    /// Type `name` into an open Save-As prompt, replacing whatever it was seeded
    /// with.
    fn type_name(state: &mut EditorState, name: &str) {
        clear(state);
        for c in name.chars() {
            push_char(state, c);
        }
    }

    /// A Save-As fixture: a process-unique bare NAME, plus the
    /// `levels/<name>.lvl` file behind it as a self-deleting guard.
    ///
    /// The DIRECTORY is not ours to choose: [`normalize_path`] always yields
    /// `levels/<name>.lvl`, so a Save-As test can only exercise the real code
    /// path by writing where that code writes. `levels/` is CWD-relative, which
    /// under `cargo test` (CWD = `crates/ts_editor`) is a scratch directory and
    /// NOT the shipped corpus at `../../levels/` — but it is still shared by
    /// every concurrent test process, so the NAME must carry the uniqueness:
    /// `testutil::unique_name` appends the pid and a per-call ordinal using only
    /// characters `normalize_path` keeps, so the name can be typed into the
    /// Save-As buffer verbatim.
    ///
    /// [`ScratchFile`] removes the file on EVERY exit path, panicking assertions
    /// included. A trailing `remove_file` did not: a run that died on an early
    /// assert left `levels/h2-edit-b.lvl` behind, and (when the run happened to
    /// be rooted at the repo root, where `levels/` IS the corpus) every later
    /// run of `io::tests::every_shipped_level_is_byte_stable_through_the_editor`
    /// then failed on that stub until it was deleted by hand.
    fn existing_level(base: &str) -> (String, ScratchFile) {
        let name = crate::testutil::unique_name(base);
        let file = ScratchFile::write(format!("levels/{name}.lvl"), "# SHIPPED CAMPAIGN LEVEL\n");
        (name, file)
    }

    /// THE FIX: Save-As onto an EXISTING file does not commit on the
    /// first Enter — it arms, names the file, and keeps the prompt open. A
    /// second Enter (same name) proceeds. The prompt is seeded with the current
    /// file's basename, so one mistyped character used to silently clobber a
    /// shipped campaign level.
    #[test]
    fn saveas_onto_existing_file_requires_a_second_confirm() {
        let (victim_name, victim) = existing_level("h2-victim");
        let before = std::fs::read_to_string(victim.path()).expect("fixture readable");

        let mut s = EditorState::new_empty(4, 4, 2);
        s.dirty = true;
        s.path = Some("levels/h2-mine.lvl".into());
        open_saveas(&mut s);
        type_name(&mut s, &victim_name);

        // First Enter: ARM only.
        assert!(
            !confirm(&mut s),
            "first confirm on an existing file refuses"
        );
        assert!(s.saveas_armed, "the overwrite guard is armed");
        assert!(s.saveas.is_some(), "the prompt stays open");
        assert_eq!(
            s.path.as_deref(),
            Some("levels/h2-mine.lvl"),
            "the document still points at its OWN file"
        );
        assert!(
            s.status.contains(&victim_name),
            "the status names the file: {}",
            s.status
        );
        assert!(
            s.status.contains("EXISTS"),
            "the status says why: {}",
            s.status
        );
        assert_eq!(
            std::fs::read_to_string(victim.path()).expect("victim readable"),
            before,
            "nothing was written to the existing file"
        );

        // Second Enter: proceed.
        assert!(confirm(&mut s), "the armed confirm commits");
        assert_eq!(s.path.as_deref(), Some(victim.path()), "path retargeted");
        assert!(s.saveas.is_none(), "prompt closed");
        assert!(!s.saveas_armed, "the confirm is consumed");
    }

    /// EDITING the name after arming re-opens the question — the armed
    /// confirm belongs to the name that was confirmed, never to a later one.
    #[test]
    fn editing_the_name_disarms_the_overwrite_guard() {
        let (a_name, _a) = existing_level("h2-edit-a");
        let (b_name, _b) = existing_level("h2-edit-b");

        let mut s = EditorState::new_empty(4, 4, 2);
        s.path = Some("levels/h2-edit-mine.lvl".into());
        open_saveas(&mut s);
        type_name(&mut s, &a_name);
        assert!(!confirm(&mut s), "arms on the first name");
        assert!(s.saveas_armed);

        // Retype to the OTHER existing file: the arm must not carry over.
        type_name(&mut s, &b_name);
        assert!(!s.saveas_armed, "typing disarmed the guard");
        assert!(
            !confirm(&mut s),
            "the new name arms on its own first confirm"
        );
        assert!(
            s.status.contains(&b_name),
            "status names the NEW file: {}",
            s.status
        );
        assert_eq!(
            s.path.as_deref(),
            Some("levels/h2-edit-mine.lvl"),
            "still nothing retargeted"
        );
    }

    /// The guard costs nothing where there is nothing to destroy — a
    /// NEW name commits on the first Enter, and so does re-saving over the
    /// document's OWN path (that is what Save-As on the current file means).
    #[test]
    fn saveas_to_a_new_name_or_to_itself_commits_immediately() {
        // A name with no file behind it. Process-unique, so no leftover from a
        // concurrent run (or a previous crashed one) can put a file there and
        // turn "free name" into "existing name".
        let free_name = crate::testutil::unique_name("h2-brand-new-name-xyz");
        let mut s = EditorState::new_empty(4, 4, 2);
        s.path = None;
        open_saveas(&mut s);
        type_name(&mut s, &free_name);
        assert!(
            !std::path::Path::new(&format!("levels/{free_name}.lvl")).exists(),
            "the premise: the name really is free"
        );
        assert!(confirm(&mut s), "a free name commits on the first Enter");
        assert!(!s.saveas_armed);

        // The document's own (existing) path.
        let (own_name, own) = existing_level("h2-own");
        let mut s2 = EditorState::new_empty(4, 4, 2);
        s2.path = Some(own.path().to_string());
        open_saveas(&mut s2);
        type_name(&mut s2, &own_name);
        assert!(
            confirm(&mut s2),
            "re-saving over your own file is not a clobber"
        );
        assert!(!s2.saveas_armed);
        assert_eq!(s2.path.as_deref(), Some(own.path()));
    }
}
