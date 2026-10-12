//! TEST-ONLY shared filesystem-fixture helpers.
//!
//! Every editor test that touches the disk used to invent its own fixture name,
//! and most of those names were global constants: `levels/h2-victim.lvl`,
//! `../../levels/_scratch/w24-offer.lvl`. Two things break that:
//!
//! * `cargo test` runs the suite on a thread pool, so two tests in ONE process
//!   race on a shared name;
//! * two test PROCESSES on one worktree (a CI matrix sharing a
//!   checkout, a `--dump` run alongside a test run) race on it too — and the
//!   loser's `remove_file` deletes the winner's fixture mid-assert.
//!
//! Observed damage: a crashed run left a 25-byte `levels/h2-edit-b.lvl` behind
//! and every subsequent run of
//! `io::tests::every_shipped_level_is_byte_stable_through_the_editor` — which
//! enumerates `levels/*.lvl` — then failed on a file no test under review had
//! anything to do with. A test that poisons the corpus directory for all future
//! runs is worse than a flaky test.
//!
//! This module is the ONE uniqueness scheme for the crate's tests. It was
//! hoisted out of `io.rs`, where it lived as a private `scratch_path`, so that
//! `io.rs`, `open.rs`, and `saveas.rs` cannot drift into three different
//! answers to the same question.

use std::sync::atomic::{AtomicU64, Ordering};

/// A monotonic per-process ordinal. Combined with the pid it makes every
/// fixture name distinct both across processes and across concurrent tests
/// inside one process.
fn ordinal() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// `<pid>-<ordinal>` — the uniqueness suffix, made of characters that survive
/// [`crate::saveas::normalize_path`]'s filename filter (ASCII alphanumerics,
/// `-`, `_`) so it can be typed into the Save-As buffer verbatim.
#[must_use]
pub fn unique_tag() -> String {
    format!("{}-{:03}", std::process::id(), ordinal())
}

/// A filename-safe fixture NAME unique to this process and this call, with
/// `base` kept as the readable prefix so a failure still says which fixture it
/// was. For call sites where the DIRECTORY is forced by the code under test and
/// only the name is ours to choose (Save-As, whose `normalize_path` always
/// yields `levels/<name>.lvl`).
#[must_use]
pub fn unique_name(base: &str) -> String {
    format!("{base}-{}", unique_tag())
}

/// Scratch-fixture root for THIS test process: `<temp>/ts_editor-scratch-<pid>`.
///
/// Behaviour-neutral for the tests that use it: they parse through
/// [`re_level::load_string`], which never sets `Level::source_path`, so
/// `re_level`'s asset resolution (`assets::level_asset_candidates`) has no file
/// anchor and probes CWD-relative candidates only — identical whether the
/// fixture sits under `levels/_scratch/` or under the OS temp dir. Keeping it
/// out of the repo also means a leaked fixture can never dirty the worktree.
#[must_use]
pub fn scratch_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ts_editor-scratch-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// A scratch-fixture path unique to this process AND to this call.
///
/// The authored `name` stays in the path, so a failure that quotes the path
/// still names the fixture. Siblings the save path creates (`<path>.tmp`,
/// `<path>.bak`) inherit the uniqueness for free.
#[must_use]
pub fn scratch_path(name: &str) -> String {
    scratch_dir()
        .join(format!("{:03}-{name}", ordinal()))
        .to_string_lossy()
        .into_owned()
}

/// A fixture file that DELETES ITSELF when it goes out of scope — including
/// when the test that owns it panics (the workspace keeps `panic = "unwind"`,
/// so unwinding runs `Drop`).
///
/// A trailing `let _ = fs::remove_file(..)` at the end of a test is skipped by
/// the very failure that most needs cleaning up: an early `assert!` leaves the
/// file on disk forever. That is how `levels/h2-edit-b.lvl` was born. Use this
/// for any fixture written into a directory the repo (or a later test run)
/// cares about.
pub struct ScratchFile {
    path: String,
}

impl ScratchFile {
    /// Create the file's parent directory if needed, write `contents`, and take
    /// ownership of the cleanup.
    ///
    /// # Panics
    ///
    /// If the directory or the file cannot be created — a fixture that failed to
    /// materialize must fail the test loudly, not silently test nothing.
    pub fn write(path: impl Into<String>, contents: &str) -> Self {
        let path = path.into();
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|e| panic!("fixture dir for {path}: {e}"));
            }
        }
        std::fs::write(&path, contents).unwrap_or_else(|e| panic!("fixture {path}: {e}"));
        Self { path }
    }

    /// Take over cleanup of a path the CODE UNDER TEST wrote (a save, an
    /// autosave, a playtest sidecar). Nothing is written or required to exist
    /// here — this is the cleanup half only.
    pub fn adopt(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// The path the fixture was written to.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        // Best-effort: a test that already deleted the file (or a directory the
        // OS is holding) must not turn cleanup into a second failure.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Every shipped `levels/*.lvl`, as `(file name, path)`, sorted by name.
///
/// The ONE enumeration of the corpus for tests — `io`'s byte-stability sweep and
/// `corpus`'s cycle sweep both read it, so neither can end up checking a
/// different set of levels from the other.
#[must_use]
pub fn shipped_levels() -> Vec<(String, std::path::PathBuf)> {
    let dir = std::path::Path::new("../../levels");
    let mut v: Vec<(String, std::path::PathBuf)> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("lvl"))
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().to_string();
            Some((name, p))
        })
        .collect();
    v.sort();
    v
}
