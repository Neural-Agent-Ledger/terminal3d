//! Where a relative path in the game's content actually lives.
//!
//! Every asset reference the game reads used to be resolved against the **process
//! working directory**. That is not a stable anchor for anything: it is set by
//! whoever launched us. Explorer sets it to the exe's folder (so double-clicking
//! happened to work), a Start-menu shortcut sets it to `%HOMEPATH%`, a PATH install
//! sets it to wherever the player was standing, and `cargo test` sets it to the
//! crate directory. The same bytes on disk therefore resolved differently depending
//! on how the program was started — which produced two separate defects: a shipped
//! `terminal-state.exe` could not find `levels/`, and a `[texture] file = …` inside
//! a `.lvl` silently failed to load under `cargo test`, so the golden frames pinned
//! the untextured fallback render instead of the image sampler they were curated for.
//!
//! This module replaces the CWD with two stable anchors:
//!
//! * **For an asset named by a `.lvl`** — the level file's own location. A level at
//!   `<root>/levels/TX5-genplate.lvl` writing `file = assets/texture/genplate_tex.png`
//!   means "the assets that ship with me", and the only thing that reliably knows
//!   where that is, is the level file's own path. See [`crate::assets::resolve_level_asset`].
//! * **For the `levels/` directory itself** — an explicit `TS_CONTENT_ROOT` env var
//!   when set to a real directory, then the running executable's location, then its
//!   ancestors (which is what makes a `cargo`-built binary under `target/debug/` find
//!   the repo's content tree from any CWD). See [`crate::assets::find_content_dir`].
//!
//! The CWD survives in both orders as the LAST candidate, so every workflow that
//! relies on it today (`cargo run` from the repo root, `--dump levels/01.lvl`,
//! `ts_editor`, `re_imgqa`) keeps working unchanged.
//!
//! **`TS_CONTENT_ROOT` (added for the two-repo split).** Once the engine and the game live in
//! two separate cargo workspaces sharing one `target/` directory outside either repo
//! root (to build Bevy once, not twice), neither the executable-ancestor walk nor the
//! CWD reliably lands inside either repo's own `levels/`. `TS_CONTENT_ROOT`, if set and
//! naming a directory that exists, is the first candidate probed — set via each repo's
//! own `.cargo/config.toml` (`[env] TS_CONTENT_ROOT = { value = ".", relative = true }`),
//! it resolves to that repo's root for every `cargo run`/`cargo test` invocation. It
//! MUST be read at runtime (`std::env::var`, see [`crate::assets::content_roots`]) rather than baked
//! in at compile time (`env!`/`option_env!`): a compile-time read would enter cargo's
//! build fingerprint and make the two workspaces invalidate and rebuild each other's
//! artifacts merely because their `.cargo/config.toml` values differ. A shipped `.exe`
//! never has this variable set, so its behaviour — executable dir, then ancestors, then
//! CWD — is completely unaffected; this is purely a development/test-time anchor.
//!
//! **A candidate is only accepted if it exists on disk.** Nothing here invents a
//! path or silently substitutes a different file: when no candidate exists the
//! resolver reports the whole search so the caller can say exactly what it looked
//! for. Pure filesystem queries — no terminal, no rendering.

use std::path::{Path, PathBuf};

/// How many ancestors of a starting directory are probed when looking for a
/// content directory (see [`find_content_dir`]).
///
/// Four is chosen to cover the deepest layout cargo produces and no more:
/// `<root>/target/<triple>/<profile>/deps/` is exactly four steps below `<root>`,
/// which is where an integration-test binary lives. Bounding the walk keeps the
/// search from wandering up into unrelated directories (e.g. finding some
/// stranger's `C:\levels`) and keeps it O(1).
const MAX_ANCESTOR_PROBE: usize = 4;

/// The directory containing the running executable, if it can be determined.
///
/// `std::env::current_exe` can fail (or yield a path with no parent) on exotic
/// platforms; that is not an error worth propagating — the caller simply falls
/// through to its next anchor.
#[must_use]
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(Path::to_path_buf)
}

/// A directory and up to [`MAX_ANCESTOR_PROBE`] of its ancestors, nearest first.
fn dir_and_ancestors(start: &Path) -> Vec<PathBuf> {
    start
        .ancestors()
        .take(MAX_ANCESTOR_PROBE + 1)
        .map(Path::to_path_buf)
        .collect()
}

/// The directories that may hold the game's content tree, in priority order.
///
/// 0. `TS_CONTENT_ROOT`, if the env var is set AND names a directory that exists.
///    Read at runtime via `std::env::var` — never
///    `env!`/`option_env!` — so it cannot enter the build fingerprint. Absent (as
///    on a shipped `.exe`, and in any environment that never set it) this step
///    contributes nothing and every candidate below is exactly what it was before
///    this anchor existed.
/// 1. The executable's own directory — the shipped layout: `terminal-state.exe`
///    sits beside `levels/` and `assets/`.
/// 2. Its bounded ancestors — a `cargo`-built binary lives at
///    `<root>/target/debug/terminal-state.exe`, so the repo's content tree is two
///    or three steps up. This is what lets `cargo run` and `cargo test` work from
///    ANY working directory.
/// 3. The current working directory — last, and only as a compatibility anchor for
///    the "run it from the repo root" workflow.
///
/// Duplicates are removed while preserving order, so a bare exe sitting in the repo
/// root does not probe the same directory twice.
#[must_use]
pub fn content_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = content_root_env_candidate(std::env::var("TS_CONTENT_ROOT").ok()) {
        roots.push(dir);
    }
    if let Some(dir) = exe_dir() {
        roots.extend(dir_and_ancestors(&dir));
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    dedup_preserving_order(roots)
}

/// Whether an already-read `TS_CONTENT_ROOT` value should be used as a
/// content root: only when `value` is `Some` and names a directory that
/// exists on disk.
///
/// Factored out of [`content_roots`] so this rule can be tested directly
/// against controlled inputs instead of by mutating the process-wide
/// `TS_CONTENT_ROOT` environment variable — every test in this binary shares
/// that variable, and `cargo test` itself now sets it for the whole run (via
/// this workspace's `.cargo/config.toml`), so mutating it is reserved for the
/// handful of tests that specifically prove the env-anchor's precedence.
fn content_root_env_candidate(value: Option<String>) -> Option<PathBuf> {
    let dir = PathBuf::from(value?);
    dir.is_dir().then_some(dir)
}

/// Find a content directory (`"levels"`, `"assets"`, …) under one of the
/// [`content_roots`].
///
/// Returns the first root where `<root>/<name>` is a directory. `name` is joined
/// verbatim; an absolute `name` short-circuits to itself (and still has to exist).
///
/// # Errors
///
/// [`NotFound`] when no root has such a directory — carrying every place that was
/// looked, so the caller can say where it searched. Never a guess, never a
/// fabricated path.
pub fn find_content_dir(name: &str) -> Result<PathBuf, NotFound> {
    let named = Path::new(name);
    if named.is_absolute() {
        let searched = vec![named.to_path_buf()];
        return named
            .is_dir()
            .then(|| named.to_path_buf())
            .ok_or_else(|| NotFound {
                what: name.to_string(),
                searched,
            });
    }
    let searched: Vec<PathBuf> = content_roots().into_iter().map(|r| r.join(name)).collect();
    let hit = searched.iter().find(|p| p.is_dir()).cloned();
    hit.ok_or_else(|| NotFound {
        what: name.to_string(),
        searched,
    })
}

/// Nothing on disk matched a search. Carries the full candidate list so the caller
/// can tell the player exactly where it looked instead of "file not found".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotFound {
    /// The relative path that was being resolved (as authored).
    pub what: String,
    /// Every absolute candidate that was probed, in the order they were probed.
    pub searched: Vec<PathBuf>,
}

impl std::fmt::Display for NotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}' not found; looked in:", self.what)?;
        for p in &self.searched {
            write!(f, "\n    {}", p.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for NotFound {}

/// The candidates for a relative asset path authored inside a `.lvl`, in the order
/// they are probed.
///
/// `level_source` is the `.lvl`'s own path ([`crate::Level::source_path`]); `None`
/// (a level parsed from a string) means there is no file to anchor to and only the
/// CWD-relative candidate remains — exactly the historical behaviour.
///
/// For a level at `<root>/levels/NAME.lvl`:
///
/// 1. `<root>/levels/<rel>` — an asset that sits beside the level file itself.
/// 2. `<root>/<rel>` — an asset beside the level TREE. This is the convention every
///    shipped level actually uses (`file = assets/texture/genplate_tex.png`,
///    `file = levels/<model>.vox`), and it is what makes an installed
///    `<install>/levels/*.lvl` find `<install>/assets/…`.
/// 3. `<cwd>/<rel>` — the historical anchor, kept last so nothing that works today
///    stops working.
///
/// An ABSOLUTE `rel` is its own single candidate.
#[must_use]
pub fn level_asset_candidates(level_source: Option<&Path>, rel: &str) -> Vec<PathBuf> {
    let relp = Path::new(rel);
    if relp.is_absolute() {
        return vec![relp.to_path_buf()];
    }
    let mut out: Vec<PathBuf> = Vec::with_capacity(3);
    if let Some(dir) = level_source.and_then(Path::parent) {
        out.push(dir.join(relp));
        if let Some(up) = dir.parent() {
            out.push(up.join(relp));
        }
    }
    // The CWD-relative candidate, spelled as the bare relative path so the error
    // message reads the way the author wrote it.
    out.push(relp.to_path_buf());
    dedup_preserving_order(out)
}

/// Resolve a relative asset path authored inside a `.lvl` to a path that EXISTS.
///
/// Probes [`level_asset_candidates`] in order and returns the first hit.
/// Resolution is read-only: the authored string is never rewritten, so a level
/// always round-trips byte-for-byte.
///
/// # Errors
///
/// [`NotFound`] when no candidate exists, listing every one that was probed. The
/// caller decides whether that is fatal (`build_map_checked`) or skippable
/// (`build_map`, which renders the level without the missing asset).
pub fn resolve_level_asset(level_source: Option<&Path>, rel: &str) -> Result<PathBuf, NotFound> {
    let searched = level_asset_candidates(level_source, rel);
    let hit = searched.iter().find(|p| p.exists()).cloned();
    hit.ok_or_else(|| NotFound {
        what: rel.to_string(),
        searched,
    })
}

/// Deduplicate while preserving first-seen order. The lists here are at most a
/// handful of entries, so the quadratic scan is cheaper than building a set.
fn dedup_preserving_order(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::with_capacity(paths.len());
    for p in paths {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The workspace root, derived at COMPILE time so these tests do not depend on
    /// the very thing they are testing (the process CWD).
    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    /// THE BUG. A shipped level writes `assets/texture/genplate_tex.png`, which is
    /// relative to the level TREE's parent, not to the CWD. Under `cargo test` the
    /// CWD is `crates/re_level`, so the bare relative path does not exist — yet the
    /// texture must still resolve, because the level knows where it lives.
    #[test]
    fn texture_beside_the_level_tree_resolves_regardless_of_cwd() {
        let lvl = root().join("levels").join("TX5-genplate.lvl");
        assert!(lvl.is_file(), "fixture: TX5-genplate.lvl must ship");
        let rel = "assets/texture/genplate_tex.png";
        assert!(
            !Path::new(rel).exists(),
            "precondition: the CWD-relative path must NOT resolve here, \
             otherwise this test proves nothing (cwd = {:?})",
            std::env::current_dir()
        );
        let hit = resolve_level_asset(Some(&lvl), rel).expect("resolves via the level's own tree");
        assert!(hit.is_file(), "resolved to a real file: {}", hit.display());
        assert!(
            hit.ends_with("assets/texture/genplate_tex.png"),
            "{}",
            hit.display()
        );
    }

    /// A `[vox] file = levels/<model>.vox` is likewise level-tree relative — proven
    /// here with any other real file that sits beside the level tree the same way a
    /// `.vox` model would (resolution is existence-only; it never decodes).
    #[test]
    fn vox_model_beside_the_level_tree_resolves_regardless_of_cwd() {
        let lvl = root().join("levels").join("TX5-genplate.lvl");
        let hit = resolve_level_asset(Some(&lvl), "levels/TX6-subcell-ladder.lvl")
            .expect("resolves via the level's own tree");
        assert!(hit.is_file(), "{}", hit.display());
    }

    /// An asset sitting NEXT TO the level file wins over one further up — the
    /// nearest anchor is the most specific one.
    #[test]
    fn asset_beside_the_level_file_wins_over_the_tree_parent() {
        // `A5-child.lvl` sits in `levels/`, so from `levels/A5-inset.lvl` the
        // sibling candidate resolves first.
        let lvl = root().join("levels").join("A5-inset.lvl");
        let hit = resolve_level_asset(Some(&lvl), "A5-child.lvl").expect("sibling resolves");
        assert_eq!(hit, root().join("levels").join("A5-child.lvl"));
    }

    /// An absolute path is used verbatim and is its own only candidate — no
    /// searching, no surprises.
    #[test]
    fn absolute_asset_paths_are_used_verbatim() {
        let abs = root()
            .join("assets")
            .join("texture")
            .join("genplate_tex.png");
        let abs_str = abs.to_string_lossy().to_string();
        let lvl = root().join("levels").join("TX5-genplate.lvl");
        assert_eq!(
            level_asset_candidates(Some(&lvl), &abs_str),
            vec![abs.clone()]
        );
        assert_eq!(resolve_level_asset(Some(&lvl), &abs_str).unwrap(), abs);
    }

    /// A level parsed from a STRING has no anchor, so exactly one candidate remains:
    /// the historical CWD-relative one. Behaviour is unchanged for that path.
    #[test]
    fn a_sourceless_level_keeps_the_historical_cwd_candidate_only() {
        assert_eq!(
            level_asset_candidates(None, "assets/texture/x.png"),
            vec![PathBuf::from("assets/texture/x.png")]
        );
    }

    /// A miss is REPORTED, never guessed: the error names the authored path and
    /// every place that was probed.
    #[test]
    fn a_missing_asset_reports_every_candidate() {
        let lvl = root().join("levels").join("TX5-genplate.lvl");
        let err = resolve_level_asset(Some(&lvl), "assets/texture/no_such_file.png")
            .expect_err("must not resolve");
        assert_eq!(err.what, "assets/texture/no_such_file.png");
        assert_eq!(
            err.searched.len(),
            3,
            "level dir, tree parent, cwd: {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("no_such_file.png"), "{msg}");
        assert!(msg.contains("looked in"), "{msg}");
    }

    /// THE OTHER BUG. `levels/` is found from the running binary's location, not
    /// from the CWD. Under `cargo test` the binary lives in `target/.../deps/`, so
    /// only the ancestor walk can find the repo's `levels/`.
    ///
    /// Takes `env_lock()` (uncontended in the common case) because its assertion
    /// — THIS SPECIFIC `levels/` dir, not just any directory named `levels` —
    /// would be a false failure if it happened to run while one of the
    /// `TS_CONTENT_ROOT` precedence tests below has temporarily pointed that env
    /// var at an unrelated scratch directory.
    #[test]
    fn the_levels_directory_is_found_from_the_executable_not_the_cwd() {
        let _lock = env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let found = find_content_dir("levels").expect("levels/ is findable from the test binary");
        assert!(found.is_dir(), "{}", found.display());
        assert!(
            found.join("01.lvl").is_file(),
            "found the REAL levels dir, not some empty namesake: {}",
            found.display()
        );
    }

    /// A directory that exists nowhere is an error listing the search, not a
    /// silent empty result.
    #[test]
    fn a_missing_content_dir_reports_every_root_searched() {
        let err = find_content_dir("no-such-content-dir-4e7b").expect_err("must not be found");
        assert!(!err.searched.is_empty(), "the search list is never empty");
        assert!(err.to_string().contains("no-such-content-dir-4e7b"));
    }

    /// The ancestor probe is BOUNDED — it must not walk to the filesystem root.
    #[test]
    fn the_ancestor_probe_is_bounded() {
        let deep = Path::new("/a/b/c/d/e/f/g/h/i");
        assert_eq!(dir_and_ancestors(deep).len(), MAX_ANCESTOR_PROBE + 1);
    }

    /// The root list never repeats a directory (a bare exe in the CWD would
    /// otherwise probe the same place twice).
    #[test]
    fn content_roots_are_deduplicated() {
        let roots = content_roots();
        let mut seen = roots.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), roots.len(), "duplicate root in {roots:?}");
    }

    // ── TS_CONTENT_ROOT ────────────────────────────────────────────────────

    /// The lock shared by every test below that reads or writes the
    /// process-wide `TS_CONTENT_ROOT` (directly, or through [`content_roots`]
    /// / [`find_content_dir`]) — including
    /// `the_levels_directory_is_found_from_the_executable_not_the_cwd` above.
    /// A `Mutex<()>` behind a `OnceLock` because `static` items cannot run
    /// non-const initialisers.
    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// RAII guard: holds [`env_lock`] for its lifetime and restores
    /// `TS_CONTENT_ROOT` to whatever it was before on drop — including when a
    /// wrapped assertion panics — so one test's env mutation can never leak
    /// into a sibling running later in the same process.
    struct ContentRootEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prior: Option<String>,
    }

    impl ContentRootEnvGuard {
        /// Point `TS_CONTENT_ROOT` at `dir` for the guard's lifetime.
        fn set(dir: &Path) -> Self {
            let lock = env_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let prior = std::env::var("TS_CONTENT_ROOT").ok();
            std::env::set_var("TS_CONTENT_ROOT", dir);
            Self { _lock: lock, prior }
        }

        /// Clear `TS_CONTENT_ROOT` for the guard's lifetime (simulating an
        /// environment — such as a shipped `.exe` — that never set it).
        fn unset() -> Self {
            let lock = env_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let prior = std::env::var("TS_CONTENT_ROOT").ok();
            std::env::remove_var("TS_CONTENT_ROOT");
            Self { _lock: lock, prior }
        }
    }

    impl Drop for ContentRootEnvGuard {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(v) => std::env::set_var("TS_CONTENT_ROOT", v),
                None => std::env::remove_var("TS_CONTENT_ROOT"),
            }
        }
    }

    /// Pure-logic tests for the "set AND names a real directory" rule — no
    /// process env mutation, so these carry no race risk at all.
    #[test]
    fn content_root_env_candidate_accepts_a_real_directory() {
        let r = root();
        assert!(r.is_dir(), "precondition");
        assert_eq!(
            content_root_env_candidate(Some(r.to_string_lossy().to_string())),
            Some(r)
        );
    }

    #[test]
    fn content_root_env_candidate_rejects_a_path_that_is_not_a_directory() {
        let file = root().join("levels").join("TX5-genplate.lvl");
        assert!(file.is_file(), "precondition: a FILE, not a directory");
        assert_eq!(
            content_root_env_candidate(Some(file.to_string_lossy().to_string())),
            None
        );
    }

    #[test]
    fn content_root_env_candidate_rejects_a_directory_that_does_not_exist() {
        let bogus = root().join("no-such-dir-4e7b-content-root");
        assert!(!bogus.exists(), "precondition");
        assert_eq!(
            content_root_env_candidate(Some(bogus.to_string_lossy().to_string())),
            None
        );
    }

    #[test]
    fn content_root_env_candidate_is_none_when_unset() {
        assert_eq!(content_root_env_candidate(None), None);
    }

    /// PRECEDENCE. When `TS_CONTENT_ROOT` names a real directory,
    /// `content_roots()` probes it FIRST — ahead of the executable-ancestor
    /// and CWD anchors — even though that directory has nothing to do with
    /// the repo, proving precedence rather than mere presence.
    #[test]
    fn ts_content_root_is_probed_first_when_set_and_a_directory() {
        let dir = std::env::temp_dir().join(format!(
            "ts_content_root_precedence_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("create a scratch dir");
        let _guard = ContentRootEnvGuard::set(&dir);

        let roots = content_roots();
        assert_eq!(
            roots.first(),
            Some(&dir),
            "TS_CONTENT_ROOT must be the FIRST candidate: {roots:?}"
        );
    }

    /// PRECEDENCE, proven end-to-end through [`find_content_dir`]: a
    /// `levels/` subdirectory under `TS_CONTENT_ROOT` is found even though
    /// the real repo also has one via the executable-ancestor anchor.
    #[test]
    fn ts_content_root_wins_the_levels_lookup_over_the_executable_anchor() {
        let dir = std::env::temp_dir().join(format!(
            "ts_content_root_levels_lookup_{}_{}",
            std::process::id(),
            line!()
        ));
        let fake_levels = dir.join("levels");
        std::fs::create_dir_all(&fake_levels).expect("create a scratch levels/ dir");
        let _guard = ContentRootEnvGuard::set(&dir);

        let found = find_content_dir("levels").expect("the scratch levels/ dir is findable");
        assert_eq!(
            found, fake_levels,
            "TS_CONTENT_ROOT must win over the real repo's levels/: {found:?}"
        );
    }

    /// `TS_CONTENT_ROOT` set to a path that is not a real directory (deleted,
    /// mistyped, or never created) contributes NOTHING — it is skipped, never
    /// substituted for a guess.
    #[test]
    fn ts_content_root_is_ignored_when_it_is_not_a_directory() {
        let bogus = std::env::temp_dir().join(format!(
            "ts_content_root_does_not_exist_{}_{}",
            std::process::id(),
            line!()
        ));
        assert!(!bogus.exists(), "precondition: must not exist");
        let _guard = ContentRootEnvGuard::set(&bogus);

        let roots = content_roots();
        assert!(
            !roots.contains(&bogus),
            "a non-directory TS_CONTENT_ROOT must not enter the search list: {roots:?}"
        );
    }

    /// THE REGRESSION GUARD for E13: with `TS_CONTENT_ROOT` unset, the exact
    /// pre-E13 algorithm result — executable dir + bounded ancestors, then
    /// CWD, de-duplicated, with NO env-derived entry — is reproduced
    /// byte-for-byte. This is what "unset keeps today's behaviour exactly"
    /// means, proven rather than asserted by inspection.
    #[test]
    fn unset_ts_content_root_keeps_todays_behaviour_exactly() {
        let _guard = ContentRootEnvGuard::unset();

        let roots = content_roots();

        let mut expected: Vec<PathBuf> = Vec::new();
        if let Some(dir) = exe_dir() {
            expected.extend(dir_and_ancestors(&dir));
        }
        if let Ok(cwd) = std::env::current_dir() {
            expected.push(cwd);
        }
        let expected = dedup_preserving_order(expected);

        assert_eq!(
            roots, expected,
            "TS_CONTENT_ROOT unset must not perturb the pre-E13 root list"
        );
    }
}
