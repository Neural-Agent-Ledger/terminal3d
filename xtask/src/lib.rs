//! `xtask` — repository-local developer tasks for the `terminal3d` workspace.
//!
//! Ships one check today, `layers`, which enforces the directory-based
//! dependency direction:
//!
//! ```text
//! engine/*  may depend on:  engine/*, external               (normal, build, AND dev)
//! editor/*  may depend on:  engine/*, editor/*, external      (normal, build, AND dev)
//! apps/*    may depend on:  engine/*, external                (normal, build, AND dev)
//! xtask     may depend on:  external only
//! ```
//!
//! and one narrower rule that direction alone cannot express: no `engine/*`
//! crate other than `re_present` may pull in the terminal backend
//! (`crossterm` / `ratatui` / `bevy_ratatui`), even transitively — see
//! `docs/LAYERING.md` and `CLAUDE.md` rule 2 ("Only `re_present`
//! touches the terminal").
//!
//! `cargo xtask layers` (`main.rs`) and `cargo test -p xtask` (the
//! `#[test]` below) both call [`check_workspace`], so the command a human
//! runs and the assertion CI runs can never drift apart.
//!
//! # Example
//!
//! ```no_run
//! let metadata = xtask::workspace_metadata();
//! let report = xtask::check_workspace(&metadata);
//! assert!(report.is_ok(), "{}", report.describe());
//! ```

use std::collections::HashMap;

use cargo_metadata::{DependencyKind, Metadata, MetadataCommand, Package, PackageId};

/// The four top-level directories a workspace member can live in.
///
/// One path segment below the workspace root — every crate is two
/// directories below the repo root, e.g. `engine/re_core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    Engine,
    Editor,
    Apps,
    Xtask,
}

impl Layer {
    /// Human-readable name, used in violation messages. The match stays
    /// exhaustive (no `_` arm) so a fifth [`Layer`] variant fails to compile
    /// here until this function accounts for it.
    const fn name(self) -> &'static str {
        match self {
            Self::Engine => "engine",
            Self::Editor => "editor",
            Self::Apps => "apps",
            Self::Xtask => "xtask",
        }
    }

    /// May a crate in `self` depend — normal, build, *or* dev — on a crate in
    /// `other`? Both `editor/*` rows of the table collapse to "engine or
    /// editor" because the normal-vs-dev columns are identical for every
    /// layer; nothing here currently needs the distinction.
    const fn may_depend_on(self, other: Self) -> bool {
        match self {
            Self::Engine | Self::Apps => matches!(other, Self::Engine),
            Self::Editor => matches!(other, Self::Engine | Self::Editor),
            Self::Xtask => false,
        }
    }
}

/// Crate names that must never be reachable from an `engine/*` crate other
/// than [`TERMINAL_BACKEND_OWNER`]. `bevy_ratatui` is included alongside its
/// own dependencies `crossterm`/`ratatui` so the check fires at the first
/// crate that pulls the backend in, not only at the leaves.
const TERMINAL_BACKEND_CRATES: &[&str] = &["crossterm", "ratatui", "bevy_ratatui"];

/// The one engine crate allowed to depend on the terminal backend.
const TERMINAL_BACKEND_OWNER: &str = "re_present";

/// One layering rule broken by one edge in the resolved dependency graph.
#[derive(Debug, Clone)]
pub struct Violation(pub String);

/// The result of running every check in [`check_workspace`].
#[derive(Debug, Default)]
pub struct Report {
    pub violations: Vec<Violation>,
    /// How many workspace members were classified — used by the CLI to prove
    /// it actually walked a non-empty graph, not a mis-pointed empty one.
    pub crates_checked: usize,
}

impl Report {
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    /// Multi-line human-readable summary, used by both the CLI and test
    /// failure messages so the two surfaces read identically.
    #[must_use]
    pub fn describe(&self) -> String {
        if self.violations.is_empty() {
            format!(
                "cargo xtask layers: OK — {} workspace crates, no layering violations",
                self.crates_checked
            )
        } else {
            let mut out = format!(
                "cargo xtask layers: {} violation(s) across {} workspace crates:\n",
                self.violations.len(),
                self.crates_checked
            );
            for v in &self.violations {
                out.push_str("  - ");
                out.push_str(&v.0);
                out.push('\n');
            }
            out
        }
    }
}

/// Run `cargo metadata --format-version 1 --locked` against **this workspace's**
/// root manifest (`term3d/Cargo.toml`), regardless of the caller's current
/// directory.
///
/// `--locked` mirrors `no_net_in_game_build.rs`'s rationale: it reads the
/// committed `Cargo.lock` as-is instead of attempting a re-resolve, so the
/// check needs no network access and cannot perturb the lockfile.
///
/// # Panics
///
/// Panics if `cargo metadata` cannot be run or returns malformed output —
/// both indicate a broken toolchain, not a layering violation, so there is no
/// useful `Result` to hand back to a caller.
#[must_use]
pub fn workspace_metadata() -> Metadata {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // `xtask` sits directly under the workspace root (`term3d/xtask`), so the
    // workspace manifest is exactly one level up — unlike a `crates/<name>`
    // member, which would need two.
    let workspace_manifest = manifest_dir
        .parent()
        .expect("xtask/ has a parent directory")
        .join("Cargo.toml");

    MetadataCommand::new()
        .manifest_path(&workspace_manifest)
        .other_options(vec!["--locked".to_owned()])
        .exec()
        .expect("`cargo metadata` failed — is the toolchain on PATH and Cargo.lock committed?")
}

/// Classify a workspace member by the first path segment of its manifest
/// directory relative to `workspace_root`. Returns `None` for a member that
/// is not under any of the four known top-level directories — a
/// classification bug in this tool (or a new top-level directory that needs
/// a [`Layer`] variant), not a dependency violation, so callers should treat
/// it as a hard error rather than silently skipping the crate.
fn classify(pkg: &Package, workspace_root: &camino::Utf8Path) -> Option<Layer> {
    let manifest_dir = pkg.manifest_path.parent()?;
    let rel = manifest_dir.strip_prefix(workspace_root).ok()?;
    match rel.components().next()?.as_str() {
        "engine" => Some(Layer::Engine),
        "editor" => Some(Layer::Editor),
        "apps" => Some(Layer::Apps),
        "xtask" => Some(Layer::Xtask),
        _ => None,
    }
}

/// Check the layer-direction table: every internal edge (workspace member to
/// workspace member) in the resolved graph, of any dependency kind, must be
/// allowed by [`Layer::may_depend_on`].
///
/// This one *does* want the resolved graph (`resolve.nodes[].deps`), not the
/// raw manifest list: an edge only exists here if Cargo actually resolved and
/// built it, which is exactly what "does X depend on Y" should mean for a
/// direction check. (Contrast [`check_terminal_isolation`], which reads raw
/// manifests instead, for a reason specific to feature unification.)
fn check_direction(
    metadata: &Metadata,
    members: &HashMap<&PackageId, (&Package, Layer)>,
) -> Vec<Violation> {
    let Some(resolve) = metadata.resolve.as_ref() else {
        return vec![Violation(
            "cargo metadata returned no resolve graph — run with dependencies present, not --no-deps"
                .to_owned(),
        )];
    };

    let mut violations = Vec::new();
    for node in &resolve.nodes {
        let Some(&(src_pkg, src_layer)) = members.get(&node.id) else {
            continue; // external crate; direction only governs internal edges
        };
        for dep in &node.deps {
            let Some(&(dst_pkg, dst_layer)) = members.get(&dep.pkg) else {
                continue; // dependency on an external crate — not a direction concern
            };
            if !src_layer.may_depend_on(dst_layer) {
                let kinds = dep_kinds_str(&dep.dep_kinds);
                violations.push(Violation(format!(
                    "{} ({}/) depends on {} ({}/) [{kinds}] — {}/ may only depend on {}",
                    src_pkg.name,
                    src_layer.name(),
                    dst_pkg.name,
                    dst_layer.name(),
                    src_layer.name(),
                    allowed_targets_str(src_layer),
                )));
            }
        }
    }
    violations
}

/// Comma-joined dependency-kind labels for a `NodeDep`, e.g. `"normal, dev"`.
fn dep_kinds_str(dep_kinds: &[cargo_metadata::DepKindInfo]) -> String {
    if dep_kinds.is_empty() {
        // Older cargo_metadata schemas (pre "unit-graph") may report no
        // dep_kinds for a normal dependency; treat absence as normal.
        return "normal".to_owned();
    }
    dep_kinds
        .iter()
        .map(|k| match k.kind {
            DependencyKind::Normal => "normal",
            DependencyKind::Development => "dev",
            DependencyKind::Build => "build",
            DependencyKind::Unknown => "unknown",
        })
        .collect::<Vec<_>>()
        .join(", ")
}

const fn allowed_targets_str(layer: Layer) -> &'static str {
    match layer {
        Layer::Engine | Layer::Apps => "engine/*",
        Layer::Editor => "engine/* and editor/*",
        Layer::Xtask => "external crates",
    }
}

/// Check that no `engine/*` crate other than [`TERMINAL_BACKEND_OWNER`]
/// *declares* a dependency on a terminal-backend crate, whether directly or
/// through `re_present` with its terminal feature reachable.
///
/// This deliberately reads each package's own manifest-level dependency list
/// (`Package::dependencies`, i.e. what `[dependencies]`/`[dev-dependencies]`/
/// `[build-dependencies]` literally say) rather than walking the *resolved*
/// graph (`resolve.nodes[].deps`, what `check_direction` uses). The resolved
/// graph has exactly one node per package for the whole workspace, and Cargo
/// unifies that package's enabled features across every consumer — so if
/// `editor/ts_editor` legitimately enables `re_present`'s `backend-terminal`
/// feature, the resolved graph shows `re_present` (and therefore anything
/// that depends on it, transitively) as having `bevy_ratatui` reachable, even
/// for a consumer like `engine/re_runtime` whose own edge specifies
/// `default-features = false` and never asked for it. A manifest-level check
/// has no such unification: it only sees what THIS crate's own `Cargo.toml`
/// requests, which is exactly the question "does this engine crate itself
/// depend on the terminal backend" is asking.
fn check_terminal_isolation(members: &HashMap<&PackageId, (&Package, Layer)>) -> Vec<Violation> {
    let mut violations = Vec::new();
    for &(pkg, layer) in members.values() {
        if layer != Layer::Engine || pkg.name.as_str() == TERMINAL_BACKEND_OWNER {
            continue;
        }
        for dep in &pkg.dependencies {
            if TERMINAL_BACKEND_CRATES.contains(&dep.name.as_str()) {
                violations.push(Violation(format!(
                    "engine crate `{}` directly depends on `{}` (the terminal backend) \
                     — only `{TERMINAL_BACKEND_OWNER}` may",
                    pkg.name, dep.name
                )));
            } else if dep.name == TERMINAL_BACKEND_OWNER
                && (dep.uses_default_features
                    || dep.features.iter().any(|f| f == "backend-terminal"))
            {
                violations.push(Violation(format!(
                    "engine crate `{}` depends on `{TERMINAL_BACKEND_OWNER}` with the terminal \
                     backend reachable (uses_default_features={}, features={:?}) — depend on it \
                     with `default-features = false` and without the `backend-terminal` feature",
                    pkg.name, dep.uses_default_features, dep.features
                )));
            }
        }
    }
    violations
}

/// Run every layering check against an already-fetched [`Metadata`].
///
/// Split out from [`workspace_metadata`] so a test can call `cargo metadata`
/// once and then run this same function on the `Metadata` value it already
/// parsed.
///
/// # Panics
///
/// Panics if a workspace member's manifest directory is not under `engine/`,
/// `editor/`, `apps/` or `xtask/` — that is a classification bug in this
/// tool (or a mis-placed crate), not a layering violation to report as one.
#[must_use]
pub fn check_workspace(metadata: &Metadata) -> Report {
    let workspace_root = &metadata.workspace_root;
    let members: HashMap<&PackageId, (&Package, Layer)> = metadata
        .workspace_members
        .iter()
        .filter_map(|id| {
            let pkg = metadata.packages.iter().find(|p| &p.id == id)?;
            let layer = classify(pkg, workspace_root).unwrap_or_else(|| {
                panic!(
                    "workspace member `{}` at {} is not under engine/, editor/, apps/ or xtask/ \
                     — classify() needs a new case, or this crate is mis-placed",
                    pkg.name, pkg.manifest_path
                )
            });
            Some((id, (pkg, layer)))
        })
        .collect();

    let mut violations = check_direction(metadata, &members);
    violations.extend(check_terminal_isolation(&members));

    Report {
        violations,
        crates_checked: members.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure logic, no `cargo metadata` involved: the allowed-direction table
    /// itself, spelled out so a future edit to `may_depend_on` that silently
    /// loosens a rule fails here first.
    #[test]
    fn direction_table_matches_the_design_doc() {
        let allowed = [
            (Layer::Engine, Layer::Engine, true),
            (Layer::Engine, Layer::Editor, false),
            (Layer::Engine, Layer::Apps, false),
            (Layer::Engine, Layer::Xtask, false),
            (Layer::Editor, Layer::Engine, true),
            (Layer::Editor, Layer::Editor, true),
            (Layer::Editor, Layer::Apps, false),
            (Layer::Apps, Layer::Engine, true),
            (Layer::Apps, Layer::Editor, false),
            (Layer::Apps, Layer::Apps, false),
            (Layer::Xtask, Layer::Engine, false),
            (Layer::Xtask, Layer::Editor, false),
            (Layer::Xtask, Layer::Apps, false),
            (Layer::Xtask, Layer::Xtask, false),
        ];
        for (from, to, expected) in allowed {
            assert_eq!(
                from.may_depend_on(to),
                expected,
                "{}/ depending on {}/ should be {}",
                from.name(),
                to.name(),
                expected
            );
        }
    }

    /// The live integration check: run against this actual workspace's
    /// resolved graph. This is the same function `cargo xtask layers` calls,
    /// so `cargo test -p xtask` and the CLI can never disagree.
    ///
    /// A mutation check (adding `re_lvlcheck` as an `engine/*` crate's
    /// dev-dependency, confirming this test then fails, then reverting) was
    /// performed by hand against a scratch copy of the manifest. It is not
    /// wired in as a permanent test because doing so would require checking
    /// in a second, deliberately-broken manifest tree purely to exercise the
    /// checker, which is more machinery than the one-line `may_depend_on`
    /// logic above warrants.
    #[test]
    fn layering_rules_hold() {
        let metadata = workspace_metadata();
        let report = check_workspace(&metadata);
        assert!(
            report.crates_checked > 0,
            "classified zero workspace crates — metadata path is wrong"
        );
        assert!(report.is_ok(), "{}", report.describe());
    }
}
