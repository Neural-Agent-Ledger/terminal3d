//! Every external file a `.lvl` names, and whether it is actually there.
//!
//! A level references files it does not contain: `[texture NAME] file`, the three
//! `[sky]` images, `[vox NAME] file`, and an `[inset]`'s child scene. Each of those
//! is resolved through [`crate::assets::resolve_level_asset`], which reports a
//! [`crate::assets::NotFound`] carrying **every candidate it probed**.
//!
//! That report used to reach nobody. `build_map` skips a broken asset best-effort
//! (correct — a player would rather have an untextured wall than no game), and
//! `build_map_checked` — the one function that surfaced the error — had no caller
//! outside `re_level`'s own unit tests. Deleting a level's bound texture file
//! therefore produced an untextured level and **zero output**: an authored,
//! parsed, tested asset that is simply not on screen, with nothing anywhere saying
//! so.
//!
//! This module is the reporting surface, deliberately split in two so each
//! consumer pays only for what it needs:
//!
//! * [`Level::missing_assets`] — pure path RESOLUTION (a handful of
//!   `Path::exists` calls), no read, no decode. Cheap enough for the game's load
//!   path, which uses it to name what it is about to render without.
//! * [`Level::audit_assets`] — resolution **plus** read and decode/parse of every
//!   image and `.vox`. The linter's version (`re_lvlcheck`), where paying for a
//!   full decode to answer "is this level valid?" is the whole point.
//!
//! Both collect **every** issue. `build_map_checked` stops at the first (it
//! `?`-propagates), which is fine for "can this map be built" but useless as a
//! gate: a level with three broken textures would report one, get fixed, and fail
//! again. Neither function mutates anything and neither builds a `Map`.

use std::path::PathBuf;

use crate::assets::{self, NotFound};
use crate::{InsetContent, Level};

/// Which authored section an external-file reference came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssetKind {
    /// `[texture NAME] file` — an image decoded into `Map::textures`.
    Texture,
    /// `[sky] cylinder` / `cap_top` / `cap_bottom` — the sky images.
    Sky,
    /// `[vox NAME] file` — a `MagicaVoxel` model stamped into the room.
    Vox,
    /// `[inset] child` — another `.lvl` referenced as a child scene.
    Inset,
}

impl AssetKind {
    /// A short lowercase tag for report lines.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Texture => "texture",
            Self::Sky => "sky",
            Self::Vox => "vox",
            Self::Inset => "inset",
        }
    }
}

/// One external file a level names, exactly as authored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetRef {
    /// Which section it came from.
    pub kind: AssetKind,
    /// The section as a human reads it: `[texture wall]`, `[sky] cylinder`,
    /// `[vox house]`, `[inset #0] child`.
    pub label: String,
    /// The path string as written in the `.lvl` — never rewritten, so an author
    /// can find the line by searching for it.
    pub authored: String,
    /// The reference's index within its OWN kind's collection, in parse order:
    /// `[texture]` → [`Level::textures`], `[vox]` → [`Level::voxes`], `[inset]`
    /// → `Level::overlay.insets`, and for `[sky]` the slot order (`0` cylinder,
    /// `1` cap_top, `2` cap_bottom).
    ///
    /// NOT the position in [`Level::asset_refs`]'s returned vector: that vector
    /// skips references with an empty path, so the two drift apart the moment a
    /// level leaves one `[sky]` slot unset. This field exists so a consumer can
    /// address the offending CLAUSE — `re_lvlcheck` turns it into a
    /// `Location::Element` an editor can select — without re-deriving
    /// `asset_refs`'s ordering or scraping `label` back into an index.
    pub index: usize,
}

/// Why an [`AssetRef`] does not load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetFault {
    /// No candidate path exists on disk. Carries the full probe list.
    Missing(NotFound),
    /// A candidate exists but could not be read (permissions, a directory, a
    /// vanished file).
    Unreadable(String),
    /// The bytes are there but are not a valid image / `.vox`.
    Undecodable(String),
}

impl std::fmt::Display for AssetFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(nf) => write!(f, "{nf}"),
            Self::Unreadable(e) => write!(f, "unreadable — {e}"),
            Self::Undecodable(e) => write!(f, "undecodable — {e}"),
        }
    }
}

/// One broken asset reference: what was authored, and what is wrong with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetIssue {
    /// The reference that failed.
    pub asset: AssetRef,
    /// What went wrong (a [`AssetFault::Missing`] names every path probed).
    pub fault: AssetFault,
}

impl std::fmt::Display for AssetIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} cannot load '{}': {}",
            self.asset.label, self.asset.authored, self.fault
        )
    }
}

impl Level {
    /// Every external file this level names, in a stable order: `[texture]`s in
    /// parse order, then the `[sky]` slots (`cylinder`, `cap_top`, `cap_bottom`), then
    /// `[vox]`es in parse order, then `[inset]` children.
    ///
    /// A reference with an EMPTY path names no file and is not listed — an
    /// `[inset]` with `child =` unset is an inert placeholder by design (it draws
    /// the neutral fallback colour), and an empty `file =` is a parse-time error
    /// the parser already rejects. Pure: no disk access at all.
    #[must_use]
    pub fn asset_refs(&self) -> Vec<AssetRef> {
        let mut out: Vec<AssetRef> = Vec::new();
        let mut push = |kind: AssetKind, index: usize, label: String, authored: &str| {
            if !authored.is_empty() {
                out.push(AssetRef {
                    kind,
                    label,
                    authored: authored.to_string(),
                    index,
                });
            }
        };
        for (i, t) in self.textures.iter().enumerate() {
            push(
                AssetKind::Texture,
                i,
                format!("[texture {}]", t.name),
                &t.file,
            );
            // `frames = clip.gif` is the ANIMATED texture's pixel source — when it is
            // authored it is the file the surface actually shows, so a missing one is
            // exactly the silent "authored asset that never appears" this module
            // exists to report. Listed as its own reference so the message names the
            // key that is broken.
            if let Some(fp) = &t.frames {
                push(
                    AssetKind::Texture,
                    i,
                    format!("[texture {}] frames", t.name),
                    fp,
                );
            }
        }
        for (slot_i, (slot, path)) in [
            ("cylinder", &self.sky.cylinder),
            ("cap_top", &self.sky.cap_top),
            ("cap_bottom", &self.sky.cap_bottom),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(p) = path {
                push(AssetKind::Sky, slot_i, format!("[sky] {slot}"), p);
            }
        }
        for (i, v) in self.voxes.iter().enumerate() {
            push(AssetKind::Vox, i, format!("[vox {}]", v.name), &v.file);
        }
        for (i, inset) in self.overlay.insets.iter().enumerate() {
            let InsetContent::ChildScene(child) = &inset.content;
            push(AssetKind::Inset, i, format!("[inset #{i}] child"), child);
        }
        out
    }

    /// Every asset reference that does not RESOLVE to a file on disk, with the
    /// full candidate list for each.
    ///
    /// Resolution only — no file is read and no image is decoded, so this costs a
    /// few `Path::exists` calls per reference and can sit on the game's load path
    /// without doubling its decode work. A file that resolves but is corrupt is
    /// NOT reported here (see [`audit_assets`](Level::audit_assets)).
    ///
    /// Every failure is collected; there is no first-error short-circuit.
    #[must_use]
    pub fn missing_assets(&self) -> Vec<AssetIssue> {
        let base = self.asset_base();
        self.asset_refs()
            .into_iter()
            .filter_map(
                |asset| match assets::resolve_level_asset(base, &asset.authored) {
                    Ok(_) => None,
                    Err(nf) => Some(AssetIssue {
                        asset,
                        fault: AssetFault::Missing(nf),
                    }),
                },
            )
            .collect()
    }

    /// Every asset reference that does not resolve, read, AND decode — the
    /// thorough audit.
    ///
    /// A superset of [`missing_assets`](Level::missing_assets): each resolved
    /// image is decoded by the same `image` crate `build_map` uses and each `.vox`
    /// is parsed by the same `dot_vox` loader, so a truncated PNG or a corrupt
    /// model is caught here rather than at render time. That costs a full decode
    /// per asset, which is why the game does not call this — the linter does.
    ///
    /// An `[inset]` child is checked for EXISTENCE only: it is another `.lvl`, and
    /// its own validity is what linting that file answers (`lvlcheck` walks the
    /// whole directory). Parsing it here would duplicate that and recurse.
    ///
    /// Every failure is collected; there is no first-error short-circuit.
    #[must_use]
    pub fn audit_assets(&self) -> Vec<AssetIssue> {
        let base = self.asset_base();
        let mut out: Vec<AssetIssue> = Vec::new();
        for asset in self.asset_refs() {
            let resolved: PathBuf = match assets::resolve_level_asset(base, &asset.authored) {
                Ok(p) => p,
                Err(nf) => {
                    out.push(AssetIssue {
                        asset,
                        fault: AssetFault::Missing(nf),
                    });
                    continue;
                }
            };
            if asset.kind == AssetKind::Inset {
                continue;
            }
            let bytes = match std::fs::read(&resolved) {
                Ok(b) => b,
                Err(e) => {
                    out.push(AssetIssue {
                        asset,
                        fault: AssetFault::Unreadable(format!("{}: {e}", resolved.display())),
                    });
                    continue;
                }
            };
            let decoded: Result<(), String> = match asset.kind {
                AssetKind::Texture | AssetKind::Sky => image::load_from_memory(&bytes)
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
                // Scale 1: the audit only asks "does this parse", and the authored
                // downscale cannot turn a valid model invalid.
                AssetKind::Vox => crate::load_vox_solids(&bytes, 1).map(|_| ()),
                AssetKind::Inset => Ok(()),
            };
            if let Err(e) = decoded {
                out.push(AssetIssue {
                    asset,
                    fault: AssetFault::Undecodable(e),
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load, load_string};
    use std::path::Path;

    /// The workspace root, derived at COMPILE time (the tests must not depend on
    /// the process CWD — that is the very thing asset resolution replaced).
    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    fn level_path(name: &str) -> PathBuf {
        root().join("levels").join(name)
    }

    /// A minimal valid level, with `{EXTRA}` sections appended.
    fn level_with(extra: &str) -> String {
        format!(
            "\
[meta]
name = assetaudit
[room]
size = 6 4 4
ambient = solid
{extra}[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
        )
    }

    /// Three broken references in one level: `[texture]`, `[sky]` and `[vox]`.
    fn three_broken() -> String {
        level_with(
            "[texture pic]\nfile = no/such/pic.png\n\
             [sky]\ncylinder = no/such/sky.png\n\
             [vox house]\nfile = no/such/model.vox\noffset = 1,1,1\n",
        )
    }

    /// Every authored reference is listed, in section order, with the label a
    /// human can grep the `.lvl` for.
    #[test]
    fn asset_refs_lists_every_authored_reference_in_order() {
        let lv = load_string(&three_broken(), "<refs>").expect("parses");
        let refs = lv.asset_refs();
        let got: Vec<(&str, &str)> = refs
            .iter()
            .map(|r| (r.kind.tag(), r.label.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("texture", "[texture pic]"),
                ("sky", "[sky] cylinder"),
                ("vox", "[vox house]"),
            ],
            "{refs:?}"
        );
        assert_eq!(refs[0].authored, "no/such/pic.png");
    }

    /// THE GATE FIX. `build_map_checked` `?`-propagates, so a level with three
    /// broken assets reports ONE — fix it and the next run fails again. The audit
    /// collects all three in a single pass.
    #[test]
    fn every_broken_reference_is_collected_not_just_the_first() {
        let lv = load_string(&three_broken(), "<all>").expect("parses");

        let one = lv
            .build_map_checked()
            .expect_err("build_map_checked reports something");
        assert!(one.contains("pic.png"), "it stops at the first: {one}");
        assert!(
            !one.contains("model.vox"),
            "…and never reaches the vox: {one}"
        );

        let all = lv.missing_assets();
        assert_eq!(all.len(), 3, "every broken reference is reported: {all:?}");
        let joined = all
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        for expected in ["pic.png", "sky.png", "model.vox"] {
            assert!(
                joined.contains(expected),
                "{expected} missing from:\n{joined}"
            );
        }
    }

    /// A miss names every path that was probed — the message an author can act on
    /// without guessing which anchor was used.
    #[test]
    fn a_missing_asset_names_every_candidate_probed() {
        let mut broken = load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        broken.textures[0].file = "assets/texture/definitely_not_here.png".into();
        let issues = broken.missing_assets();
        assert_eq!(issues.len(), 1);
        let msg = issues[0].to_string();
        assert!(msg.starts_with("[texture genplate] cannot load"), "{msg}");
        assert!(msg.contains("looked in"), "{msg}");
        let AssetFault::Missing(nf) = &issues[0].fault else {
            panic!("expected Missing, got {:?}", issues[0].fault)
        };
        assert_eq!(nf.searched.len(), 3, "level dir, tree parent, cwd: {nf:?}");
        for cand in &nf.searched {
            assert!(
                cand.ends_with("assets/texture/definitely_not_here.png"),
                "{}",
                cand.display()
            );
        }
    }

    /// A shipped level's assets resolve from ANY cwd (the level anchors them), so
    /// the gate is quiet when nothing is wrong.
    #[test]
    fn a_shipped_level_reports_no_missing_assets() {
        for name in [
            "TX5-genplate.lvl",
            "TX6-subcell-ladder.lvl",
            "TX7-edge-glyphs.lvl",
            "A5-inset.lvl",
        ] {
            let lv = load(level_path(name)).expect("shipped level loads");
            assert!(!lv.asset_refs().is_empty(), "{name} references assets");
            assert!(
                lv.missing_assets().is_empty(),
                "{name}: {:?}",
                lv.missing_assets()
            );
            assert!(
                lv.audit_assets().is_empty(),
                "{name}: {:?}",
                lv.audit_assets()
            );
        }
    }

    /// The split that keeps the game's load path cheap: a file that EXISTS but is
    /// not an image passes `missing_assets` (which never opens it) and is caught
    /// by `audit_assets` (which decodes).
    #[test]
    fn resolution_is_cheap_decoding_is_the_audit() {
        // A real file that is certainly not a PNG: a level file.
        let mut bad = load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        bad.textures[0].file = "levels/TX5-genplate.lvl".into();

        assert!(
            bad.missing_assets().is_empty(),
            "the path resolves — resolution alone cannot know it is not an image"
        );
        let issues = bad.audit_assets();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(
            matches!(issues[0].fault, AssetFault::Undecodable(_)),
            "{:?}",
            issues[0].fault
        );
        assert!(
            issues[0].to_string().contains("undecodable"),
            "{}",
            issues[0]
        );
    }

    /// A `.vox` reference that resolves to non-`.vox` bytes is an UNDECODABLE vox,
    /// not a missing one.
    ///
    /// No shipped level authors `[vox]` any more (the only ones that did stamped
    /// an unlicensed asset kit, since removed), so the reference is injected
    /// in-memory onto a healthy level, pointing at that level's own `.lvl` text —
    /// a real, resolvable file that is certainly not a `.vox`.
    #[test]
    fn a_corrupt_vox_is_reported_as_undecodable() {
        let mut bad = load(level_path("TX5-genplate.lvl")).expect("TX5 loads");
        bad.voxes.push(crate::LevelVox {
            name: "gone".into(),
            file: "levels/TX5-genplate.lvl".into(),
            ..Default::default()
        });
        let issues = bad.audit_assets();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].asset.kind, AssetKind::Vox);
        assert!(
            matches!(issues[0].fault, AssetFault::Undecodable(_)),
            "{:?}",
            issues[0].fault
        );
    }

    /// A level with no external references has nothing to audit and never errors.
    #[test]
    fn a_level_with_no_assets_audits_clean() {
        let lv = load_string(&level_with(""), "<bare>").expect("parses");
        assert!(lv.asset_refs().is_empty());
        assert!(lv.missing_assets().is_empty());
        assert!(lv.audit_assets().is_empty());
    }

    /// A level parsed from a STRING has no anchor, so only the cwd candidate
    /// remains — the historical behaviour, reported rather than hidden.
    #[test]
    fn a_sourceless_level_still_reports_with_its_one_candidate() {
        let lv = load_string(&three_broken(), "<nosrc>").expect("parses");
        assert!(lv.source_path.is_none(), "load_string leaves no anchor");
        let issues = lv.missing_assets();
        let AssetFault::Missing(nf) = &issues[0].fault else {
            panic!("expected Missing")
        };
        assert_eq!(
            nf.searched,
            vec![Path::new("no/such/pic.png").to_path_buf()]
        );
    }

    /// An `[inset]` child that is missing is reported like any other asset — and
    /// one that resolves is NOT parsed by the audit (it is linted as its own
    /// file), so a child scene's own contents never turn into a parent's error.
    #[test]
    fn an_inset_child_is_checked_for_existence_only() {
        let mut broken = load(level_path("A5-inset.lvl")).expect("A5-inset loads");
        assert!(
            broken
                .asset_refs()
                .iter()
                .any(|r| r.kind == AssetKind::Inset),
            "A5-inset names a child scene"
        );
        assert!(broken.audit_assets().is_empty(), "the child resolves");

        let InsetContent::ChildScene(child) = &mut broken.overlay.insets[0].content;
        *child = "A5-no-such-child.lvl".into();
        let issues = broken.audit_assets();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].asset.kind, AssetKind::Inset);
        assert!(
            issues[0].to_string().contains("A5-no-such-child.lvl"),
            "{}",
            issues[0]
        );
    }
}
