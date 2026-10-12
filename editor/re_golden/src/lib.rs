//! `re_golden` — the golden-framebuffer regression harness.
//!
//! The render path it drives (`re_level::load` → `re_world`'s map →
//! `re_render`'s `composite_dump_once`) uses only engine crates, so this
//! crate has no dependency on any host game crate.
//!
//! # Why a library, not just a test file
//!
//! The harness (how a level is rendered and hashed, how a store is
//! read/written, how failures are classified) is split from the CORPUS (which
//! levels, which store file): the harness lives here once, and any number of
//! thin test files each supply their own corpus and store path. This crate's
//! own `tests/engine_corpus.rs` checks the engine/example/archive levels
//! against `tests/golden/engine_fb_hashes.txt`; a host game can add its own
//! test file checking its own story levels against its own store, calling
//! straight into this crate's [`check_corpus`].
//!
//! Every call site shares [`check_corpus`], parameterized over the store path
//! and the curated level list instead of hardcoding both.
//!
//! # What is hashed
//!
//! For each curated level, [`render_level_hash`] calls the same render entry
//! point the `--dump` CLI uses (`re_render::composite::composite_dump_once`)
//! at time = 0 on a fixed 100×32 grid, and hashes the resulting framebuffer
//! TWICE:
//!
//! * `shade` — row-major `{glyph, fg, bg}` (12 bytes/cell, u32-LE): the LOOK.
//! * `geom`  — row-major `{depth, side, surface, material}` (11 bytes/cell):
//!   the CAST.
//!
//! Two digests, because "the look changed" and "the cast changed" are
//! different failures that used to arrive as one red line; see
//! [`Digests::diff_note`].
//!
//! # This suite FAILS CLOSED
//!
//! Every curated entry is positively accounted for on every run — matched,
//! mismatched, unreadable (the level file is missing or no longer parses), or
//! curated with no golden on record. A hash stored for a key that is no
//! longer curated (`ORPHAN-GOLDEN`) is caught too. The only way to pass is for
//! every curated key to have matched; there is no silent "skipped" bucket.
//! See [`check_corpus`] for the exact accounting.
//!
//! # Refresh procedure
//!
//! When a render change is INTENTIONAL and approved, re-run the owning test
//! with `TS_BLESS_GOLDEN=1` set (see each call site's own module doc for its
//! exact `cargo test` invocation). Commit the updated store file together with
//! the render change and a dated note in that store's `tests/golden/README.md`.
//! NEVER bless to silence an unexpected failure — investigate the render diff
//! first (`Digests::diff_note` says which channel moved).
#![warn(missing_docs)]

use re_core::Framebuffer;
use re_level::load;
use re_render::{composite::composite_dump_once, registry_for_map, Emitter, Fov};
use re_world::CameraPose;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

// Framebuffer dimensions — must match the host game's own `DUMP_COLS`/
// `DUMP_ROWS` so the hash captures the same grid its CLI `--dump` produces.
const COLS: usize = 100;
const ROWS: usize = 32;

/// Hex-encode a digest byte array. sha2 0.11 (`digest` 0.11) returns its own
/// `hybrid_array::Array<u8, N>` from `finalize()` rather than the old
/// `generic_array::GenericArray`, and that type does not implement
/// `LowerHex` — so `format!("{:x}", ...)` no longer compiles directly.
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("write to String never fails");
    }
    s
}

/// Placeholder `geom` for a store line written before the geometry channel
/// existed.
const GEOM_UNKNOWN: &str = "-";

/// The two independent digests of one rendered frame.
///
/// The framebuffer carries SHADING channels (`glyph`, `fg`, `bg`) and GEOMETRY
/// channels (`depth`, `side`, `surface`, `material`). Hashing them separately is
/// what lets a red run say *which layer moved*: before this split existed, a
/// glyph-ramp tweak and a raycaster regression produced the same one-line
/// "digest moved", and the reader had to bisect to find out which.
///
/// * `shade` moved, `geom` identical → the CAST is unchanged; a shader/palette
///   change did this. The expected shape after `/iterate-render` work.
/// * BOTH moved → the GEOMETRY moved: the DDA, the map, or the camera. Treat as
///   a raycaster regression until proven otherwise.
/// * `geom` moved, `shade` identical → geometry changed with no visible
///   consequence (odd rather than impossible — a material swap behind an
///   identical glyph/colour); worth reading before blessing.
#[derive(Clone, PartialEq, Eq)]
pub struct Digests {
    /// SHA-256 over row-major `{glyph, fg, bg}` — the LOOK.
    pub shade: String,
    /// SHA-256 over row-major `{depth, side, surface, material}` — the CAST.
    pub geom: String,
}

impl Digests {
    /// Serialised form in the store: `<shade> <geom>` on one line.
    #[must_use]
    pub fn to_store(&self) -> String {
        format!("{} {}", self.shade, self.geom)
    }

    /// Parse a store value. A single token is a LEGACY line, written before
    /// the geometry channel existed: its `geom` is recorded as
    /// [`GEOM_UNKNOWN`], which compares unequal to every real digest, so the
    /// line reports as a mismatch naming the missing channel instead of
    /// silently passing.
    #[must_use]
    pub fn from_store(v: &str) -> Self {
        let mut it = v.split_whitespace();
        Self {
            shade: it.next().unwrap_or_default().to_string(),
            geom: it.next().unwrap_or(GEOM_UNKNOWN).to_string(),
        }
    }

    /// Human-readable account of WHICH channel(s) moved, for the mismatch
    /// report.
    #[must_use]
    pub fn diff_note(&self, other: &Self) -> &'static str {
        // A legacy line has no geom on record at all — that is a coverage gap
        // to be blessed shut, NOT a geometry regression. Saying so plainly
        // matters: the first run after this channel was added reports every
        // level, and reading that as raycaster regressions would be exactly
        // the wrong conclusion.
        if self.geom == GEOM_UNKNOWN {
            return if self.shade == other.shade {
                "no geom on record (legacy line) — shading unchanged; bless to record it"
            } else {
                "no geom on record (legacy line) AND the shading moved — investigate the shade diff first"
            };
        }
        match (self.shade == other.shade, self.geom == other.geom) {
            (false, true) => "SHADING only — the cast is unchanged",
            (false, false) => "GEOMETRY moved — suspect the raycaster/map/camera",
            (true, false) => "geometry moved with identical shading — read before blessing",
            (true, true) => "identical",
        }
    }
}

/// Render one level and return its two lowercase hex SHA-256 digests.
///
/// # Errors
///
/// Returns `Err(reason)` when the level file is missing or fails to parse. The
/// reason is carried back to the caller rather than printed-and-forgotten: a
/// level that cannot be rendered is a COVERAGE HOLE, not a skip, and the
/// caller ([`check_corpus`]) fails the suite on it.
///
/// # Panics
///
/// Panics if the renderer produces a framebuffer smaller than the fixed
/// `COLS`×`ROWS` grid — an internal invariant of `re_render`, never a
/// function of level content, so there is nothing a caller can recover from.
pub fn render_level_hash(level_path: &Path) -> Result<Digests, String> {
    if !level_path.exists() {
        return Err(format!(
            "level file does not exist: {}",
            level_path.display()
        ));
    }

    let lv = match load(level_path) {
        Ok(l) => l,
        Err(e) => {
            return Err(format!("failed to parse {}: {e}", level_path.display()));
        }
    };

    let map = lv.build_map();
    let cam: CameraPose = lv.spawn_camera();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();

    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, &cam, &registry, &Fov::default(), &emitters, &mut fb);

    let mut shade = Sha256::new();
    let mut geom = Sha256::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            let cell = fb.get(col, row).expect("in-bounds cell");
            // SHADING channels — byte-for-byte the pre-existing digest, so
            // every stored `shade` hash stays valid across this move.
            shade.update(cell.glyph.to_le_bytes());
            shade.update(cell.fg.to_le_bytes());
            shade.update(cell.bg.to_le_bytes());
            // GEOMETRY channels — what the CAST resolved, before any shading
            // ran. `depth` is hashed through `to_bits` because a missed ray is
            // `f32::INFINITY` and `NaN != NaN`: the bit pattern is stable
            // where the float comparison is not.
            geom.update(cell.depth.to_bits().to_le_bytes());
            geom.update(cell.side.to_le_bytes());
            geom.update((cell.surface as u32).to_le_bytes());
            geom.update(cell.material.to_le_bytes());
        }
    }
    Ok(Digests {
        shade: to_hex(&shade.finalize()),
        geom: to_hex(&geom.finalize()),
    })
}

/// A golden-hash store: one file mapping a curated level key to its recorded
/// [`Digests`] (serialised as `key = <shade> <geom>`).
///
/// Lines starting with `#` or blank are comments/ignored. This is the type
/// `check_corpus` reads at the start of a run and, in bless mode, rebuilds and
/// writes back — see [`GoldenStore::load`] and [`GoldenStore::save`].
#[derive(Default)]
pub struct GoldenStore {
    entries: BTreeMap<String, String>,
}

impl GoldenStore {
    /// Load a store from disk. Returns an empty store when the file does not
    /// exist (first-bless scenario).
    ///
    /// # Panics
    ///
    /// Panics if the file exists but cannot be read, or a non-comment,
    /// non-blank line is not `key = value` — both are authoring errors in a
    /// file this crate itself writes, never a runtime condition to recover
    /// from silently.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        if !path.exists() {
            return Self::default();
        }
        let raw = std::fs::read_to_string(path).expect("failed to read golden store");
        let mut entries = BTreeMap::new();
        for (lineno, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((k, v)) = trimmed.split_once('=') else {
                panic!(
                    "{}:{}: expected `key = <shade> <geom>`, got: {:?}",
                    path.display(),
                    lineno + 1,
                    line
                );
            };
            entries.insert(k.trim().to_string(), v.trim().to_string());
        }
        Self { entries }
    }

    /// Write this store back to disk with a header comment, creating parent
    /// directories as needed.
    ///
    /// # Panics
    ///
    /// Panics if the parent directory cannot be created or the file cannot be
    /// written — a bless run that silently fails to persist would be worse
    /// than one that stops loudly.
    pub fn save(&self, path: &Path, git_sha: &str) {
        use std::fmt::Write as _;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create the golden store's directory");
        }
        let mut out = format!(
            "# Golden framebuffer hashes — DO NOT edit by hand.\n\
             # Regenerate with TS_BLESS_GOLDEN=1 set on the owning `cargo test` invocation\n\
             # (see this store's tests/golden/README.md entry for the exact command).\n\
             # Generated: {} from commit {git_sha}\n\
             # Format: key = <shade-sha256> <geom-sha256>, two independent digests per level.\n\
             #   shade = row-major {{glyph,fg,bg}} cells (12 bytes/cell, u32 LE) — the LOOK.\n\
             #   geom  = row-major {{depth,side,surface,material}} (11 bytes/cell) — the CAST.\n\
             # A mismatch in shade alone is a shader change; a mismatch in both is a\n\
             # raycaster/map/camera change. The split is what makes a red run diagnosable.\n\n",
            chrono_now_or_unknown()
        );
        for (k, v) in &self.entries {
            // `String`'s `Write` impl never fails — `expect` documents that rather
            // than propagating an `io::Result` that can never actually be `Err`.
            writeln!(out, "{k} = {v}").expect("write to String never fails");
        }
        std::fs::write(path, out).expect("failed to write golden store");
    }

    /// The recorded digests for `key`, parsed via [`Digests::from_store`].
    fn get(&self, key: &str) -> Option<Digests> {
        self.entries.get(key).map(|v| Digests::from_store(v))
    }

    /// Every key on record — used to find orphans (a stored key that is no
    /// longer curated).
    fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Build a fresh store from `(key, Digests)` pairs — used in bless mode so
    /// a retired key cannot linger and a live key cannot be dropped.
    fn from_fresh(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            entries: pairs.into_iter().collect(),
        }
    }

    /// Number of recorded entries.
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Return the current date as a string, or "unknown" if unavailable. Using
/// only `std` — no chrono dep — so date is approximate (seconds since epoch →
/// YYYY-MM-DD); good enough for curation provenance.
fn chrono_now_or_unknown() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "unknown".to_string();
    };
    let secs = dur.as_secs();
    // Simple Julian-day conversion (good for 2001-2099).
    let days = secs / 86400;
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Return the HEAD git SHA (short), or "unknown" if git is unavailable.
fn head_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// What happened to one curated entry on this run. Exactly one variant is
/// recorded per curated key — that is what makes the accounting total.
enum Outcome {
    /// Rendered, compared against the stored golden, identical. The only
    /// non-failing outcome.
    Matched,
    /// Rendered and compared, but at least one of the two digests moved.
    Mismatch { expected: Digests, actual: Digests },
    /// Rendered fine, but there is no hash on record, so NOTHING was compared.
    NoGolden { actual: Digests },
    /// Could not be rendered at all — missing file or parse error.
    Unreadable { why: String },
}

/// How to read each failure category, appended to a failing assertion.
const CATEGORY_LEGEND: &str = "
Reading the categories:

  MISMATCH      - a real render change. If it is INTENTIONAL and approved, re-run
                  with TS_BLESS_GOLDEN=1 set and commit the updated store together
                  with the render change and a tests/golden/README.md note. If it
                  is UNEXPECTED, investigate the render diff before blessing.
  UNREADABLE    - the level file is missing or no longer parses, so NOTHING was
                  checked for it. That is a coverage hole, not a regression.
                  Restore or fix the level; do NOT bless around it.
  NO-GOLDEN     - curated but never recorded, so NOTHING was compared. Bless only
                  if this is a deliberate NEW anchor at a reviewed, stable revision.
  ORPHAN-GOLDEN - a level was dropped from the corpus and its coverage went with
                  it. Restore the entry, or retire the stale hash deliberately by
                  re-blessing.

Blessing is NEVER the way to clear UNREADABLE or NO-GOLDEN: it would record a hash
for whatever the tree happens to render right now and call the hole closed.
";

/// Run the golden-framebuffer check for one corpus against one store file.
///
/// `store_path` is the store to read (and, under `TS_BLESS_GOLDEN=1`, to
/// rewrite). `corpus` is `(key, level_path)` pairs.
///
/// Parameterized over the store path and the curated level list so any number
/// of call sites (e.g. this crate's own engine corpus, and a host game's
/// campaign corpus) can share the render/hash/classify/report logic without
/// duplicating it.
///
/// # Panics
///
/// Asserts (and so panics) on any curated key that is not cleanly MATCHED —
/// MISMATCH, UNREADABLE, NO-GOLDEN or ORPHAN-GOLDEN — and on internal
/// accounting inconsistencies that would mean the suite is failing open. Call
/// this directly from a `#[test]` function; it IS the test body.
pub fn check_corpus(store_path: &Path, corpus: &[(&str, PathBuf)]) {
    // `TS_BLESS_GOLDEN=1` → regenerate (write) instead of assert.
    let bless = std::env::var_os("TS_BLESS_GOLDEN").is_some();

    let stored = GoldenStore::load(store_path);
    let git_sha = head_sha();

    // ── Precondition: the curated keys must be unique ─────────────────────
    // Two entries sharing a key would collapse into one store slot and make
    // the per-key accounting below silently under-count.
    {
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        for (key, _) in corpus {
            *seen.entry(*key).or_insert(0) += 1;
        }
        let dupes: Vec<String> = seen
            .iter()
            .filter(|(_, n)| **n > 1)
            .map(|(k, n)| format!("{k} (x{n})"))
            .collect();
        assert!(
            dupes.is_empty(),
            "[golden] corpus contains duplicate key(s): {}\n\
             Each key maps to one line in {}, duplicates make the coverage\n\
             accounting lie about how many levels were checked.",
            dupes.join(", "),
            store_path.display()
        );
    }

    // ── Render every curated level and classify the result ─────────────────
    let mut outcomes: Vec<(&str, Outcome)> = Vec::with_capacity(corpus.len());
    for (key, path) in corpus {
        let outcome = match render_level_hash(path) {
            Err(why) => Outcome::Unreadable { why },
            Ok(actual) => match stored.get(key) {
                None => Outcome::NoGolden { actual },
                Some(expected) if expected == actual => Outcome::Matched,
                Some(expected) => Outcome::Mismatch { expected, actual },
            },
        };
        outcomes.push((key, outcome));
    }

    // ── Bless mode ───────────────────────────────────────────────────────
    // Blessing rebuilds the store from THIS run's curated set, so a retired
    // key cannot linger and a live key cannot be dropped. A level that will
    // not render is still fatal here: writing a store missing its key would
    // quietly shrink coverage for every future run.
    if bless {
        let unreadable: Vec<String> = outcomes
            .iter()
            .filter_map(|(k, o)| match o {
                Outcome::Unreadable { why } => Some(format!("  {k}: {why}")),
                _ => None,
            })
            .collect();
        assert!(
            unreadable.is_empty(),
            "[golden] refusing to bless: {} curated level(s) could not be rendered.\n{}\n\n\
             Blessing now would write a store with no entry for them, silently\n\
             removing them from the checked set. Fix the level(s) first.",
            unreadable.len(),
            unreadable.join("\n")
        );

        let mut fresh: Vec<(String, String)> = Vec::with_capacity(outcomes.len());
        for (key, outcome) in &outcomes {
            let hash = match outcome {
                Outcome::Matched => stored
                    .get(key)
                    .expect("Matched implies a stored hash")
                    .to_store(),
                Outcome::Mismatch { actual, .. } | Outcome::NoGolden { actual } => {
                    actual.to_store()
                }
                Outcome::Unreadable { .. } => unreachable!("asserted empty above"),
            };
            println!("[golden] BLESS {key} = {hash}");
            fresh.push(((*key).to_string(), hash));
        }
        let fresh_store = GoldenStore::from_fresh(fresh);
        assert_eq!(
            fresh_store.len(),
            corpus.len(),
            "[golden] blessed store has {} entries for {} curated levels",
            fresh_store.len(),
            corpus.len()
        );
        fresh_store.save(store_path, &git_sha);
        println!(
            "[golden] Blessed {} level(s) at {git_sha} into {}.",
            fresh_store.len(),
            store_path.display()
        );
        return;
    }

    // ── Assert mode: every curated entry must be positively accounted for ──
    let mut matched = 0usize;
    let mut unreadable: Vec<String> = Vec::new();
    let mut no_golden: Vec<String> = Vec::new();
    let mut mismatched: Vec<String> = Vec::new();

    for (key, outcome) in &outcomes {
        match outcome {
            Outcome::Matched => {
                matched += 1;
            }
            Outcome::Unreadable { why } => {
                unreadable.push(format!("  UNREADABLE  {key}: {why}"));
            }
            Outcome::NoGolden { actual } => {
                no_golden.push(format!(
                    "  NO-GOLDEN   {key}: rendered {}, nothing on record",
                    actual.to_store()
                ));
            }
            Outcome::Mismatch { expected, actual } => {
                // Name WHICH channel moved — the whole point of the split digest.
                mismatched.push(format!(
                    "  MISMATCH    {key}: {}\n                expected shade {} geom {}\n                actual   shade {} geom {}",
                    expected.diff_note(actual),
                    expected.shade,
                    expected.geom,
                    actual.shade,
                    actual.geom
                ));
            }
        }
    }

    // A stored hash whose key is no longer curated means an entry was removed
    // from the corpus — the checked set shrank without the count of curated
    // entries revealing it. Caught here, not by the arithmetic below.
    let curated_keys: Vec<&str> = corpus.iter().map(|(k, _)| *k).collect();
    let orphans: Vec<String> = stored
        .keys()
        .filter(|k| !curated_keys.contains(k))
        .map(|k| format!("  ORPHAN-GOLDEN {k}: a hash is stored but the key is not curated"))
        .collect();

    println!(
        "[golden] {matched} matched, {} mismatched, {} unreadable, {} without a golden, {} orphan golden(s) — of {} curated ({})",
        mismatched.len(),
        unreadable.len(),
        no_golden.len(),
        orphans.len(),
        corpus.len(),
        store_path.display()
    );

    let mut problems: Vec<String> = Vec::new();
    problems.extend(unreadable);
    problems.extend(no_golden);
    problems.extend(mismatched);
    problems.extend(orphans);

    assert!(
        problems.is_empty(),
        "[golden] {} problem(s) — the golden framebuffer suite did NOT fully verify:\n{}\n\n\
         Each curated level's rendered {COLS}×{ROWS} grid is hashed TWICE — (glyph,fg,bg)\n\
         as `shade`, (depth,side,surface,material) as `geom` — and both are compared to\n\
         {}. Each MISMATCH line names which channel moved.\n{CATEGORY_LEGEND}",
        problems.len(),
        problems.join("\n"),
        store_path.display()
    );

    // Backstop. If every category above is empty then every curated entry
    // must have matched; if this ever fires, some path found a new way to not
    // check a level and the suite is failing open again.
    assert_eq!(
        matched,
        corpus.len(),
        "[golden] accounting mismatch: {matched} level(s) verified out of {} curated,\n\
         yet no problem was reported. A curated entry escaped classification —\n\
         the suite is failing open. Fix the accounting before trusting a green run.",
        corpus.len()
    );
}
