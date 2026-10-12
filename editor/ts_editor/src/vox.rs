//! VOX / "box model" import & placement.
//!
//! Places MagicaVoxel `.vox` models into the level as `[vox]` stamps. A stamp is
//! a [`re_level::LevelVox`] — a file path + an `(off_x,off_y,off_z)` anchor + a
//! surface [`re_level::VoxMaterial`] + an integer downscale `scale`. On reload
//! `re_level::build_map` reads the `.vox`, downscales, offsets, and stamps every
//! solid voxel as the chosen surface; so a placed model survives save/reload via
//! the `[vox]` section (the `re_voximport::write_vox` writer).
//!
//! ## Why a map REBUILD (and how painted geometry is preserved)
//! The editor's `state.map` is the live painted grid; vox geometry only appears
//! when `build_map` stamps it. So after every place/delete we rebuild the map
//! from the level. [`rebuild_map`] is a BASELINE-DIFF MERGE: it builds a
//! fresh procedural map from the level, then carries over every cell where the
//! live map diverged from the last procedural baseline
//! (`EditorState::preview_baseline`) — those divergences are exactly the user's
//! hand edits (paint, erase, 45° shapes, painted material ids), so painted
//! cells win any overlap with the fresh stamps (documented, deliberate) while
//! newly placed/deleted entities appear/disappear correctly. (The previous
//! capture-the-whole-map-into-`[cells]` mechanism made the dense capture
//! authoritative over EVERYTHING, wiping every freshly-stamped vox/node/decal
//! cell — measured: a placed `.vox`'s 1000 cells → 0 after that rebuild.) The
//! stamped vox voxels show in BOTH the top-down grid (they are real `Map` cells
//! now) and the 3D preview (which raycasts `state.map`).
//!
//! Pure functions over [`EditorState`] — no I/O beyond reading the `.vox` files
//! that `build_map` already reads, no terminal. Validate-before-mutate: an
//! out-of-range selection or an empty file list is a defined no-op.
//!
//! ## Keys (documented in `chrome.rs` help + `main.rs`/`edit.rs` routing)
//! * `v` — enter vox mode / cycle the selected `.vox` file forward.
//! * `c` — cycle the surface material (solid → trace → matrix).
//! * `+` / `-` — adjust the integer scale (clamped `>= 1`).
//! * `space` — place the selected model at the cursor (or re-select an existing
//!   stamp's anchor for editing).
//! * `x` — delete the vox stamp nearest the cursor.
use crate::state::EditorState;
use re_level::VoxMaterial;

/// Scan for selectable `.vox` files. The shipped levels reference their models
/// as `levels/<name>.vox` (read relative to the process cwd), so we scan the
/// `levels/` directory and return paths in that same `levels/<name>.vox` form —
/// a stamp placed here resolves on reload exactly like a hand-authored `[vox]`.
///
/// Deterministic: the returned list is sorted. An unreadable/missing directory
/// yields an empty list (the editor still works; the user just has nothing to
/// cycle — placing is a no-op until a model exists). The relative path keeps the
/// scan engine-pure (no absolute machine paths leak into a saved level).
pub fn scan_vox_files() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir("levels") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("vox") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    out.push(format!("levels/{name}"));
                }
            }
        }
    }
    out.sort();
    out
}

/// The currently-selected `.vox` file path, or `None` when none were found.
pub fn selected_file(state: &EditorState) -> Option<&str> {
    if state.vox_files.is_empty() {
        return None;
    }
    let i = state.vox_file_sel.min(state.vox_files.len() - 1);
    Some(state.vox_files[i].as_str())
}

/// Enter vox mode (if off) or cycle the selected `.vox` file FORWARD (if on),
/// wrapping. Entering vox mode leaves the object selector + material mode (the
/// modes are mutually exclusive) and clears any stale stamp selection. A no-op
/// cycle (with a hint) when no `.vox` files were found.
pub fn cycle_vox(state: &mut EditorState) {
    state.place = None;
    state.material_mode = false;
    if state.vox_files.is_empty() {
        state.vox_mode = true;
        state.selected_vox = None;
        state.status = "no .vox files in levels/".into();
        return;
    }
    if !state.vox_mode {
        // First entry: turn the mode on, keep the current selection.
        state.vox_mode = true;
        state.vox_file_sel = state.vox_file_sel.min(state.vox_files.len() - 1);
    } else {
        state.vox_file_sel = (state.vox_file_sel + 1) % state.vox_files.len();
    }
    state.selected_vox = None;
    state.status = vox_summary(state);
}

/// Cycle the surface material a vox stamp renders as (solid → trace → matrix →
/// solid). Targets the SELECTED stamp when one is selected (re-stamping it), else
/// the next-placement default. Marks dirty + rebuilds when an existing stamp
/// changed.
pub fn cycle_material(state: &mut EditorState) {
    let next = match state.vox_material {
        VoxMaterial::Solid => VoxMaterial::Trace,
        VoxMaterial::Trace => VoxMaterial::Matrix,
        VoxMaterial::Matrix => VoxMaterial::Solid,
    };
    state.vox_material = next;
    if let Some(idx) = selected_index(state) {
        state.level.voxes[idx].material = next;
        rebuild_map(state);
        state.mark_dirty();
        state.status = format!("vox material {}", next.as_str());
    } else {
        state.status = format!("vox material {} (next)", next.as_str());
    }
}

/// Adjust the integer scale by `delta`, clamped `>= 1`. Targets the SELECTED
/// stamp when one is selected (re-stamping it), else the next-placement default.
pub fn adjust_scale(state: &mut EditorState, delta: i32) {
    state.vox_scale = (state.vox_scale + delta).max(1);
    if let Some(idx) = selected_index(state) {
        state.level.voxes[idx].scale = state.vox_scale;
        rebuild_map(state);
        state.mark_dirty();
        state.status = format!("vox scale {}", state.vox_scale);
    } else {
        state.status = format!("vox scale {} (next)", state.vox_scale);
    }
}

/// Place the selected `.vox` model at the cursor — append a [`re_level::LevelVox`]
/// anchored at `(cursor.x, cursor.y, z)` with the current material + scale, then
/// rebuild the map so the stamped geometry appears in the top-down + preview.
///
/// Re-placing on an EXISTING stamp's anchor SELECTS it for editing (so `c`/`+`/`-`
/// retarget it) rather than pushing a duplicate. A no-op (with a hint) when no
/// `.vox` files are available.
pub fn place_at_cursor(state: &mut EditorState) {
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    // Re-place on an existing stamp anchored here → select, do not duplicate.
    if let Some(idx) = vox_at(state, x, y, z) {
        state.selected_vox = Some(idx);
        // Mirror the selected stamp's params into the live edit defaults so the
        // panel + subsequent c/+/- act on what the user sees.
        state.vox_material = state.level.voxes[idx].material;
        state.vox_scale = state.level.voxes[idx].scale.max(1);
        state.status = format!("selected vox '{}'", state.level.voxes[idx].name);
        return;
    }
    let Some(file) = selected_file(state).map(str::to_string) else {
        state.status = "no .vox to place (none in levels/)".into();
        return;
    };
    let name = unique_vox_name(state);
    state.level.voxes.push(re_level::LevelVox {
        name: name.clone(),
        file,
        off_x: x,
        off_y: y,
        off_z: z,
        material: state.vox_material,
        scale: state.vox_scale.max(1),
    });
    state.selected_vox = Some(state.level.voxes.len() - 1);
    rebuild_map(state);
    state.mark_dirty();
    state.status = format!("placed vox '{name}' @ {x},{y},{z}");
}

/// Delete the vox stamp NEAREST the cursor (on any layer — vox stamps are 3D and
/// a model's anchor may sit on a different z than the painting layer). A no-op
/// (with a hint) when the level has no vox stamps. Rebuilds the map so the
/// stamped geometry disappears from the views.
pub fn delete_at_cursor(state: &mut EditorState) {
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    let Some(idx) = nearest_vox(state, x, y, z) else {
        state.status = "no vox stamps to delete".into();
        return;
    };
    let name = state.level.voxes[idx].name.clone();
    state.level.voxes.remove(idx);
    // HIERARCHY index fixup: drop links touching the removed vox and shift every
    // later vox link down one (the `Vec::remove` shifted them).
    state
        .level
        .hier_fixup_after_remove(re_level::HierKind::Vox, idx);
    // Fix the selection after the Vec shifts (never edit a shifted neighbour).
    state.selected_vox = match state.selected_vox {
        Some(s) if s == idx => None,
        Some(s) if s > idx => Some(s - 1),
        other => other,
    };
    rebuild_map(state);
    state.mark_dirty();
    state.status = format!("removed vox '{name}'");
}

/// Index of a vox stamp whose anchor is EXACTLY `(x,y,z)`, if any.
pub fn vox_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    state
        .level
        .voxes
        .iter()
        .position(|v| (v.off_x, v.off_y, v.off_z) == (x, y, z))
}

/// Index of the vox stamp whose anchor is nearest `(x,y,z)` (squared distance),
/// or `None` when there are no stamps. Ties resolve to the earliest index.
fn nearest_vox(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    state
        .level
        .voxes
        .iter()
        .enumerate()
        .min_by_key(|(_, v)| {
            let dx = (v.off_x - x) as i64;
            let dy = (v.off_y - y) as i64;
            let dz = (v.off_z - z) as i64;
            dx * dx + dy * dy + dz * dz
        })
        .map(|(i, _)| i)
}

/// The valid selected-stamp index, or `None` when nothing is selected / the
/// selection drifted out of range.
fn selected_index(state: &EditorState) -> Option<usize> {
    state.selected_vox.filter(|&i| i < state.level.voxes.len())
}

/// A unique `box<k>` name for a freshly-placed stamp (so the saved `[vox NAME]`
/// section names never collide — collisions would be confusing on reload).
fn unique_vox_name(state: &EditorState) -> String {
    let mut k = 0usize;
    loop {
        let candidate = format!("box{k}");
        if !state.level.voxes.iter().any(|v| v.name == candidate) {
            return candidate;
        }
        k += 1;
    }
}

/// Rebuild `state.map` from the level so vox stamps (and everything else) show in
/// the top-down + 3D preview, WITHOUT losing freely-painted geometry.
///
/// BASELINE-DIFF MERGE: build a FRESH map straight from the level
/// (`build_map` — geometry, objects, materials, overlay, atmosphere), then
/// carry the user's hand edits over it: every cell whose `(kind, side_data,
/// shape)` in the LIVE map diverges from the last procedural BASELINE
/// ([`EditorState::preview_baseline`]) is a hand edit and wins; every other
/// cell takes the fresh value, so newly placed entities/vox stamps appear and
/// deleted ones disappear. The fresh (pre-merge) map becomes the new baseline.
///
/// `build_map` (not `build_map_checked`) is used so a missing/bad `.vox` or
/// texture is a best-effort skip rather than a hard error mid-edit; the
/// save-time lvlcheck surfaces real problems.
///
/// `level.cells` is NEVER touched — a `[cells]`-bearing level's
/// original grid is stamped by `build_map` exactly as at load, and a
/// procedural level stays `cells == None`, so the conditional-baking decision
/// (and today's save path) see exactly the loaded provenance. Mutating it here
/// (the old mechanism) would have made every entity/material edit silently
/// flatten a procedural level on the next save.
pub fn rebuild_map(state: &mut EditorState) {
    // Build against the AS-LOADED `[cells]` provenance: the save path overwrites
    // `level.cells` with a live-map capture, and building against that stale
    // capture would freeze every procedural stamp into geometry. The swap is
    // restored immediately, so `level.cells` is observably untouched.
    let saved_cells = std::mem::replace(&mut state.level.cells, state.loaded_cells.clone());
    let fresh = state.level.build_map();
    state.level.cells = saved_cells;
    let mut merged = fresh.clone();
    let live = &state.map;
    let usable_baseline = state
        .preview_baseline
        .as_ref()
        .filter(|b| b.size() == live.size() && fresh.size() == live.size());
    if let Some(base) = usable_baseline {
        // Carry each hand-edited cell (differs from the baseline) over the
        // fresh stamp. O(cells) compares — microseconds at editor map sizes.
        let (sx, sy, sz) = fresh.size();
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                    let kind = live.cell_at(xi, yi, zi);
                    let sd = live.side_data_at(xi, yi, zi);
                    let shape = live.shape_at(xi, yi, zi);
                    if kind != base.cell_at(xi, yi, zi)
                        || sd != base.side_data_at(xi, yi, zi)
                        || shape != base.shape_at(xi, yi, zi)
                    {
                        merged.set_cell(x, y, z, kind, sd);
                        merged.set_shape(x, y, z, shape);
                    }
                }
            }
        }
    } else if fresh.size() == live.size() {
        // No usable baseline (defensive — the sites that replace the map keep
        // it in step): legacy behaviour, the live grid is authoritative
        // wholesale. Fresh stamps surface on the NEXT rebuild once the baseline
        // set below has landed.
        let (sx, sy, sz) = fresh.size();
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                    merged.set_cell(
                        x,
                        y,
                        z,
                        live.cell_at(xi, yi, zi),
                        live.side_data_at(xi, yi, zi),
                    );
                    merged.set_shape(x, y, z, live.shape_at(xi, yi, zi));
                }
            }
        }
    }
    // else: the level size changed without a map remap (never happens via the
    // editor's resize path) — keep the fresh map wholesale.
    state.preview_baseline = Some(fresh);
    state.map = merged;
}

/// Frames the DEBOUNCED preview rebuild waits after the last edit before firing
/// (~150 ms at the editor's 60 Hz loop) — long enough to coalesce a fast paint
/// stroke into one rebuild, short enough that the preview still feels live.
pub const REBUILD_DEBOUNCE_FRAMES: u8 = 9;

/// Does this level reference ASSETS whose decode `build_map` re-runs from disk
/// on every call — `[texture]`/`[sky]` image decodes, `[vox]` file reads,
/// `[inset]` child-scene loads? Measured (debug build): a rebuild
/// on an asset-bearing level costs ~40-50 ms per call vs
/// ~31 µs asset-free (09-code-wall). The debounced auto-rebuild therefore SKIPS asset-bearing
/// levels (explicit `Shift+R` rebuild instead); asset-free levels — most of the
/// shipped set — rebuild automatically.
pub const fn level_has_assets(level: &re_level::Level) -> bool {
    !level.textures.is_empty()
        || !level.voxes.is_empty()
        || !level.overlay.insets.is_empty()
        || level.sky.cylinder.is_some()
        || level.sky.cap_top.is_some()
        || level.sky.cap_bottom.is_some()
}

/// Tick the DEBOUNCED preview rebuild countdown one frame (the 60 Hz loop calls
/// this once per frame). Returns `true` when a rebuild actually ran this tick.
///
/// When the countdown expires:
/// * an ASSET-FREE level (the common case) rebuilds via [`rebuild_map`] — cheap
///   (µs), so the preview reflects entity/material edits within ~150 ms;
/// * an ASSET-BEARING level ([`level_has_assets`]) SKIPS the auto-rebuild —
///   `build_map` would re-decode images / re-read `.vox` files from disk
///   (40-50 ms, measured) on every debounce expiry — and hints the status line
///   toward the explicit `Shift+R` manual rebuild instead — an asset-bearing
///   level only rebuilds at load and on explicit request.
pub fn tick_preview_rebuild(state: &mut EditorState) -> bool {
    match state.preview_rebuild_delay {
        None => false,
        Some(0) => {
            state.preview_rebuild_delay = None;
            if level_has_assets(&state.level) {
                state.status =
                    "preview stale (level has image/vox assets) — Shift+R rebuilds".into();
                false
            } else {
                rebuild_map(state);
                true
            }
        }
        Some(n) => {
            state.preview_rebuild_delay = Some(n - 1);
            false
        }
    }
}

/// EXPLICIT preview rebuild (`Shift+R`): unconditionally rebuild the map from
/// the level, asset re-decode included. The manual escape hatch for
/// asset-bearing levels whose debounced auto-rebuild is skipped (see
/// [`tick_preview_rebuild`]); harmless (and instant) on asset-free levels.
/// Cancels any pending debounce — the rebuild it was waiting for just happened.
pub fn manual_rebuild(state: &mut EditorState) {
    state.preview_rebuild_delay = None;
    rebuild_map(state);
    state.status = "preview rebuilt".into();
}

/// A compact one-line summary of the vox-mode selection (panel / status line).
pub fn vox_summary(state: &EditorState) -> String {
    match selected_file(state) {
        None => "no .vox files (levels/)".to_string(),
        Some(f) => {
            // Show just the file stem to keep the line short.
            let stem = f.rsplit('/').next().unwrap_or(f);
            let sel = if state.selected_vox.is_some() {
                " [sel]"
            } else {
                ""
            };
            format!(
                "[{}/{}] {} {} x{}{}",
                state
                    .vox_file_sel
                    .min(state.vox_files.len().saturating_sub(1))
                    + 1,
                state.vox_files.len(),
                stem,
                state.vox_material.as_str(),
                state.vox_scale.max(1),
                sel,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CellKind;

    /// A state with a synthetic vox-file list (so tests don't depend on the
    /// on-disk `levels/` scan).
    fn state_with_files(files: &[&str]) -> EditorState {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.vox_files = files.iter().map(|f| f.to_string()).collect();
        s.vox_file_sel = 0;
        s.vox_mode = false;
        s
    }

    #[test]
    fn scan_returns_sorted_relative_paths() {
        // Run against whatever `levels/` is visible from the test cwd (the crate
        // dir under `cargo test`). We do NOT mutate the process cwd — that is
        // global and would race the relative-path reads in other tests. The
        // contract holds regardless of how many files are found: every result is
        // a `levels/<name>.vox` relative path, and the list is sorted.
        let files = scan_vox_files();
        assert!(files.iter().all(|f| f.ends_with(".vox")), "only .vox paths");
        assert!(
            files.iter().all(|f| f.starts_with("levels/")),
            "relative levels/ paths (no absolute machine paths leak)"
        );
        let mut sorted = files.clone();
        sorted.sort();
        assert_eq!(files, sorted, "scan is deterministic / sorted");
    }

    #[test]
    fn cycle_enters_mode_then_advances_and_wraps() {
        let mut s = state_with_files(&["levels/a.vox", "levels/b.vox"]);
        // First press enters mode, keeps selection 0.
        cycle_vox(&mut s);
        assert!(s.vox_mode);
        assert_eq!(s.vox_file_sel, 0);
        // Next steps advance + wrap.
        cycle_vox(&mut s);
        assert_eq!(s.vox_file_sel, 1);
        cycle_vox(&mut s);
        assert_eq!(s.vox_file_sel, 0, "wraps");
        // Entering vox mode leaves the other modes.
        assert!(!s.material_mode);
        assert!(s.place.is_none());
    }

    #[test]
    fn cycle_with_no_files_is_a_hinted_noop() {
        let mut s = state_with_files(&[]);
        cycle_vox(&mut s);
        assert!(s.vox_mode);
        assert!(selected_file(&s).is_none());
    }

    #[test]
    fn place_appends_a_vox_at_the_cursor_and_marks_dirty() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        s.cursor = (3, 4);
        s.z = 1;
        place_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 1);
        let v = &s.level.voxes[0];
        assert_eq!(v.file, "levels/a.vox");
        assert_eq!((v.off_x, v.off_y, v.off_z), (3, 4, 1));
        assert_eq!(v.material, VoxMaterial::Solid);
        assert_eq!(v.scale, 1);
        assert_eq!(s.selected_vox, Some(0));
        assert!(s.dirty);
    }

    #[test]
    fn replace_on_existing_anchor_selects_not_duplicates() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        s.cursor = (2, 2);
        s.z = 0;
        place_at_cursor(&mut s);
        // A different cell → a second stamp.
        s.cursor = (5, 5);
        place_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 2);
        s.selected_vox = None;
        // Re-placing on the FIRST anchor selects index 0, no duplicate.
        s.cursor = (2, 2);
        place_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 2, "no duplicate on re-place");
        assert_eq!(s.selected_vox, Some(0));
    }

    #[test]
    fn place_with_no_files_is_a_noop() {
        let mut s = state_with_files(&[]);
        s.vox_mode = true;
        place_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 0);
        assert!(!s.dirty);
    }

    #[test]
    fn delete_removes_nearest_and_fixes_selection() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        // Two stamps at distinct anchors.
        s.cursor = (1, 1);
        s.z = 0;
        place_at_cursor(&mut s);
        s.cursor = (6, 6);
        place_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 2);
        s.selected_vox = Some(1);
        // Cursor near the FIRST stamp → it is deleted; selection follows.
        s.cursor = (1, 1);
        delete_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 1);
        assert_eq!(s.selected_vox, Some(0));
        assert_eq!((s.level.voxes[0].off_x, s.level.voxes[0].off_y), (6, 6));
    }

    #[test]
    fn delete_with_no_stamps_is_a_noop() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        delete_at_cursor(&mut s);
        assert_eq!(s.level.voxes.len(), 0);
        assert!(!s.dirty);
    }

    #[test]
    fn cycle_material_advances_solid_trace_matrix() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        assert_eq!(s.vox_material, VoxMaterial::Solid);
        cycle_material(&mut s);
        assert_eq!(s.vox_material, VoxMaterial::Trace);
        cycle_material(&mut s);
        assert_eq!(s.vox_material, VoxMaterial::Matrix);
        cycle_material(&mut s);
        assert_eq!(s.vox_material, VoxMaterial::Solid, "wraps");
    }

    #[test]
    fn cycle_material_retargets_selected_stamp() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        s.cursor = (2, 2);
        place_at_cursor(&mut s);
        assert_eq!(s.selected_vox, Some(0));
        cycle_material(&mut s); // Solid → Trace
        assert_eq!(s.level.voxes[0].material, VoxMaterial::Trace);
        assert!(s.dirty);
    }

    #[test]
    fn adjust_scale_steps_and_clamps_at_one() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        assert_eq!(s.vox_scale, 1);
        adjust_scale(&mut s, 1);
        assert_eq!(s.vox_scale, 2);
        adjust_scale(&mut s, -1);
        adjust_scale(&mut s, -1); // clamps at 1
        assert_eq!(s.vox_scale, 1);
    }

    #[test]
    fn adjust_scale_retargets_selected_stamp() {
        let mut s = state_with_files(&["levels/a.vox"]);
        s.vox_mode = true;
        s.cursor = (3, 3);
        place_at_cursor(&mut s);
        adjust_scale(&mut s, 2); // 1 → 3
        assert_eq!(s.level.voxes[0].scale, 3);
        assert!(s.dirty);
    }

    /// `rebuild_map` preserves freely-painted geometry (captured into `[cells]`
    /// and re-stamped LAST). A painted wall on a cell survives a rebuild.
    #[test]
    fn rebuild_preserves_painted_cells() {
        let mut s = EditorState::new_empty(6, 6, 4);
        s.map.set(2, 3, 1, CellKind::WallSolid);
        s.map.set(0, 0, 0, CellKind::Floor);
        rebuild_map(&mut s);
        assert_eq!(
            s.map.cell_at(2, 3, 1),
            CellKind::WallSolid,
            "painted wall kept"
        );
        assert_eq!(
            s.map.cell_at(0, 0, 0),
            CellKind::Floor,
            "painted floor kept"
        );
    }

    /// GUARD: `rebuild_map` must NOT mutate `level.cells`. A
    /// procedural document (cells = None) stays None after any number of
    /// rebuilds — otherwise the conditional `[cells]` baking decision would always
    /// bake after the first entity/material edit, flattening procedural levels
    /// on save. A `[cells]`-bearing document keeps its ORIGINAL grid, not the
    /// scratch capture.
    #[test]
    fn rebuild_does_not_mutate_level_cells() {
        // Procedural: cells None before and after.
        let mut s = EditorState::new_empty(6, 6, 4);
        assert!(s.level.cells.is_none(), "fresh document is procedural");
        s.map.set(2, 2, 0, CellKind::WallSolid);
        rebuild_map(&mut s);
        rebuild_map(&mut s);
        assert!(
            s.level.cells.is_none(),
            "rebuild must not set level.cells on a procedural document"
        );

        // Cells-bearing: the ORIGINAL grid survives (not the scratch capture).
        let mut s2 = EditorState::new_empty(4, 4, 2);
        let original = re_level::CellGrid::from_map(&s2.map);
        s2.level.cells = Some(original);
        let before = format!("{:?}", s2.level.cells);
        s2.map.set(1, 1, 0, CellKind::WallMatrix); // diverge map from the grid
        rebuild_map(&mut s2);
        assert_eq!(
            format!("{:?}", s2.level.cells),
            before,
            "rebuild restores the original level.cells, not the capture"
        );
    }

    /// DEBOUNCE: scheduling arms the countdown; ticking it down runs the
    /// rebuild exactly once (asset-free level) and the entity edit lands in the
    /// map. Uses a NODE — `build_map` stamps nodes as `Object` cells, so the
    /// rebuilt map is observable (this is exactly the preview-fidelity gap:
    /// `entity::place_at_cursor` never touches `state.map` itself).
    #[test]
    fn debounced_rebuild_fires_once_after_countdown() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let _ = s.level.push_node(5, 5, 1);
        assert_eq!(s.map.cell_at(5, 5, 1), CellKind::Empty, "not stamped yet");

        s.schedule_preview_rebuild();
        assert_eq!(s.preview_rebuild_delay, Some(REBUILD_DEBOUNCE_FRAMES));
        let mut rebuilds = 0;
        for _ in 0..=REBUILD_DEBOUNCE_FRAMES {
            if tick_preview_rebuild(&mut s) {
                rebuilds += 1;
            }
        }
        assert_eq!(rebuilds, 1, "exactly one rebuild after the countdown");
        assert!(s.preview_rebuild_delay.is_none(), "countdown cleared");
        assert_eq!(
            s.map.cell_at(5, 5, 1),
            CellKind::Object,
            "the node placement reached the preview map"
        );
        // Idle ticks after that are no-ops.
        assert!(!tick_preview_rebuild(&mut s));
    }

    /// BASELINE MERGE: painted edits AND fresh entity stamps coexist after
    /// a rebuild — the old dense-capture mechanism could only keep one of the
    /// two (it wiped every fresh stamp). Also covers DELETION: removing the
    /// node and rebuilding clears its stamp.
    #[test]
    fn rebuild_merges_paint_and_fresh_stamps() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.map.set(2, 2, 0, CellKind::WallMatrix); // hand paint
        let _ = s.level.push_node(6, 6, 1); // entity only in the level
        rebuild_map(&mut s);
        assert_eq!(s.map.cell_at(2, 2, 0), CellKind::WallMatrix, "paint kept");
        assert_eq!(
            s.map.cell_at(6, 6, 1),
            CellKind::Object,
            "node stamp appears"
        );

        // Delete the node → its stamp disappears on the next rebuild, while the
        // paint still survives.
        s.level.nodes.clear();
        rebuild_map(&mut s);
        assert_eq!(
            s.map.cell_at(6, 6, 1),
            CellKind::Empty,
            "deleted node unstamped"
        );
        assert_eq!(
            s.map.cell_at(2, 2, 0),
            CellKind::WallMatrix,
            "paint still kept"
        );
    }

    /// BASELINE MERGE: a user ERASE of a procedurally-stamped cell is a
    /// hand edit too — it survives subsequent rebuilds (erase wins over the
    /// fresh stamp, mirroring painted-wins).
    #[test]
    fn rebuild_keeps_user_erase_of_a_stamped_cell() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let _ = s.level.push_node(4, 4, 1);
        rebuild_map(&mut s);
        assert_eq!(s.map.cell_at(4, 4, 1), CellKind::Object, "stamped");
        // User erases the stamped cell (the x key path writes Empty).
        s.map.set(4, 4, 1, CellKind::Empty);
        rebuild_map(&mut s);
        assert_eq!(
            s.map.cell_at(4, 4, 1),
            CellKind::Empty,
            "user erase wins over the fresh stamp"
        );
    }

    /// WIRING: the entity edit choke points (place/delete/adjust/cycle in
    /// `entity.rs`) and the material-def edit paths (`material.rs`) arm the
    /// debounce flag themselves — no caller has to remember to.
    #[test]
    fn entity_and_material_edit_paths_schedule_the_rebuild() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(crate::state::EntityKind::Node);
        s.cursor = (3, 3);
        crate::entity::place_at_cursor(&mut s);
        assert!(s.preview_rebuild_delay.is_some(), "entity place schedules");

        s.preview_rebuild_delay = None;
        crate::entity::adjust_property(&mut s, 1); // node pos +1
        assert!(
            s.preview_rebuild_delay.is_some(),
            "entity field edit schedules"
        );

        s.preview_rebuild_delay = None;
        crate::material::new_material(&mut s);
        assert!(s.preview_rebuild_delay.is_some(), "new material schedules");

        s.preview_rebuild_delay = None;
        crate::material::adjust_field(&mut s, 1); // fg palette step
        assert!(
            s.preview_rebuild_delay.is_some(),
            "material def edit schedules"
        );
    }

    /// REGRESSION GUARD: a SAVE (`capture_and_serialize` overwrites
    /// `level.cells` with a live capture) must not poison later rebuilds —
    /// rebuild_map builds against the AS-LOADED cells provenance, so an entity
    /// placed after a save still appears.
    #[test]
    fn rebuild_still_shows_new_entities_after_a_save_capture() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let _ = crate::io::capture_and_serialize(&mut s); // save half: sets level.cells
        assert!(s.level.cells.is_some(), "save capture happened");
        let _ = s.level.push_node(6, 2, 1);
        rebuild_map(&mut s);
        assert_eq!(
            s.map.cell_at(6, 2, 1),
            CellKind::Object,
            "node placed AFTER a save still reaches the preview"
        );
        assert!(
            s.level.cells.is_some(),
            "the save capture itself is left in place (save behavior unchanged)"
        );
    }

    /// RE-ARM: an edit during the countdown restarts it (one rebuild per
    /// edit burst, never per keystroke).
    #[test]
    fn scheduling_again_rearms_the_countdown() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.schedule_preview_rebuild();
        for _ in 0..3 {
            assert!(!tick_preview_rebuild(&mut s));
        }
        assert_eq!(s.preview_rebuild_delay, Some(REBUILD_DEBOUNCE_FRAMES - 3));
        s.schedule_preview_rebuild(); // burst continues
        assert_eq!(
            s.preview_rebuild_delay,
            Some(REBUILD_DEBOUNCE_FRAMES),
            "re-armed"
        );
    }

    /// ASSET GATE: a level with `[texture]`/`[vox]`/`[inset]`/`[sky]`
    /// assets does NOT auto-rebuild when the debounce expires — `build_map`
    /// would re-decode/re-read from disk (40-50 ms measured) on every
    /// expiry. The status hints at the explicit `Shift+R` rebuild instead.
    #[test]
    fn asset_bearing_level_skips_auto_rebuild() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.level.textures.push(re_level::LevelTexture {
            name: "tx".into(),
            file: "assets/texture/does-not-matter.png".into(),
            max_size: None,
            height: None,
            ..Default::default()
        });
        assert!(level_has_assets(&s.level));
        s.preview_rebuild_delay = Some(0); // countdown expiring this tick
        let rebuilt = tick_preview_rebuild(&mut s);
        assert!(!rebuilt, "asset-bearing level must not auto-rebuild");
        assert!(s.preview_rebuild_delay.is_none(), "countdown consumed");
        assert!(
            s.status.contains("Shift+R"),
            "status hints the manual rebuild: {}",
            s.status
        );
    }

    /// `level_has_assets` is false for a plain procedural/painted level, true
    /// for each asset-bearing section kind.
    #[test]
    fn level_has_assets_classification() {
        let s = EditorState::new_empty(4, 4, 2);
        assert!(!level_has_assets(&s.level), "plain level is asset-free");

        let mut with_vox = s.level.clone();
        with_vox.voxes.push(re_level::LevelVox {
            name: "b".into(),
            file: "levels/x.vox".into(),
            off_x: 0,
            off_y: 0,
            off_z: 0,
            material: VoxMaterial::Solid,
            scale: 1,
        });
        assert!(level_has_assets(&with_vox), "[vox] is an asset");

        let mut with_sky = s.level.clone();
        with_sky.sky.cylinder = Some("assets/sky.png".into());
        assert!(level_has_assets(&with_sky), "[sky] is an asset");

        let mut with_inset = s.level;
        with_inset
            .overlay
            .insets
            .push(re_level::OverlayInset::default());
        assert!(
            level_has_assets(&with_inset),
            "[inset] child scene is an asset"
        );
    }

    /// `Shift+R` manual rebuild works even on an asset-bearing level and cancels
    /// a pending debounce.
    #[test]
    fn manual_rebuild_is_unconditional_and_cancels_debounce() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.level.textures.push(re_level::LevelTexture {
            name: "tx".into(),
            file: "no-such-file.png".into(), // decode failure is a soft skip
            max_size: None,
            height: None,
            ..Default::default()
        });
        let _ = s.level.push_node(3, 3, 1);
        s.schedule_preview_rebuild();
        manual_rebuild(&mut s);
        assert!(
            s.preview_rebuild_delay.is_none(),
            "pending debounce cancelled"
        );
        assert_eq!(s.map.cell_at(3, 3, 1), CellKind::Object, "map rebuilt");
        assert!(
            s.level.cells.is_none(),
            "the level.cells guard holds for the manual path too"
        );
    }
}
