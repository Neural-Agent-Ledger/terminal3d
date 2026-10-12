//! NEW-LEVEL wizard: a tiny in-TUI form to create a fresh level.
//!
//! Fields: size X, Y, Z, a name, and a "perimeter shell" toggle. The form is a
//! small centered panel. Keys while the wizard is open (handled in
//! `edit::apply_key` when `state.wizard.is_some()`):
//! * `Tab`        → focus the next field (X → Y → Z → Name → Shell → X …)
//! * `+` / `-`    → adjust the focused numeric field (size 1..=512); on the Name
//!   field, cycle a small preset name list; on Shell, toggle.
//! * `Enter`      → CREATE: replace the document with a fresh empty level of that
//!   size (optionally with a 1-cell perimeter wall shell), clearing undo history.
//!   `dirty` is set (the new doc is unsaved).
//! * `Esc`        → CANCEL: close the form, leave the current level untouched.
//!
//! Pure: building the level and applying it touch only `EditorState` data — no
//! I/O, no terminal.
use crate::state::EditorState;
use re_world::CellKind;

/// Which field the wizard's `+`/`-`/`c` keys edit. `Tab` steps through these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WizardField {
    SizeX,
    SizeY,
    SizeZ,
    Name,
    Shell,
}

impl WizardField {
    /// Cycle X → Y → Z → Name → Shell → X.
    pub fn next(self) -> WizardField {
        match self {
            WizardField::SizeX => WizardField::SizeY,
            WizardField::SizeY => WizardField::SizeZ,
            WizardField::SizeZ => WizardField::Name,
            WizardField::Name => WizardField::Shell,
            WizardField::Shell => WizardField::SizeX,
        }
    }
}

/// Minimum / maximum authorable dimension per axis.
pub const MIN_DIM: usize = 1;
pub const MAX_DIM: usize = 512;

/// A handful of preset names the Name field cycles through with `+`/`-`. Free
/// TEXT entry is also supported now (type printable chars into `name_buf` while
/// the Name field is focused); the presets remain a quick starting point.
pub const NAME_PRESETS: [&str; 6] = ["untitled", "room", "hall", "vault", "maze", "arena"];

/// Live state of the open new-level form.
#[derive(Clone)]
pub struct Wizard {
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,
    /// The preset cursor `+`/`-` step through when the Name field is focused; it
    /// OVERWRITES `name_buf` with the chosen preset so cycling is still useful.
    pub name_idx: usize,
    /// The editable name buffer. Free-text typing (printable chars) appends here
    /// while the Name field is focused; `[Clr]` clears it. Backspace is not
    /// available (re_present's `EditorKeyCode` has no Backspace — see the report).
    pub name_buf: String,
    pub shell: bool,
    pub focus: WizardField,
}

impl Default for Wizard {
    /// A sensible starting form: a 16x16x4 shelled room named "untitled".
    fn default() -> Self {
        Wizard {
            sx: 16,
            sy: 16,
            sz: 4,
            name_idx: 0,
            name_buf: NAME_PRESETS[0].to_string(),
            shell: true,
            focus: WizardField::SizeX,
        }
    }
}

impl Wizard {
    /// The current name: the editable buffer (free-text). Falls back to the first
    /// preset when the buffer is empty so a created level always has a name.
    pub fn name(&self) -> &str {
        if self.name_buf.is_empty() {
            NAME_PRESETS[0]
        } else {
            &self.name_buf
        }
    }

    /// Append a printable character to the name buffer (free-text entry). Only
    /// called while the Name field is focused.
    pub fn name_push(&mut self, c: char) {
        self.name_buf.push(c);
    }

    /// Clear the name buffer (the `[Clr]` button).
    pub fn name_clear(&mut self) {
        self.name_buf.clear();
    }

    /// Backspace: pop the last character of the name buffer (free-text editing).
    /// No-op when the buffer is empty. Only called while the Name field is focused.
    pub fn name_pop(&mut self) {
        self.name_buf.pop();
    }

    /// Adjust the focused field by `delta` (`+1`/`-1`). Sizes clamp to
    /// `MIN_DIM..=MAX_DIM`; Name cycles the presets (overwriting the buffer);
    /// Shell toggles.
    pub fn adjust(&mut self, delta: i32) {
        let step = |v: usize| -> usize {
            ((v as i32 + delta).clamp(MIN_DIM as i32, MAX_DIM as i32)) as usize
        };
        match self.focus {
            WizardField::SizeX => self.sx = step(self.sx),
            WizardField::SizeY => self.sy = step(self.sy),
            WizardField::SizeZ => self.sz = step(self.sz),
            WizardField::Name => {
                let n = NAME_PRESETS.len() as i32;
                self.name_idx = (self.name_idx as i32 + delta).rem_euclid(n) as usize;
                self.name_buf = NAME_PRESETS[self.name_idx].to_string();
            }
            WizardField::Shell => self.shell = !self.shell,
        }
    }

    /// Advance focus to the next field.
    pub fn focus_next(&mut self) {
        self.focus = self.focus.next();
    }
}

/// CREATE the level described by `wiz` into `state`, replacing the current map +
/// level, clearing undo/redo history, resetting cursor/z, and marking dirty. With
/// `shell` on, a 1-cell `WallSolid` perimeter is laid on every z-layer (floor /
/// ceiling included) so the fresh room is enclosed.
pub fn apply(state: &mut EditorState, wiz: &Wizard) {
    let (sx, sy, sz) = (
        wiz.sx.clamp(MIN_DIM, MAX_DIM),
        wiz.sy.clamp(MIN_DIM, MAX_DIM),
        wiz.sz.clamp(MIN_DIM, MAX_DIM),
    );
    // Build a fresh empty document and adopt its content fields. We do NOT replace
    // the whole `EditorState` (it carries the scanned vox list, view mode, etc.);
    // just the edited content + history.
    let fresh = EditorState::new_empty(sx, sy, sz);
    state.map = fresh.map;
    // Adopt the fresh document's procedural baseline too (the shell laid
    // below is a hand edit on top of it, preserved by the rebuild diff-merge).
    state.preview_baseline = fresh.preview_baseline;
    state.preview_rebuild_delay = None;
    state.loaded_cells = None; // a new document has no [cells] provenance
    state.level = fresh.level;
    // A NEW document starts with the canonical vocabularies only — the previously
    // opened file's authored glyphs/presets are not part of it (`crate::vocab`).
    state.vocab = crate::vocab::Vocab::default();
    state.level.name = wiz.name().to_string();
    state.level.title = wiz.name().to_string();
    state.cam = fresh.cam;
    state.cursor = (0, 0);
    state.z = 0;
    state.pan = (0, 0);
    // A NEW level must NOT inherit the previously-opened file's
    // path — otherwise the first `s` would silently overwrite that file. Clear it
    // so the next save routes through Save-As to name the fresh document.
    state.path = None;
    state.selected_node = None;
    state.selected_emitter = None;
    state.selected_trigger = None;
    state.selected_vox = None;
    state.fill_anchor = None;
    // A NEW document has no load provenance: drop the previous file's preserved
    // comments/raw sections (they must not re-emit into this fresh level on
    // save) and its `[cells]`-at-load flag (1.4-prep — wizard/new = false).
    state.reset_load_provenance();
    state.history.clear();

    if wiz.shell {
        lay_shell(state, sx, sy, sz);
    }
    state.dirty = true;
    state.status = format!("new level {sx}x{sy}x{sz} \"{}\"", wiz.name());
}

/// Lay a 1-cell `WallSolid` perimeter shell: every cell on the outer faces of the
/// `sx*sy*sz` box (the six bounding planes) becomes solid, enclosing the room.
fn lay_shell(state: &mut EditorState, sx: usize, sy: usize, sz: usize) {
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let on_face =
                    x == 0 || y == 0 || z == 0 || x == sx - 1 || y == sy - 1 || z == sz - 1;
                if on_face {
                    state.map.set(x, y, z, CellKind::WallSolid);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wizard_creates_level_of_requested_size() {
        let mut s = EditorState::new_empty(16, 16, 4);
        let wiz = Wizard {
            sx: 10,
            sy: 8,
            sz: 3,
            shell: false,
            ..Wizard::default()
        };
        apply(&mut s, &wiz);
        assert_eq!(s.map.size(), (10, 8, 3));
        assert_eq!(s.level.size, (10, 8, 3));
        assert!(s.dirty);
        // No shell ⇒ a corner cell is empty.
        assert_eq!(s.map.cell_at(0, 0, 0), CellKind::Empty);
    }

    #[test]
    fn wizard_lays_perimeter_shell() {
        let mut s = EditorState::new_empty(4, 4, 2);
        let wiz = Wizard {
            sx: 5,
            sy: 5,
            sz: 2,
            shell: true,
            ..Wizard::default()
        };
        apply(&mut s, &wiz);
        // Outer faces are solid.
        assert_eq!(s.map.cell_at(0, 0, 0), CellKind::WallSolid);
        assert_eq!(s.map.cell_at(4, 4, 1), CellKind::WallSolid);
        assert_eq!(s.map.cell_at(2, 0, 0), CellKind::WallSolid);
        // The interior of a 5x5x2 box: every cell is on a face (z spans 0..=1 only,
        // so there is no interior z), thus even the centre is shelled on z faces.
        // Use a taller box to prove an interior cell stays empty.
        let wiz2 = Wizard {
            sx: 5,
            sy: 5,
            sz: 5,
            shell: true,
            ..Wizard::default()
        };
        let mut s2 = EditorState::new_empty(4, 4, 2);
        apply(&mut s2, &wiz2);
        assert_eq!(
            s2.map.cell_at(2, 2, 2),
            CellKind::Empty,
            "interior is hollow"
        );
    }

    /// Creating a NEW level clears any inherited file path so a
    /// subsequent save can't silently overwrite the previously-opened file.
    #[test]
    fn new_resets_path() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.path = Some("levels/22-tower.lvl".into());
        let wiz = Wizard::default();
        apply(&mut s, &wiz);
        assert!(s.path.is_none(), "New level drops the inherited path");
        assert!(s.dirty, "New level is unsaved");
    }

    #[test]
    fn name_pop_removes_last_char_and_is_noop_when_empty() {
        let mut w = Wizard::default();
        w.name_clear();
        w.name_push('h');
        w.name_push('i');
        assert_eq!(w.name_buf, "hi");
        w.name_pop();
        assert_eq!(w.name_buf, "h", "name_pop removes the last char");
        w.name_pop();
        assert_eq!(w.name_buf, "");
        // Empty buffer: no panic.
        w.name_pop();
        assert_eq!(w.name_buf, "");
    }

    #[test]
    fn focus_cycles_all_fields() {
        let mut f = WizardField::SizeX;
        f = f.next();
        assert_eq!(f, WizardField::SizeY);
        f = f.next();
        assert_eq!(f, WizardField::SizeZ);
        f = f.next();
        assert_eq!(f, WizardField::Name);
        f = f.next();
        assert_eq!(f, WizardField::Shell);
        f = f.next();
        assert_eq!(f, WizardField::SizeX);
    }

    #[test]
    fn adjust_clamps_size_and_cycles_name() {
        let mut w = Wizard {
            sx: 1,
            focus: WizardField::SizeX,
            ..Wizard::default()
        };
        w.adjust(-1);
        assert_eq!(w.sx, MIN_DIM, "size clamps at MIN_DIM");
        w.focus = WizardField::Name;
        let start = w.name_idx;
        w.adjust(1);
        assert_ne!(w.name_idx, start, "name cycles");
        w.focus = WizardField::Shell;
        let shell = w.shell;
        w.adjust(1);
        assert_eq!(w.shell, !shell, "shell toggles");
    }
}
