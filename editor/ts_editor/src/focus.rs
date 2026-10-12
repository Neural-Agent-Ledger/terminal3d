//! FOCUS / INPUT-CONTEXT STACK for the editor's modal UI.
//!
//! Key dispatch is a STACK of input contexts walked top-down: each frame
//! either CONSUMES the key or passes it to the frame below. A residual guard
//! chain in `edit::apply_key` is the ROOT frame below every entry here and
//! still owns any modal not listed in [`InputContext`]; a modal only moves
//! onto this stack by deleting its guard from that chain, so the two never
//! double-handle a key.
//!
//! The stack is DERIVED from the modal state (`state.rename` / `help_open` /
//! `dir_picker` / …) rather than stored: the modal `Option`/`bool` fields stay
//! the single source of truth, so the stack can never desynchronize from what
//! is actually open (the classic dual-bookkeeping failure).
//!
//! Esc is NOT routed here: `main::quit_on_esc` owns the Esc close-precedence
//! chain.
use crate::state::EditorState;
use re_present::EditorKey;

/// One input-context frame. Ordered by DISPATCH PRIORITY — the variant listed
/// first is the topmost frame when its modal is open (mirroring the legacy
/// guard order rename → help → dir-picker).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputContext {
    /// The `:` command palette (`state.cmd_palette`). Topmost: while open,
    /// every key belongs to the query line.
    CommandPalette,
    /// The F2 rename prompt (`state.rename`).
    Rename,
    /// The `?` help popup (`state.help_open`).
    Help,
    /// The emitter 3×3×3 direction picker (`state.dir_picker`).
    DirPicker,
    /// The INLINE NUMERIC editor over a typed inspector row
    /// (`state.num_edit`). Fully modal: while open, every key belongs to the
    /// number being typed (which is why bare digits stay free to be geometry
    /// brushes at the root).
    NumEdit,
    /// The ENUM OPTION popup over a typed inspector row (`state.enum_pick`).
    EnumPick,
    /// The COLOUR overlay over a typed inspector row (`state.color_edit`).
    ColorEdit,
    /// The read-only validation drawer (`state.validation_open`). The
    /// first NON-modal frame: it consumes only its scroll/close keys and
    /// DECLINES everything else down the stack (editing keeps working under
    /// the open drawer).
    Validation,
}

/// The ACTIVE context stack, topmost frame FIRST (dispatch order). Derived
/// from the modal state each call — the modal fields remain the SSOT.
pub fn active_stack(state: &EditorState) -> Vec<InputContext> {
    let mut stack = Vec::new();
    if state.cmd_palette.is_some() {
        stack.push(InputContext::CommandPalette);
    }
    if state.rename.is_some() {
        stack.push(InputContext::Rename);
    }
    if state.help_open {
        stack.push(InputContext::Help);
    }
    if state.dir_picker.is_some() {
        stack.push(InputContext::DirPicker);
    }
    // The three ROW EDITORS sit above the validation drawer and below the
    // legacy modals: they are opened FROM the inspector, so nothing that can be
    // open at the same time outranks them except a genuinely global modal.
    if state.num_edit.is_some() {
        stack.push(InputContext::NumEdit);
    }
    if state.enum_pick.is_some() {
        stack.push(InputContext::EnumPick);
    }
    if state.color_edit.is_some() {
        stack.push(InputContext::ColorEdit);
    }
    if state.validation_open {
        stack.push(InputContext::Validation);
    }
    stack
}

/// Walk the stack top-down and offer `key` to each frame. Returns `true` when
/// a frame CONSUMED the key (dispatch stops — the key never reaches a lower
/// frame or the legacy chain), `false` when the stack is empty for this key
/// (the caller continues into the legacy chain / root frame).
///
/// Every migrated modal is FULLY modal (an open modal consumes every key
/// routed to it), so consumption == "the frame's modal is open"; the per-frame
/// handlers are the untouched `apply_*_key` bodies.
pub fn route_key(state: &mut EditorState, key: EditorKey) -> bool {
    // Walk the stack top-down; each frame either CONSUMES the key (dispatch
    // stops) or DECLINES it to the frame below. The fully-modal frames
    // (palette / rename / help / dir-picker) always consume; the VALIDATION
    // drawer is the first declining frame — it takes only scroll/close keys.
    for frame in active_stack(state) {
        let consumed = match frame {
            InputContext::CommandPalette => {
                crate::keymap::palette_key(state, key);
                true
            }
            InputContext::Rename => {
                crate::rename::apply_key(state, key);
                true
            }
            InputContext::Help => {
                crate::edit::apply_help_key(state, key);
                true
            }
            InputContext::DirPicker => {
                crate::edit::apply_dir_picker_key(state, key);
                true
            }
            InputContext::NumEdit => {
                crate::rowedit::num_key(state, key);
                true
            }
            InputContext::EnumPick => {
                crate::rowedit::enum_key(state, key);
                true
            }
            InputContext::ColorEdit => {
                crate::rowedit::color_key(state, key);
                true
            }
            InputContext::Validation => crate::validate::key(state, key),
        };
        if consumed {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_present::{EditorKey, EditorKeyCode as E};

    fn key(c: char) -> EditorKey {
        EditorKey {
            code: E::Char(c),
            shift: false,
        }
    }

    #[test]
    fn empty_stack_consumes_nothing() {
        let mut s = EditorState::new_empty(8, 8, 2);
        assert!(active_stack(&s).is_empty());
        assert!(
            !route_key(&mut s, key('x')),
            "no frame → key falls to the root chain"
        );
    }

    /// Frame priority mirrors the legacy guard order: rename above help above
    /// dir-picker.
    #[test]
    fn stack_orders_rename_above_help_above_dir_picker() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.help_open = true;
        s.dir_picker = Some(crate::picker::DirPickerState {
            emitter: 0,
            cx: 0,
            cy: 0,
            cz: 0,
        });
        assert_eq!(
            active_stack(&s),
            vec![InputContext::Help, InputContext::DirPicker]
        );
        s.rename = Some(crate::rename::RenameState {
            target: crate::rename::RenameTarget::Node(0),
            buf: "n".into(),
        });
        assert_eq!(
            active_stack(&s)[0],
            InputContext::Rename,
            "rename is the topmost frame"
        );
    }

    /// A key consumed at frame N never reaches frame N+1: `?` offered to an
    /// open HELP frame closes the help; a dir-picker sitting BELOW it stays
    /// open and unmoved.
    #[test]
    fn consumed_key_never_reaches_the_frame_below() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.help_open = true;
        s.dir_picker = Some(crate::picker::DirPickerState {
            emitter: 0,
            cx: 0,
            cy: 0,
            cz: 0,
        });
        let before = s.dir_picker;
        assert!(route_key(&mut s, key('?')), "help consumed the toggle");
        assert!(!s.help_open, "help closed itself");
        assert_eq!(s.dir_picker, before, "the frame below never saw the key");
        // The next key now routes to the dir-picker (help frame is gone).
        assert!(
            route_key(&mut s, key('q')),
            "dir-picker consumes while open"
        );
    }

    /// The rename frame types into the rename buffer (its `apply_key` body is
    /// the untouched modal handler).
    #[test]
    fn rename_frame_consumes_typed_chars() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(1, 1, 0);
        s.place = Some(crate::state::EntityKind::Node);
        s.selected_node = Some(0);
        crate::rename::open(&mut s);
        assert!(s.rename.is_some(), "rename prompt opened");
        let before = s.rename.as_ref().unwrap().buf.clone();
        assert!(route_key(&mut s, key('x')), "rename consumed the char");
        let after = s.rename.as_ref().unwrap().buf.clone();
        assert_eq!(
            after,
            format!("{before}x"),
            "char landed in the rename buffer"
        );
    }

    /// The COMMAND PALETTE is the TOPMOST frame — above an open rename/
    /// help — and consumes every key into its query.
    #[test]
    fn command_palette_is_topmost_and_consumes() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.help_open = true;
        crate::keymap::open_palette(&mut s);
        assert_eq!(
            active_stack(&s)[0],
            InputContext::CommandPalette,
            "palette outranks help"
        );
        assert!(route_key(&mut s, key('q')), "palette consumed the char");
        assert_eq!(s.cmd_palette.as_ref().unwrap().query, "q");
        assert!(s.help_open, "the frame below never saw the key");
    }

    /// The VALIDATION drawer is the first DECLINING frame — it consumes
    /// its scroll/close keys and passes everything else to the chain below.
    #[test]
    fn validation_frame_declines_non_scroll_keys() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.validation_open = true;
        s.last_check = vec!["verdict: PASS".into(), "PASS notes — ok".into()];
        assert_eq!(active_stack(&s), vec![InputContext::Validation]);
        assert!(
            route_key(
                &mut s,
                EditorKey {
                    code: E::Down,
                    shift: false
                }
            ),
            "Down is the drawer's (scroll)"
        );
        assert_eq!(s.validation_scroll, 1);
        assert!(
            !route_key(&mut s, key('x')),
            "a non-drawer key DECLINES down to the legacy chain"
        );
        assert!(s.validation_open, "declining does not close the drawer");
    }
}
