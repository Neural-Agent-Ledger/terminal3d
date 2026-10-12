//! ONE KEYMAP TABLE — the single source every discoverability surface is
//! GENERATED from:
//!
//! * the status bar's mode-aware HINT segment ([`hint_line`]),
//! * the `?` help cheat-sheet body ([`help_lines`] / [`help_lines_filtered`],
//!   consumed by `chrome` — the sheet is scrolled and live-filtered there, so
//!   its ~185 generated lines are all reachable on a 40-row terminal),
//! * the `:` COMMAND PALETTE candidates ([`palette_entries`] /
//!   [`palette_matches`]).
//!
//! Because all three render from [`KEYMAP`], a binding absent from the table
//! effectively does not exist to the player, and the three surfaces can never
//! drift apart — the drift-proof test below iterates the table and proves
//! every palette-eligible entry appears in BOTH the help and the palette.
//!
//! SCOPE: the table is the source for DISPLAY + PALETTE only. Key DISPATCH
//! stays in `edit::apply_key` and the modal handlers; palette execution
//! re-enters those existing paths by synthesizing the bound key
//! ([`Binding::key`]) through `edit::apply_key`, so no action gains a second
//! code path.
//!
//! The FUZZY SCORER is hand-rolled behind [`score`]: a swap-able
//! `fn(query, candidate) -> Option<i32>` seam, no dependency added;
//! `nucleo-matcher` remains a documented later swap if this proves inadequate.

use crate::state::{CmdPaletteState, EditorState};
use re_present::{EditorKey, EditorKeyCode as E};

/// The INPUT CONTEXT a binding belongs to — which surface owns the key while
/// that surface has focus. Mirrors the dispatch reality (guard chain + focus
/// stack), so the hint bar can show exactly the keys that work *right now*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Context {
    /// The root 2D-edit surface (no modal, no tool-specific overlay).
    Root,
    /// The outliner has focus (`state.tree.open`).
    Tree,
    /// A text field (note body / pool / label / inset path) has the caret.
    TextEntry,
    /// Material mode (`m`).
    Material,
    /// Object placement/selection mode (`o`).
    Object,
    /// Vox stamp mode (`v`).
    Vox,
    /// Staircase stamp tool (`8`).
    Stair,
    /// The part mask-edit modal (`q`).
    Mask,
    /// The 3D preview free-look capture.
    Preview,
    /// The `:` command palette itself.
    CmdPalette,
    /// The read-only validation drawer (`V`).
    Validation,
    /// The LEVEL SETTINGS modal (`L`) — the form over `[meta]` / `[room]` /
    /// `[effects]` / `[palette]` / `[sky]` / `[texture]` / `[door]` / `[inputs]`
    /// / `[sequence]`. It captures the keyboard modally, and its own keys (`a`
    /// add row, `x` remove row, `n`/`r` step names, `R` resize, typed text) are
    /// nothing like the root surface's, so without this variant they appeared
    /// in no hint bar, no `?` sheet and no `:` palette at all.
    Settings,
    /// The INLINE NUMERIC editor over a typed inspector row.
    RowNum,
    /// The ENUM OPTION popup over a typed inspector row.
    RowEnum,
    /// The COLOUR overlay over a typed inspector row.
    RowColor,
    // ── modals that swallow the keyboard with no discoverability variant ────
    // Each of these swallows the keyboard (in `edit::apply_key` or via
    // `focus::route_key`) but had no variant, so `hint_line` showed the keys of
    // whatever surface was UNDERNEATH — the hint bar actively lied. Each one is
    // paired with a sentinel-fenced dispatch scan (see the tests below), so its
    // rows can never drift from the keys it really handles.
    /// The CRASH-RECOVERY offer (`state.autosave_offer`) — the frontmost modal:
    /// a newer autosave sidecar was found and owns the keyboard until answered.
    AutosaveOffer,
    /// The F2 RENAME prompt (`state.rename`).
    Rename,
    /// The `?` HELP SHEET (`state.help_open`) — the generated cheat-sheet
    /// popup itself.
    ///
    /// This variant REVERSES an earlier deliberate decision (see the note in
    /// [`active_context`]): while the sheet handled exactly one key (`?`), a
    /// context here would have repeated one line under a window that lists
    /// every binding in the editor. The sheet now SCROLLS and FILTERS, and the
    /// keys that do that are the one set of keys the sheet cannot teach by
    /// listing them — a user who cannot scroll never reaches the row that says
    /// how to scroll. So they live in the table like everything else, and the
    /// fenced dispatch scan keeps them honest.
    Help,
    /// The emitter 3x3x3 DIRECTION PICKER (`state.dir_picker`).
    DirPicker,
    /// The LOAD/OPEN overlay (`state.open`).
    Open,
    /// The SAVE-AS filename prompt (`state.saveas`).
    SaveAs,
    /// The NEW-LEVEL wizard (`state.wizard`).
    Wizard,
    /// The object-TYPE CREATE palette (`state.palette`) — the `i` overlay, not
    /// the `:` command palette ([`Context::CmdPalette`]).
    ObjectPalette,
}

impl Context {
    /// Short display name (palette rows + hints).
    pub const fn label(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Tree => "outliner",
            Self::TextEntry => "text",
            Self::Material => "material",
            Self::Object => "object",
            Self::Vox => "vox",
            Self::Stair => "stair",
            Self::Mask => "mask",
            Self::Preview => "3D fly",
            Self::CmdPalette => "palette",
            Self::Validation => "validation",
            Self::Settings => "settings",
            Self::RowNum => "number",
            Self::RowEnum => "list",
            Self::RowColor => "colour",
            Self::AutosaveOffer => "recover",
            Self::Rename => "rename",
            Self::Help => "help",
            Self::DirPicker => "aim",
            Self::Open => "open",
            Self::SaveAs => "save as",
            Self::Wizard => "new level",
            Self::ObjectPalette => "create",
        }
    }
}

/// One row of the keymap table. `chord` is the DISPLAY chord (empty for a
/// documentation-only line); `key` is the synthesizable dispatch key for
/// palette execution (`None` = listed but not directly executable — the
/// palette then teaches the chord instead of running it).
pub struct Binding {
    /// Which focus context the chord works in.
    pub context: Context,
    /// Display chord (`"F2"`, `":"`, `"; '"`, `""` for prose lines).
    pub chord: &'static str,
    /// Human label ("Rename object (cascades refs)").
    pub label: &'static str,
    /// Help-sheet section this row renders under (see [`GROUPS`]).
    pub group: &'static str,
    /// `Some(short)` puts `chord=short` into the status-bar hint segment for
    /// this binding's context.
    pub hint: Option<&'static str>,
    /// The key the palette synthesizes through `edit::apply_key` to execute
    /// this action via the EXISTING dispatch path.
    pub key: Option<E>,
    /// `true` lists the row as a command-palette candidate.
    pub palette: bool,
}

/// Help-sheet section order. Every [`Binding::group`] value appears here.
pub const GROUPS: &[&str] = &[
    "2D edit",
    "inspector rows",
    "level settings",
    "panels / discover",
    "view / files",
    "undo / fill / new",
    "tree / palette",
    "objects",
    "pipe (wire 2 nodes)",
    "material",
    "vox / box",
    "emitter",
    "3D fly (preview)",
];

/// A doc-only row (no chord, not executable) — prose the help sheet keeps.
const fn doc(group: &'static str, label: &'static str) -> Binding {
    Binding {
        context: Context::Root,
        chord: "",
        label,
        group,
        hint: None,
        key: None,
        palette: false,
    }
}

/// THE keymap table. Display + palette source of truth (dispatch unchanged).
pub static KEYMAP: &[Binding] = &[
    // ── 2D edit ────────────────────────────────────────────────────────────
    doc("2D edit", "arrows move cursor   Shift+arrows pan"),
    doc("2D edit", "PgUp/PgDn z layer    +/- zoom (at cursor)"),
    Binding { context: Context::Root, chord: "space", label: "paint / place with active tool", group: "2D edit", hint: None, key: Some(E::Char(' ')), palette: true },
    Binding { context: Context::Root, chord: "x", label: "erase cell / delete under cursor", group: "2D edit", hint: None, key: Some(E::Char('x')), palette: true },
    doc("2D edit", "1-6 geometry brush   9 hologram   0 portal"),
    doc("2D edit", "MOUSE: left click/drag = paint stroke"),
    doc("2D edit", "  right click/drag = erase stroke"),
    doc("2D edit", "  middle drag = pan   wheel = zoom"),
    doc("2D edit", "tool-bar row (click) sets brush/tool"),
    Binding { context: Context::Root, chord: "7", label: "wedge/ramp brush ([ ] orient)", group: "2D edit", hint: None, key: Some(E::Char('7')), palette: true },
    Binding { context: Context::Root, chord: "8", label: "staircase stamp (\\ straight/spiral)", group: "2D edit", hint: None, key: Some(E::Char('8')), palette: true },
    doc("2D edit", "  [ ] dir  +/- radius  ; ' turns"),
    doc("2D edit", "  # toggle stairwell auto-carve"),
    Binding { context: Context::Root, chord: "n", label: "tag node at cursor", group: "2D edit", hint: None, key: Some(E::Char('n')), palette: true },
    Binding { context: Context::Root, chord: "s", label: "save level", group: "2D edit", hint: None, key: Some(E::Char('s')), palette: true },
    Binding { context: Context::Root, chord: "Tab", label: "cycle view (Split/2D/3D)", group: "2D edit", hint: None, key: Some(E::Tab), palette: true },
    Binding { context: Context::Root, chord: "<", label: "floor down (whole storey)", group: "2D edit", hint: None, key: Some(E::Char('<')), palette: true },
    Binding { context: Context::Root, chord: ">", label: "floor up (whole storey)", group: "2D edit", hint: None, key: Some(E::Char('>')), palette: true },
    // ── inspector rows (typed row widgets) ─────────────────────────────────
    Binding { context: Context::Root, chord: "Enter", label: "edit focused inspector row (number / list / colour)", group: "inspector rows", hint: Some("edit row"), key: Some(E::Enter), palette: true },
    doc("inspector rows", "rows show their type: [x] bool  [< >] list"),
    doc("inspector rows", "  [-] n [+] number   #RRGGBB block colour"),
    doc("inspector rows", "MOUSE: click a row LABEL focuses that field"),
    doc("inspector rows", "  click [x] toggles   click [-]/[+] steps"),
    doc("inspector rows", "  click the number / [< >] / swatch opens an editor"),
    Binding { context: Context::RowNum, chord: "0-9 . -", label: "type a number", group: "inspector rows", hint: Some("type"), key: None, palette: false },
    Binding { context: Context::RowNum, chord: "Enter", label: "commit number (clamped to the field's range)", group: "inspector rows", hint: Some("commit"), key: None, palette: false },
    Binding { context: Context::RowNum, chord: "Esc", label: "cancel number entry", group: "inspector rows", hint: Some("cancel"), key: None, palette: false },
    Binding { context: Context::RowEnum, chord: "Up/Dn", label: "move option highlight", group: "inspector rows", hint: Some("nav"), key: None, palette: false },
    Binding { context: Context::RowEnum, chord: "Enter", label: "pick option", group: "inspector rows", hint: Some("pick"), key: None, palette: false },
    Binding { context: Context::RowEnum, chord: "Esc", label: "cancel option list", group: "inspector rows", hint: Some("cancel"), key: None, palette: false },
    Binding { context: Context::RowColor, chord: "Tab", label: "hex field / palette / used-in-level", group: "inspector rows", hint: Some("section"), key: None, palette: false },
    Binding { context: Context::RowColor, chord: "0-9 a-f", label: "type a hex colour", group: "inspector rows", hint: Some("hex"), key: None, palette: false },
    Binding { context: Context::RowColor, chord: "arrows", label: "move swatch highlight", group: "inspector rows", hint: Some("nav"), key: None, palette: false },
    Binding { context: Context::RowColor, chord: "Enter", label: "commit colour", group: "inspector rows", hint: Some("commit"), key: None, palette: false },
    Binding { context: Context::RowColor, chord: "Esc", label: "cancel colour edit", group: "inspector rows", hint: Some("cancel"), key: None, palette: false },
    // ── level settings modal ────────────────────────────────────────────────
    // EVERY key `edit::apply_settings_key` dispatches has a row here. That is an
    // enforced invariant, not a convention: `every_settings_key_is_in_the_keymap`
    // scans the fenced dispatch block in `edit.rs` and fails if a key is handled
    // there without a row here. A chord may name SEVERAL keys separated by `/`
    // (the test splits on it), which is how `Left/-` covers both.
    doc("level settings", "E opens LEVEL SETTINGS (meta/room/effects/fog/sky/textures/door/inputs/sequence)"),
    Binding { context: Context::Settings, chord: "Tab", label: "next field", group: "level settings", hint: Some("field"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Up/Down", label: "step focus (in a list section: move the row cursor)", group: "level settings", hint: Some("focus"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Left/-", label: "decrement / cycle back / toggle", group: "level settings", hint: Some("less"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Right/+/=", label: "increment / cycle forward / toggle", group: "level settings", hint: Some("more"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Space", label: "toggle bool / the highlighted input node", group: "level settings", hint: Some("toggle"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "0-9", label: "type a power_budget digit", group: "level settings", hint: Some("digits"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Backspace", label: "delete a character from the focused text field", group: "level settings", hint: None, key: None, palette: false },
    Binding { context: Context::Settings, chord: "a", label: "add a row (observe / step / texture)", group: "level settings", hint: Some("add"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "x", label: "remove the row under the cursor", group: "level settings", hint: Some("del"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "n", label: "add a name into the selected sequence step", group: "level settings", hint: Some("+name"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "r", label: "remove the name under the in-step cursor", group: "level settings", hint: Some("-name"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "R", label: "RESIZE the map to the staged W/H/D (modal stays open)", group: "level settings", hint: Some("resize"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Enter", label: "apply the whole form (one undo step)", group: "level settings", hint: Some("apply"), key: None, palette: false },
    Binding { context: Context::Settings, chord: "Esc", label: "discard the form", group: "level settings", hint: Some("cancel"), key: None, palette: false },
    // ── panels / discover ────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "?", label: "help cheat-sheet", group: "panels / discover", hint: Some("help"), key: Some(E::Char('?')), palette: true },
    Binding { context: Context::Root, chord: ":", label: "command palette (fuzzy actions)", group: "panels / discover", hint: Some("palette"), key: Some(E::Char(':')), palette: false },
    Binding { context: Context::Root, chord: "z", label: "zoom pane toggle (fullscreen focused viewport)", group: "panels / discover", hint: Some("zoom"), key: Some(E::Char('z')), palette: true },
    Binding { context: Context::Root, chord: "V", label: "validation report drawer (last save's lvlcheck)", group: "panels / discover", hint: None, key: Some(E::Char('V')), palette: true },
    doc("panels / discover", "rail icons (right edge) reopen collapsed panels"),
    doc("panels / discover", "narrow terminals shed panels to the rail"),
    Binding { context: Context::Tree, chord: "/", label: "filter outliner rows (Esc clears)", group: "panels / discover", hint: Some("filter"), key: None, palette: false },
    // The `?` SHEET'S OWN KEYS. The sheet is ~180 rows over a ~37-row window, so
    // it scrolls; and at that length it is a reference document, so it filters.
    // Fenced + scanned in `edit::apply_help_key` like every other modal.
    Binding { context: Context::Help, chord: "Up/Down", label: "scroll the help sheet one line", group: "panels / discover", hint: Some("scroll"), key: None, palette: false },
    Binding { context: Context::Help, chord: "PageUp/PageDown", label: "scroll the help sheet one page", group: "panels / discover", hint: Some("page"), key: None, palette: false },
    Binding { context: Context::Help, chord: "Home/End", label: "jump to the first / last help row", group: "panels / discover", hint: Some("ends"), key: None, palette: false },
    Binding { context: Context::Help, chord: "a-z 0-9", label: "filter help rows (Esc clears)", group: "panels / discover", hint: Some("filter"), key: None, palette: false },
    Binding { context: Context::Help, chord: "Backspace", label: "erase the last help-filter character", group: "panels / discover", hint: None, key: None, palette: false },
    Binding { context: Context::Help, chord: "?", label: "close the help sheet", group: "panels / discover", hint: Some("close"), key: None, palette: false },
    Binding { context: Context::Help, chord: "Esc", label: "clear the help filter, then close the sheet", group: "panels / discover", hint: Some("clear"), key: None, palette: false },
    Binding { context: Context::CmdPalette, chord: "Enter", label: "run selected command", group: "panels / discover", hint: Some("run"), key: None, palette: false },
    Binding { context: Context::CmdPalette, chord: "Esc", label: "close palette", group: "panels / discover", hint: Some("close"), key: None, palette: false },
    Binding { context: Context::Validation, chord: "Up/Dn", label: "scroll report", group: "panels / discover", hint: Some("scroll"), key: None, palette: false },
    Binding { context: Context::Validation, chord: "V", label: "close drawer", group: "panels / discover", hint: Some("close"), key: None, palette: false },
    // ── view / files ───────────────────────────────────────────────────────
    doc("view / files", "toolbar: [New level][Open][Save][Save As]"),
    doc("view / files", "  [Undo][Redo] [2D view][3D view]"),
    doc("view / files", "  [Split view][Level][Help]"),
    Binding { context: Context::Root, chord: "O", label: "open level overlay", group: "view / files", hint: None, key: Some(E::Char('O')), palette: true },
    Binding { context: Context::Root, chord: "A", label: "save as (name the file)", group: "view / files", hint: None, key: Some(E::Char('A')), palette: true },
    Binding { context: Context::Root, chord: "F2", label: "rename object (cascades refs)", group: "view / files", hint: None, key: Some(E::F(2)), palette: true },
    Binding { context: Context::Root, chord: "F5", label: "playtest level (confirm on FAIL)", group: "view / files", hint: None, key: Some(E::F(5)), palette: true },
    Binding { context: Context::Root, chord: "Shift+R", label: "rebuild 3D preview (assets)", group: "view / files", hint: None, key: Some(E::Char('R')), palette: true },
    Binding { context: Context::Root, chord: "E", label: "level settings ([Level] form: meta/room/effects/palette/sky/texture/inputs/sequence)", group: "view / files", hint: Some("settings"), key: Some(E::Char('E')), palette: true },
    // -- VIEW AID (onion skin, `crate::viewaids`) ----------------------------
    // OFF by default and pure view state. It lives in the table for the usual
    // reason: a binding absent from it is invisible in the hint bar, the `?`
    // sheet AND the `:` palette, i.e. it does not exist.
    Binding { context: Context::Root, chord: "G", label: "onion skin: ghost the layers below/above (off / \u{00B1}1 / \u{00B1}2)", group: "view / files", hint: Some("onion"), key: Some(E::Char('G')), palette: true },
    doc("view / files", "onion ghosts are paint, not targets: a click always edits the live z layer"),
    // -- VIEW AID (sound fields, `crate::field`) -----------------------------
    // OFF by default, pure view state — same contract as `G` above.
    Binding { context: Context::Root, chord: "F", label: "sound-field aid: tint each field's radius/inner ring by archetype family", group: "view / files", hint: Some("fields"), key: Some(E::Char('F')), palette: true },
    // -- LIVE AUDITION (`crate::field::Audition`) ----------------------------
    // OFF by default. Never opens a device under TS_AUDIO_SILENT or with no
    // device present (`re_audio::Player::open`'s own silent-degrade contract,
    // checked a step earlier by `Audition::start`).
    Binding { context: Context::Root, chord: "Shift+U", label: "live audition: play the level's ambience + drone through the field chain from the preview camera", group: "view / files", hint: Some("audition"), key: Some(E::Char('U')), palette: true },
    doc("view / files", "Backspace edits text in Save-As / name"),
    doc("view / files", "z-slider clickable"),
    // ── the FILE MODALS' own keys ───────────────────────────────────────────
    // Each block below is fenced in its dispatch and scanned by
    // `every_dispatched_key_is_in_the_keymap`, so a key handled there without a
    // row here fails the build rather than vanishing from all three surfaces.
    Binding { context: Context::Open, chord: "Up/Down", label: "move the file highlight", group: "view / files", hint: Some("nav"), key: None, palette: false },
    Binding { context: Context::Open, chord: "Enter", label: "load the highlighted level (a dirty document needs a second Enter)", group: "view / files", hint: Some("load"), key: None, palette: false },
    Binding { context: Context::Open, chord: "O", label: "close the open overlay", group: "view / files", hint: Some("close"), key: None, palette: false },
    Binding { context: Context::SaveAs, chord: "a-z 0-9", label: "type the filename (letters, digits, - and _ only)", group: "view / files", hint: Some("type"), key: None, palette: false },
    Binding { context: Context::SaveAs, chord: "Backspace", label: "delete the last character of the filename", group: "view / files", hint: Some("del"), key: None, palette: false },
    Binding { context: Context::SaveAs, chord: "Enter", label: "confirm the name and save (a second Enter confirms an overwrite)", group: "view / files", hint: Some("save"), key: None, palette: false },
    Binding { context: Context::Rename, chord: "a-z 0-9", label: "type a name character", group: "view / files", hint: Some("type"), key: None, palette: false },
    Binding { context: Context::Rename, chord: "-/_", label: "the only punctuation a name may contain (parser-safe token)", group: "view / files", hint: None, key: None, palette: false },
    Binding { context: Context::Rename, chord: "Backspace", label: "delete the last character of the name", group: "view / files", hint: Some("del"), key: None, palette: false },
    Binding { context: Context::Rename, chord: "Enter", label: "apply the new name (cascades every reference)", group: "view / files", hint: Some("apply"), key: None, palette: false },
    Binding { context: Context::AutosaveOffer, chord: "y/Y", label: "restore the newer autosave sidecar into the editor", group: "view / files", hint: Some("restore"), key: None, palette: false },
    Binding { context: Context::AutosaveOffer, chord: "n/N", label: "discard the offer (the sidecar file is kept on disk)", group: "view / files", hint: Some("discard"), key: None, palette: false },
    // ── undo / fill / new ──────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "u", label: "undo", group: "undo / fill / new", hint: None, key: Some(E::Char('u')), palette: true },
    Binding { context: Context::Root, chord: "y", label: "redo", group: "undo / fill / new", hint: None, key: Some(E::Char('y')), palette: true },
    Binding { context: Context::Root, chord: "k", label: "fill corner A at cursor", group: "undo / fill / new", hint: None, key: Some(E::Char('k')), palette: true },
    Binding { context: Context::Root, chord: "K", label: "rect fill A->cursor", group: "undo / fill / new", hint: None, key: Some(E::Char('K')), palette: true },
    Binding { context: Context::Root, chord: "L", label: "line fill A->cursor", group: "undo / fill / new", hint: None, key: Some(E::Char('L')), palette: true },
    Binding { context: Context::Root, chord: "J", label: "bucket fill at cursor", group: "undo / fill / new", hint: None, key: Some(E::Char('J')), palette: true },
    Binding { context: Context::Root, chord: "W", label: "new-level wizard", group: "undo / fill / new", hint: None, key: Some(E::Char('W')), palette: true },
    // The wizard's OWN modal keys (fenced + scanned in edit.rs).
    Binding { context: Context::Wizard, chord: "Tab", label: "focus the next wizard field", group: "undo / fill / new", hint: Some("field"), key: None, palette: false },
    Binding { context: Context::Wizard, chord: "+/=", label: "grow the focused field / next preset", group: "undo / fill / new", hint: Some("more"), key: None, palette: false },
    Binding { context: Context::Wizard, chord: "-", label: "shrink the focused field / previous preset", group: "undo / fill / new", hint: Some("less"), key: None, palette: false },
    Binding { context: Context::Wizard, chord: "a-z 0-9", label: "type into the level Name (while Name is focused)", group: "undo / fill / new", hint: Some("name"), key: None, palette: false },
    Binding { context: Context::Wizard, chord: "Backspace", label: "delete the last character of the Name", group: "undo / fill / new", hint: None, key: None, palette: false },
    Binding { context: Context::Wizard, chord: "Enter", label: "create the level and close the wizard", group: "undo / fill / new", hint: Some("create"), key: None, palette: false },
    // ── tree / palette ─────────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "t", label: "toggle outliner focus", group: "tree / palette", hint: None, key: Some(E::Char('t')), palette: true },
    Binding { context: Context::Root, chord: "i", label: "create-object palette", group: "tree / palette", hint: None, key: Some(E::Char('i')), palette: true },
    Binding { context: Context::Tree, chord: "Up/Dn", label: "navigate rows", group: "tree / palette", hint: Some("nav"), key: None, palette: false },
    Binding { context: Context::Tree, chord: "Enter", label: "select + jump to object", group: "tree / palette", hint: Some("select"), key: None, palette: false },
    Binding { context: Context::Tree, chord: "h", label: "fold group", group: "tree / palette", hint: None, key: None, palette: false },
    Binding { context: Context::Tree, chord: "z", label: "duplicate object", group: "tree / palette", hint: Some("dup"), key: None, palette: false },
    Binding { context: Context::Tree, chord: "d", label: "delete object", group: "tree / palette", hint: Some("del"), key: None, palette: false },
    Binding { context: Context::Tree, chord: "p", label: "reparent (pick, confirm)", group: "tree / palette", hint: None, key: None, palette: false },
    Binding { context: Context::Tree, chord: "u", label: "unparent / cancel reparent", group: "tree / palette", hint: None, key: None, palette: false },
    Binding { context: Context::Tree, chord: "t", label: "close outliner", group: "tree / palette", hint: Some("close"), key: None, palette: false },
    // The create palette's OWN modal keys (fenced + scanned).
    Binding { context: Context::ObjectPalette, chord: "Up/Down", label: "move the object-type highlight", group: "tree / palette", hint: Some("nav"), key: None, palette: false },
    Binding { context: Context::ObjectPalette, chord: "Enter", label: "pick the type (enters placement mode)", group: "tree / palette", hint: Some("pick"), key: None, palette: false },
    Binding { context: Context::ObjectPalette, chord: "i", label: "close the create palette", group: "tree / palette", hint: Some("close"), key: None, palette: false },
    // ── objects ────────────────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "o", label: "cycle object kind (place with space)", group: "objects", hint: None, key: Some(E::Char('o')), palette: true },
    Binding { context: Context::Object, chord: "; '", label: "focus prev/next field", group: "objects", hint: Some("field"), key: None, palette: false },
    Binding { context: Context::Object, chord: "+/-", label: "adjust focused number", group: "objects", hint: Some("adjust"), key: None, palette: false },
    Binding { context: Context::Object, chord: "c", label: "cycle enum / toggle", group: "objects", hint: Some("cycle"), key: None, palette: false },
    Binding { context: Context::Object, chord: "x", label: "delete entity under cursor", group: "objects", hint: None, key: None, palette: false },
    doc("objects", "inspector rows scroll; click a row to focus"),
    Binding { context: Context::TextEntry, chord: "arrows", label: "move caret", group: "objects", hint: Some("caret"), key: None, palette: false },
    Binding { context: Context::TextEntry, chord: "Bksp/Del", label: "delete before / at caret", group: "objects", hint: Some("edit"), key: None, palette: false },
    Binding { context: Context::TextEntry, chord: "Home/End", label: "line start / end", group: "objects", hint: Some("home/end"), key: None, palette: false },
    doc("objects", "  note body: Enter = newline"),
    doc("objects", "  pool/event label: free words"),
    doc("objects", "Door: target on/off toggle"),
    doc("objects", "Node: pos/glyph/color/io/thr/anchor/pool"),
    doc("objects", "Part: whd/off/color  q edit mask"),
    // MASK modal — one row per key `edit::apply_mask_key` dispatches.
    Binding { context: Context::Mask, chord: "Left/Right/Up/Down", label: "move the mask cursor inside the active layer", group: "objects", hint: Some("move"), key: None, palette: false },
    Binding { context: Context::Mask, chord: "[/]", label: "step the active h-layer down / up", group: "objects", hint: Some("layer"), key: None, palette: false },
    Binding { context: Context::Mask, chord: "Space", label: "toggle the voxel under the cursor solid<->empty", group: "objects", hint: Some("toggle"), key: None, palette: false },
    Binding { context: Context::Mask, chord: "x", label: "force the voxel under the cursor EMPTY", group: "objects", hint: Some("clear"), key: None, palette: false },
    Binding { context: Context::Mask, chord: "q", label: "exit mask edit", group: "objects", hint: Some("exit"), key: None, palette: false },
    doc("objects", "Ped: whd   Trig: min/max box c action"),
    // ── pipe ───────────────────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "P", label: "pipe: pick source node (space/Enter target)", group: "pipe (wire 2 nodes)", hint: None, key: Some(E::Char('P')), palette: true },
    doc("pipe (wire 2 nodes)", "P again cancels   g switch cell"),
    doc("pipe (wire 2 nodes)", "; ' field  c state/switch  +/- density/color"),
    // ── material ───────────────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "m", label: "material mode / next material", group: "material", hint: None, key: Some(E::Char('m')), palette: true },
    Binding { context: Context::Root, chord: "e", label: "new material", group: "material", hint: None, key: Some(E::Char('e')), palette: true },
    Binding { context: Context::Material, chord: "b", label: "cycle material field", group: "material", hint: Some("field"), key: None, palette: false },
    Binding { context: Context::Material, chord: "+/-", label: "adjust focused field", group: "material", hint: Some("adjust"), key: None, palette: false },
    Binding { context: Context::Material, chord: "c", label: "cycle enum field", group: "material", hint: Some("cycle"), key: None, palette: false },
    Binding { context: Context::Material, chord: "space", label: "paint material id", group: "material", hint: Some("paint"), key: None, palette: false },
    Binding { context: Context::Material, chord: "x", label: "clear cell material", group: "material", hint: None, key: None, palette: false },
    // ── vox / box ──────────────────────────────────────────────────────────
    Binding { context: Context::Root, chord: "v", label: "vox mode / cycle .vox file", group: "vox / box", hint: None, key: Some(E::Char('v')), palette: true },
    Binding { context: Context::Vox, chord: "space", label: "place vox stamp", group: "vox / box", hint: Some("place"), key: None, palette: false },
    Binding { context: Context::Vox, chord: "c", label: "cycle surface material", group: "vox / box", hint: Some("material"), key: None, palette: false },
    Binding { context: Context::Vox, chord: "+/-", label: "scale stamp", group: "vox / box", hint: Some("scale"), key: None, palette: false },
    Binding { context: Context::Vox, chord: "x", label: "delete nearest stamp", group: "vox / box", hint: Some("del"), key: None, palette: false },
    // ── emitter ────────────────────────────────────────────────────────────
    doc("emitter", "; ' focus  [group] shown  +/- num/color  c enum"),
    Binding { context: Context::Root, chord: "D", label: "emitter aim 3x3x3 picker (26 dirs)", group: "emitter", hint: None, key: Some(E::Char('D')), palette: true },
    // The picker's OWN modal keys (fenced + scanned).
    Binding { context: Context::DirPicker, chord: "Left/Right/Up/Down", label: "move the aim cursor in the current layer", group: "emitter", hint: Some("aim"), key: None, palette: false },
    Binding { context: Context::DirPicker, chord: "PageUp/]", label: "switch to the layer above", group: "emitter", hint: Some("up"), key: None, palette: false },
    Binding { context: Context::DirPicker, chord: "PageDown/[", label: "switch to the layer below", group: "emitter", hint: Some("down"), key: None, palette: false },
    Binding { context: Context::DirPicker, chord: "Enter/Space/c", label: "pick the aim cell (the centre keeps the current aim)", group: "emitter", hint: Some("pick"), key: None, palette: false },
    Binding { context: Context::DirPicker, chord: "D", label: "close the picker without changing the aim", group: "emitter", hint: Some("close"), key: None, palette: false },
    Binding { context: Context::Root, chord: "H", label: "show/hide object markers", group: "emitter", hint: None, key: Some(E::Char('H')), palette: true },
    // ── 3D fly ─────────────────────────────────────────────────────────────
    Binding { context: Context::Preview, chord: "W/A/D", label: "move (Shift+S back, R/F up/down)", group: "3D fly (preview)", hint: Some("move"), key: None, palette: false },
    Binding { context: Context::Preview, chord: ", .", label: "turn (one step per press)", group: "3D fly (preview)", hint: Some("turn"), key: None, palette: false },
    // The two rows below match `mouse.rs::apply_mouse`: free-look is
    // HOLD-to-look on the RIGHT button
    // (`capture_request = Some(true)` on RIGHT press; the ONLY thing that ends
    // capture is the RIGHT RELEASE), and a LEFT drag over the preview is inert
    // — the old drag-look path is gone. The wheel row was already true.
    doc("3D fly (preview)", "LEFT-drag preview = inert (look is RMB-held)"),
    doc("3D fly (preview)", "HOLD RIGHT over preview = free-look (release ends it)"),
    doc("3D fly (preview)", "wheel over preview = move fwd/back"),
    doc("3D fly (preview)", "Enter jump-to-cursor  Shift+N noclip"),
    Binding { context: Context::Stair, chord: "\\", label: "straight <-> spiral", group: "2D edit", hint: Some("shape"), key: None, palette: false },
    doc("3D fly (preview)", "Esc: release look / close / quit (x2 unsaved)"),
];

/// The active input context for hint-bar purposes — mirrors the dispatch
/// precedence (modal chain / focus stack first, then tool modes, then Root).
pub fn active_context(state: &EditorState) -> Context {
    // ORDER = the order `edit::apply_key` really resolves these:
    // preview-capture -> autosave offer -> mask -> `focus::route_key`
    // (cmd-palette -> rename -> help -> dir-picker -> num -> enum -> colour ->
    // validation) -> open -> save-as -> wizard -> settings -> create palette ->
    // outliner -> text entry -> tool modes -> Root. Mirroring it matters when
    // two surfaces are open at once: the hint bar must show the one that will
    // actually eat the next key.
    //
    // `state.help_open` USED TO get no context, on the reasoning that the `?`
    // popup IS the generated help sheet, its only key was `?`/Esc, and its own
    // last line ("press ? or Esc to close") already said so — a hint context
    // would have repeated that one line under a window listing every binding in
    // the editor. That held for a ONE-KEY modal. It stopped holding when the
    // sheet grew past one screen: it now scrolls (Up/Dn, PgUp/PgDn, Home/End)
    // and filters (type / Backspace), and those are precisely the keys a user
    // cannot learn FROM the sheet, because not knowing them is what leaves the
    // rest of the sheet unreachable. Hence `Context::Help`, its rows, and the
    // fenced dispatch scan over `edit::apply_help_key`.
    if state.preview_captured {
        // Free-look capture returns from `apply_key` before ANY other guard, so
        // it is the topmost hint context too (the 3D fly keys are handled by
        // `camera_input_system`, which is why the rows are documentation).
        Context::Preview
    } else if state.autosave_offer.is_some() {
        Context::AutosaveOffer
    } else if state.mask_mode {
        // The mask modal is checked BEFORE the focus stack in `apply_key`.
        Context::Mask
    } else if state.cmd_palette.is_some() {
        Context::CmdPalette
    } else if state.rename.is_some() {
        Context::Rename
    } else if state.help_open {
        // `focus::route_key` orders the stack rename -> HELP -> dir-picker, and
        // the help frame is fully modal (it consumes every key routed to it),
        // so it must sit exactly here.
        Context::Help
    } else if state.dir_picker.is_some() {
        Context::DirPicker
    } else if state.num_edit.is_some() {
        // The three ROW EDITORS are modal over every mode below them, so the
        // hint bar must show THEIR keys while one is open — ordered to match
        // the focus stack in `focus.rs`.
        Context::RowNum
    } else if state.enum_pick.is_some() {
        Context::RowEnum
    } else if state.color_edit.is_some() {
        Context::RowColor
    } else if state.validation_open {
        // The drawer is a DECLINING frame (it takes only its scroll/close keys),
        // but while it is open those are the keys worth advertising.
        Context::Validation
    } else if state.open.is_some() {
        Context::Open
    } else if state.saveas.is_some() {
        Context::SaveAs
    } else if state.wizard.is_some() {
        Context::Wizard
    } else if state.settings.is_some() {
        // The LEVEL SETTINGS form captures the keyboard modally in
        // `edit::apply_key` (after the mask modal and the focus stack, before
        // every tool mode), so it outranks the outliner / text-entry / tool
        // contexts below — the hint bar must show ITS keys while it is open.
        Context::Settings
    } else if state.palette.is_some() {
        // The object-TYPE create palette (`i`) — checked after the settings form
        // and before the outliner, exactly as `apply_key` does.
        Context::ObjectPalette
    } else if state.tree.open {
        Context::Tree
    } else if crate::entity::text_entry_active(state) {
        Context::TextEntry
    } else if state.vox_mode {
        Context::Vox
    } else if state.material_mode {
        Context::Material
    } else if state.place.is_some() {
        Context::Object
    } else if state.stair_mode {
        Context::Stair
    } else {
        Context::Root
    }
}

/// The MODE-AWARE status-bar hint segment, generated from the table: every
/// binding of the active context with a `hint` label, as `chord=hint` pairs.
/// Root always leads with `?=help` etc. because those rows carry hints in the
/// table — nothing here is hand-maintained.
pub fn hint_line(state: &EditorState) -> String {
    let ctx = active_context(state);
    let mut parts: Vec<String> = KEYMAP
        .iter()
        .filter(|b| b.context == ctx)
        .filter_map(|b| b.hint.map(|h| format!("{}={h}", b.chord)))
        .collect();
    if parts.is_empty() {
        // A context with no hinted rows still points at the help sheet.
        parts.push("?=help".into());
    }
    parts.join("  ")
}

/// The `?` help cheat-sheet body, generated from the table: one `-- group --`
/// banner per [`GROUPS`] entry, then each row as `chord  label` (doc rows print
/// their label verbatim). Non-Root rows carry a `[context]` suffix so the
/// overloads (`z`, `x`, `c`, `; '`) read unambiguously.
///
/// The UNFILTERED sheet — [`help_lines_filtered`] with an empty query.
pub fn help_lines() -> Vec<String> {
    help_lines_filtered("")
}

/// One row of the sheet, as the `?` popup prints it.
fn help_row_text(b: &Binding) -> String {
    if b.chord.is_empty() {
        b.label.to_string()
    } else if b.context == Context::Root {
        format!("{} {}", b.chord, b.label)
    } else {
        format!("{} {}  [{}]", b.chord, b.label, b.context.label())
    }
}

/// Does `row` survive the help sheet's live filter? `query` must already be
/// lowercased and trimmed. A row matches on its CHORD, its LABEL, or its GROUP,
/// so "mask", "q" and "level settings" all find something sensible.
///
/// SUBSTRING, not [`score`]. The outliner filter fuzzy-matches because it runs
/// over a handful of short object names, where a subsequence hit is nearly
/// always the row you meant. This sheet is ~180 rows of PROSE (half of them
/// doc lines a sentence long), and a subsequence match over that corpus returns
/// most of the sheet for any short query — `"ae"` alone matches a clear
/// majority of the labels, which is indistinguishable from not filtering. A
/// literal substring is what a reader of a reference document expects when they
/// type a word they half-remember, and it makes the match COUNT meaningful.
fn help_row_matches(query: &str, b: &Binding) -> bool {
    b.chord.to_lowercase().contains(query)
        || b.label.to_lowercase().contains(query)
        || b.group.to_lowercase().contains(query)
}

/// The `?` help cheat-sheet body, filtered by `query` (empty = the whole sheet).
///
/// A group's banner is emitted only when at least one of its rows survived — a
/// `-- emitter --` heading over nothing is worse than no heading — and while a
/// filter is active the banner carries the surviving row COUNT so the user can
/// see where the matches landed. The closing "press ? or Esc" line belongs to
/// the unfiltered sheet only; while filtering, the popup's own footer band says
/// it (and says it without being filtered away).
pub fn help_lines_filtered(query: &str) -> Vec<String> {
    let q = query.trim().to_lowercase();
    let mut out = Vec::with_capacity(KEYMAP.len() + GROUPS.len() + 2);
    for &g in GROUPS {
        let rows: Vec<&Binding> = KEYMAP
            .iter()
            .filter(|b| b.group == g)
            .filter(|b| q.is_empty() || help_row_matches(&q, b))
            .collect();
        if q.is_empty() {
            out.push(format!("-- {g} --"));
        } else if rows.is_empty() {
            continue;
        } else {
            out.push(format!("-- {g} ({}) --", rows.len()));
        }
        out.extend(rows.into_iter().map(help_row_text));
    }
    if q.is_empty() {
        out.push(String::new());
        out.push("press ? or Esc to close".into());
    }
    out
}

/// How many ROWS (banners excluded) the sheet shows for `query` — the "9" in
/// the popup's `filter: mask   9 of 178` readout. Pure; shared by the painter
/// and the tests so the number on screen is the number that was counted.
#[must_use]
pub fn help_match_count(query: &str) -> usize {
    let q = query.trim().to_lowercase();
    KEYMAP
        .iter()
        .filter(|b| q.is_empty() || help_row_matches(&q, b))
        .count()
}

// ─── FUZZY SCORER (hand-rolled, swap-able) ────────────────────────────────

/// Case-insensitive SUBSEQUENCE scorer: `Some(score)` when every query char
/// appears in order inside `candidate`, `None` otherwise. Higher is better:
/// consecutive matches and word-start matches earn bonuses, gaps cost a little,
/// shorter candidates win ties. Deterministic, allocation-light. The palette
/// and the outliner filter share this single seam so a later swap (e.g.
/// `nucleo-matcher`, if this scorer proves inadequate) changes ONE function.
pub fn score(query: &str, candidate: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let q: Vec<char> = query.chars().flat_map(char::to_lowercase).collect();
    let c: Vec<char> = candidate.chars().flat_map(char::to_lowercase).collect();
    let mut total = 0i32;
    let mut qi = 0usize;
    let mut prev_hit: Option<usize> = None;
    for (ci, &ch) in c.iter().enumerate() {
        if qi >= q.len() {
            break;
        }
        if ch == q[qi] {
            total += 2;
            if prev_hit == Some(ci.wrapping_sub(1)) {
                total += 3; // consecutive run bonus
            }
            let word_start = ci == 0 || !c[ci - 1].is_alphanumeric();
            if word_start {
                total += 2;
            }
            if let Some(p) = prev_hit {
                total -= i32::try_from(ci.saturating_sub(p + 1))
                    .unwrap_or(i32::MAX)
                    .min(8);
            }
            prev_hit = Some(ci);
            qi += 1;
        }
    }
    if qi < q.len() {
        return None;
    }
    // Prefer tighter candidates: small penalty for leftover length.
    total -= i32::try_from(c.len().saturating_sub(q.len())).unwrap_or(i32::MAX) / 8;
    Some(total)
}

// ─── COMMAND PALETTE ─────────────────────────────────────────────────────────

/// What executing a palette row does.
#[derive(Clone, Debug)]
pub enum PaletteAction {
    /// Synthesize this key through `edit::apply_key` (the existing dispatch).
    Key(E),
    /// Jump the cursor/selection to an outliner object.
    Goto(crate::tree::TreeTarget),
    /// Load a level file in-place.
    OpenLevel(String),
}

/// One palette candidate row: display text + chord + optional action.
#[derive(Clone, Debug)]
pub struct PaletteEntry {
    /// Row text shown (and fuzzy-matched).
    pub text: String,
    /// The chord column (teaches the fast path; empty for goto/open rows).
    pub chord: String,
    /// What Enter does; `None` rows only teach their chord.
    pub action: Option<PaletteAction>,
}

/// ALL palette candidates for the current state, unfiltered: every
/// palette-eligible keymap action, `goto <object>` for every outliner row, and
/// `open level <file>` for the scan taken when the palette opened.
pub fn palette_entries(state: &EditorState) -> Vec<PaletteEntry> {
    let mut out: Vec<PaletteEntry> = KEYMAP
        .iter()
        .filter(|b| b.palette)
        .map(|b| PaletteEntry {
            text: b.label.to_string(),
            chord: b.chord.to_string(),
            action: b.key.map(PaletteAction::Key),
        })
        .collect();
    // Outliner objects → `goto` rows (live view — always current).
    for row in crate::tree::build_tree(state) {
        if let crate::tree::TreeRow::Object { target, text, .. } = row {
            out.push(PaletteEntry {
                text: format!("goto {text}"),
                chord: String::new(),
                action: Some(PaletteAction::Goto(target)),
            });
        }
    }
    // Level files scanned at palette-open time (no per-frame disk I/O).
    if let Some(p) = &state.cmd_palette {
        for f in &p.files {
            out.push(PaletteEntry {
                text: format!("open level {f}"),
                chord: String::new(),
                action: Some(PaletteAction::OpenLevel(f.clone())),
            });
        }
    }
    out
}

/// The candidates MATCHING the live query, best score first (stable for equal
/// scores). An empty query lists everything in table order. THE row list the
/// painter, the key handler, and the mouse hit-test all consume.
pub fn palette_matches(state: &EditorState) -> Vec<PaletteEntry> {
    let query = state.cmd_palette.as_ref().map_or("", |p| p.query.as_str());
    let all = palette_entries(state);
    if query.is_empty() {
        return all;
    }
    let mut scored: Vec<(i32, PaletteEntry)> = all
        .into_iter()
        .filter_map(|e| score(query, &e.text).map(|s| (s, e)))
        .collect();
    scored.sort_by_key(|e| std::cmp::Reverse(e.0));
    scored.into_iter().map(|(_, e)| e).collect()
}

/// Open the `:` command palette (a topmost focus-stack frame). Scans
/// `levels/*.lvl` ONCE here so redraws stay disk-free.
pub fn open_palette(state: &mut EditorState) {
    state.cmd_palette = Some(CmdPaletteState {
        query: String::new(),
        sel: 0,
        files: crate::open::scan_lvl_files(),
    });
    state.status = "palette: type to filter, Enter runs, Esc closes".into();
}

/// Modal key handling while the command palette is open (routed by the focus
/// stack): printable chars type into the query, Backspace deletes, Up/Down move
/// the selection, Enter executes the selected row. Esc closes via
/// `main::quit_on_esc` (never routed here).
pub fn palette_key(state: &mut EditorState, key: EditorKey) {
    let matches_len = palette_matches(state).len();
    let Some(p) = state.cmd_palette.as_mut() else {
        return;
    };
    match key.code {
        E::Char(c) => {
            p.query.push(c);
            p.sel = 0;
        }
        E::Backspace => {
            p.query.pop();
            p.sel = 0;
        }
        E::Up => p.sel = p.sel.saturating_sub(1),
        E::Down => {
            if matches_len > 0 {
                p.sel = (p.sel + 1).min(matches_len - 1);
            }
        }
        E::Enter => execute_selected(state),
        _ => {}
    }
}

/// Execute the SELECTED palette row: close the palette first (so a synthesized
/// key routes through the normal chain, not back into the palette frame), then
/// dispatch through the existing code path for its action kind.
pub fn execute_selected(state: &mut EditorState) {
    let sel = state.cmd_palette.as_ref().map_or(0, |p| p.sel);
    let matches = palette_matches(state);
    state.cmd_palette = None;
    let Some(entry) = matches.get(sel.min(matches.len().saturating_sub(1))) else {
        state.status = "palette: no match".into();
        return;
    };
    match &entry.action {
        Some(PaletteAction::Key(code)) => {
            crate::edit::apply_key(
                state,
                EditorKey {
                    code: *code,
                    shift: false,
                },
            );
        }
        Some(PaletteAction::Goto(target)) => {
            crate::tree::select_target(state, *target);
        }
        Some(PaletteAction::OpenLevel(path)) => {
            // Go through the overlay's dirty guard, never straight to
            // `load_path` — on a dirty document the first Enter only ARMS the
            // confirm (the overlay stays up on this row), so a stray fuzzy hit
            // can never discard unsaved work and wipe the undo history.
            let path = path.clone();
            crate::open::pick_path(state, &path);
        }
        None => {
            state.status = format!("'{}' — press {}", entry.text, entry.chord);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_is_subsequence_only() {
        assert!(score("ren", "Rename object").is_some());
        assert!(
            score("rname", "Rename object").is_some(),
            "subsequence with gaps matches"
        );
        assert!(
            score("xyz", "Rename object").is_none(),
            "missing chars never match"
        );
        assert_eq!(
            score("", "anything"),
            Some(0),
            "empty query matches everything"
        );
    }

    #[test]
    fn score_prefers_word_starts_and_runs() {
        let tight = score("ren", "rename").unwrap();
        let loose = score("ren", "r e n scattered").unwrap();
        assert!(
            tight > loose,
            "consecutive run beats scattered ({tight} vs {loose})"
        );
        let start = score("op", "open level").unwrap();
        let mid = score("op", "loop").unwrap();
        assert!(
            start > mid,
            "word-start match ranks higher ({start} vs {mid})"
        );
    }

    /// DRIFT-PROOF: every palette-eligible binding in the
    /// table appears in BOTH the generated help sheet and the palette
    /// candidates — the three surfaces render from ONE table, so a binding
    /// added only to the table shows up everywhere with no other edit.
    #[test]
    fn every_palette_binding_is_in_help_and_palette() {
        let state = EditorState::new_empty(4, 4, 2);
        let help = help_lines().join("\n");
        let palette = palette_entries(&state);
        for b in KEYMAP.iter().filter(|b| b.palette) {
            assert!(
                help.contains(b.label),
                "help must list table binding '{}'",
                b.label
            );
            assert!(
                help.contains(b.chord),
                "help must show the chord '{}' for '{}'",
                b.chord,
                b.label
            );
            let (want_text, want_chord) = (b.label, b.chord);
            assert!(
                palette
                    .iter()
                    .any(|e| e.text == want_text && e.chord == want_chord),
                "palette must list table binding '{want_text}'"
            );
        }
    }

    /// Every keymap group name is registered in GROUPS (otherwise its rows
    /// would silently vanish from the generated help).
    #[test]
    fn every_binding_group_is_registered() {
        for b in KEYMAP {
            assert!(
                GROUPS.contains(&b.group),
                "binding '{}' uses unregistered group '{}'",
                b.label,
                b.group
            );
        }
    }

    #[test]
    fn hint_line_is_mode_aware() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let root = hint_line(&s);
        assert!(root.contains("?=help"), "root hint keeps ?=help: {root}");
        s.tree.open = true;
        let tree = hint_line(&s);
        assert_ne!(root, tree, "tree focus changes the hint line");
        assert!(
            tree.contains("d=del"),
            "tree hints show its own keys: {tree}"
        );
        s.tree.open = false;
        s.material_mode = true;
        let mat = hint_line(&s);
        assert_ne!(mat, root, "material mode changes the hint line");
        assert!(
            mat.contains("b=field"),
            "material hints its field key: {mat}"
        );
    }

    #[test]
    fn palette_fuzzy_finds_rename_with_chord() {
        let mut s = EditorState::new_empty(8, 8, 2);
        open_palette(&mut s);
        s.cmd_palette.as_mut().unwrap().query = "ren".into();
        let m = palette_matches(&s);
        assert!(!m.is_empty());
        let top = &m[0];
        assert!(
            top.text.to_lowercase().contains("rename"),
            "top match for 'ren' is rename, got '{}'",
            top.text
        );
        assert_eq!(top.chord, "F2", "the palette teaches the chord");
    }

    #[test]
    fn palette_finds_a_node_by_name_and_goto_selects_it() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(2, 3, 0);
        s.level.nodes[0].name = "furnace".into();
        open_palette(&mut s);
        s.cmd_palette.as_mut().unwrap().query = "furnace".into();
        let m = palette_matches(&s);
        assert!(
            m.iter()
                .any(|e| e.text.starts_with("goto") && e.text.contains("furnace")),
            "palette lists the node as a goto row"
        );
        // Enter on the top row (the goto) jumps selection to the node.
        let top_is_goto = m[0].text.contains("furnace");
        assert!(
            top_is_goto,
            "fuzzy puts the name match on top: {:?}",
            m[0].text
        );
        execute_selected(&mut s);
        assert!(s.cmd_palette.is_none(), "execution closes the palette");
        assert_eq!(s.selected_node, Some(0), "goto selected the node");
        assert_eq!(
            (s.cursor.0, s.cursor.1),
            (2, 3),
            "cursor jumped to the node"
        );
    }

    #[test]
    fn palette_lists_open_level_rows_from_open_scan() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.cmd_palette = Some(crate::state::CmdPaletteState {
            query: String::new(),
            sel: 0,
            files: vec!["levels/demo.lvl".into()],
        });
        let all = palette_entries(&s);
        assert!(
            all.iter().any(|e| e.text == "open level levels/demo.lvl"),
            "open-level rows come from the palette's open-time scan"
        );
    }

    #[test]
    fn palette_key_types_and_executes_help() {
        let mut s = EditorState::new_empty(8, 8, 2);
        open_palette(&mut s);
        for c in "help cheat".chars() {
            palette_key(
                &mut s,
                EditorKey {
                    code: E::Char(c),
                    shift: false,
                },
            );
        }
        palette_key(
            &mut s,
            EditorKey {
                code: E::Enter,
                shift: false,
            },
        );
        assert!(s.cmd_palette.is_none(), "Enter closes the palette");
        assert!(s.help_open, "the help action dispatched through apply_key");
    }

    // ── typed-row widget bindings (the table IS the surface) ─────────────────

    /// Every row-editor context has bindings in the table, so the hint bar, the
    /// `?` help sheet and the `:` palette all know those keys exist. A binding
    /// absent from the table effectively does not exist.
    #[test]
    fn every_row_editor_context_has_bindings() {
        for ctx in [Context::RowNum, Context::RowEnum, Context::RowColor] {
            let n = KEYMAP.iter().filter(|b| b.context == ctx).count();
            assert!(n >= 3, "{ctx:?} has only {n} bindings");
            assert!(
                KEYMAP
                    .iter()
                    .any(|b| b.context == ctx && b.chord == "Enter"),
                "{ctx:?} must document its commit key"
            );
            assert!(
                KEYMAP.iter().any(|b| b.context == ctx && b.chord == "Esc"),
                "{ctx:?} must document its cancel key"
            );
        }
    }

    /// The hint bar switches to the row editor's OWN keys while one is open —
    /// the whole point of a mode-aware hint segment.
    #[test]
    fn hint_bar_follows_the_open_row_editor() {
        use crate::state::{EntityField, EntityKind};
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Node);
        crate::entity::place_at_cursor(&mut s);
        let act = crate::chrome::RowAction::FocusEntity(EntityField::NodeThreshold);

        crate::rowedit::open_for(&mut s, act);
        assert_eq!(active_context(&s), Context::RowNum);
        assert!(hint_line(&s).contains("commit"), "hint: {}", hint_line(&s));
        crate::rowedit::cancel(&mut s);

        crate::rowedit::open_for(
            &mut s,
            crate::chrome::RowAction::FocusEntity(EntityField::NodeIo),
        );
        assert_eq!(active_context(&s), Context::RowEnum);
        crate::rowedit::cancel(&mut s);

        crate::rowedit::open_for(
            &mut s,
            crate::chrome::RowAction::FocusEntity(EntityField::NodeColor),
        );
        assert_eq!(active_context(&s), Context::RowColor);
        assert!(hint_line(&s).contains("hex"), "hint: {}", hint_line(&s));
        crate::rowedit::cancel(&mut s);
        assert_eq!(
            active_context(&s),
            Context::Object,
            "back to the object context"
        );
    }

    // ── MODAL DISPATCH SCANS (generalised to every modal) ────────────────────
    //
    // `every_palette_binding_is_in_help_and_palette` proves table → surfaces.
    // The scans below prove the OTHER direction, dispatch → table: a key
    // handled in a modal but absent from `KEYMAP` would be invisible in the
    // hint bar, the `?` sheet and the `:` palette, and no other test would
    // notice. The table below closes that gap for EVERY fenced modal
    // with one shared extractor and one loop.

    /// The two editor sources carrying sentinel-fenced modal dispatch, read at
    /// COMPILE time so the scans see the code that actually ships.
    const EDIT_RS: &str = include_str!("edit.rs");
    const RENAME_RS: &str = include_str!("rename.rs");

    /// ONE sentinel-fenced dispatch block: the context that owns it, the source
    /// it lives in, its fence tag, and the MINIMUM number of distinct chords the
    /// scan must find between the sentinels.
    ///
    /// The floor is PER CONTEXT on purpose: a shared total would let one fence
    /// stop matching (a renamed sentinel, a moved body) and still pass on the
    /// strength of the others, which is exactly the silent no-op the sentinels
    /// exist to prevent.
    struct DispatchFence {
        /// The keymap context whose rows must cover this block's keys.
        context: Context,
        /// Source file name — for the failure message only.
        file: &'static str,
        /// That file's text (`include_str!`).
        src: &'static str,
        /// Fence tag: the block is `// <tag>-KEYS-BEGIN` … `// <tag>-KEYS-END`.
        tag: &'static str,
        /// Minimum distinct chords the scan must find (see above).
        floor: usize,
    }

    /// EVERY modal dispatch proven against the table. Adding a modal means
    /// adding a fence here plus the sentinels in its handler — nothing else.
    static FENCES: &[DispatchFence] = &[
        DispatchFence {
            context: Context::Settings,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "SETTINGS",
            floor: 16,
        },
        DispatchFence {
            context: Context::Mask,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "MASK",
            floor: 9,
        },
        DispatchFence {
            context: Context::Wizard,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "WIZARD",
            floor: 6,
        },
        DispatchFence {
            context: Context::ObjectPalette,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "OBJPALETTE",
            floor: 4,
        },
        DispatchFence {
            context: Context::DirPicker,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "DIRPICKER",
            floor: 12,
        },
        DispatchFence {
            context: Context::Open,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "OPEN",
            floor: 4,
        },
        DispatchFence {
            context: Context::SaveAs,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "SAVEAS",
            floor: 2,
        },
        DispatchFence {
            context: Context::AutosaveOffer,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "AUTOSAVE",
            floor: 4,
        },
        DispatchFence {
            context: Context::Rename,
            file: "rename.rs",
            src: RENAME_RS,
            tag: "RENAME",
            floor: 4,
        },
        DispatchFence {
            context: Context::Help,
            file: "edit.rs",
            src: EDIT_RS,
            tag: "HELP",
            floor: 8,
        },
    ];

    /// The text BETWEEN one fence's sentinels.
    ///
    /// PANICS (fails the test) when either marker is missing, so deleting a
    /// sentinel cannot quietly turn a drift check into a no-op. Each fence is
    /// deliberately narrow — exactly one handler body — so an edit elsewhere in
    /// the file can neither widen nor narrow what is scanned.
    fn fenced_dispatch_src(f: &DispatchFence) -> &'static str {
        let begin = format!("// {}-KEYS-BEGIN", f.tag);
        let end = format!("// {}-KEYS-END", f.tag);
        let b = f.src.find(&begin).unwrap_or_else(|| {
            panic!(
                "{} must carry the {} BEGIN sentinel that fences the {:?} dispatch",
                f.file, f.tag, f.context
            )
        });
        let e = f.src.find(&end).unwrap_or_else(|| {
            panic!(
                "{} must carry the {} END sentinel that fences the {:?} dispatch",
                f.file, f.tag, f.context
            )
        });
        assert!(
            b < e,
            "the {} dispatch sentinels are out of order in {}",
            f.tag,
            f.file
        );
        &f.src[b + begin.len()..e]
    }

    /// Every `EditorKeyCode` a fenced block matches on, as the CHORD SPELLING
    /// the keymap table uses. `E::Char('a')` → `"a"`, `E::Char(' ')` → `"Space"`,
    /// `E::Tab` → `"Tab"`, and so on. Char literals inside match GUARDS (e.g.
    /// `c != 'a'`) count too — a guard that names a key is still a key the
    /// dispatch reacts to.
    ///
    /// Keys matched only through a BINDING with a predicate (`E::Char(c) if
    /// c.is_ascii_digit()`, or a bare `E::Char(c)` that types) carry no literal
    /// and so are not extracted; the table documents those ranges on its own
    /// (`0-9`, `a-z 0-9`).
    fn dispatched_chords(src: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        // Named (non-`Char`) codes: substring matches of `E::<Name>`.
        // `Home`, `End` and `Delete` joined the list when the help sheet started
        // dispatching Home/End: a named code the extractor does not know is a
        // key the scan silently forgives, which is the exact hole the scan
        // exists to close. No other fenced block dispatches them today, so the
        // additions cost nothing and cover the next block that does.
        for name in [
            "Tab",
            "PageUp",
            "PageDown",
            "Up",
            "Down",
            "Left",
            "Right",
            "Enter",
            "Backspace",
            "Esc",
            "Home",
            "End",
            "Delete",
        ] {
            if src.contains(&format!("E::{name}")) {
                out.push(name.to_string());
            }
        }
        // `E::Char('c')` and any `'c'` literal appearing in a guard.
        let bytes: Vec<char> = src.chars().collect();
        let mut i = 0usize;
        while i + 2 < bytes.len() {
            if bytes[i] == '\'' && bytes[i + 2] == '\'' {
                let c = bytes[i + 1];
                let chord = if c == ' ' {
                    "Space".to_string()
                } else {
                    c.to_string()
                };
                if !out.contains(&chord) {
                    out.push(chord);
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every chord spelling the rows of `ctx` cover. A row's chord may name
    /// several keys separated by `/` (`"Left/-"` covers both `Left` and `-`), so
    /// the set is the union of the split parts.
    ///
    /// The separator can also BE the key. `"/"` (the outliner's filter chord)
    /// splits into two empty strings and would cover nothing — so a fenced
    /// dispatch that handles `/` would be silently forgiven by the very scan
    /// that exists to catch it. A chord with no non-empty part is therefore
    /// taken literally. Inert today (`Context::Tree` is not fenced yet); it is
    /// here so it cannot become a trap the day it is.
    fn keymap_chords(ctx: Context) -> Vec<String> {
        KEYMAP
            .iter()
            .filter(|b| b.context == ctx)
            .flat_map(|b| {
                let parts: Vec<&str> = b.chord.split('/').filter(|p| !p.is_empty()).collect();
                if parts.is_empty() {
                    vec![b.chord]
                } else {
                    parts
                }
            })
            .map(str::to_string)
            .collect()
    }

    /// A chord that IS the separator covers ITSELF, not nothing.
    #[test]
    fn a_slash_chord_covers_the_slash_key() {
        let tree = keymap_chords(Context::Tree);
        assert!(
            tree.contains(&"/".to_string()),
            "the outliner's `/` filter row must cover `/`, else a fenced Tree \
             dispatch would forgive it: {tree:?}"
        );
        assert!(
            !tree.iter().any(String::is_empty),
            "and no empty spelling survives: {tree:?}"
        );
    }

    /// Assert ONE fence: the sentinels still fence a real dispatch (the floor),
    /// and every key literal inside has a row in its context.
    fn assert_fence_is_covered(f: &DispatchFence) {
        let dispatched = dispatched_chords(fenced_dispatch_src(f));
        assert!(
            dispatched.len() >= f.floor,
            "the {:?} scan found only {} chord(s) {:?} in {} (floor {}) — the \
             sentinels probably no longer fence the dispatch block",
            f.context,
            dispatched.len(),
            dispatched,
            f.file,
            f.floor
        );
        let covered = keymap_chords(f.context);
        for chord in &dispatched {
            assert!(
                covered.contains(chord),
                "{} dispatches '{chord}' in the {:?} modal but KEYMAP has no \
                 Context::{:?} binding for it — it would be invisible in the hint \
                 bar, the ? sheet and the : palette.\n  dispatched: {:?}\n  in \
                 KEYMAP:  {:?}",
                f.file,
                f.context,
                f.context,
                dispatched,
                covered
            );
        }
    }

    /// THE DRIFT TEST IN THE OTHER DIRECTION, for EVERY fenced modal.
    /// A real invariant over the shipped source, not a hand-copied list.
    #[test]
    fn every_dispatched_key_is_in_the_keymap() {
        for f in FENCES {
            assert_fence_is_covered(f);
        }
    }

    /// The original LEVEL SETTINGS coverage check, kept BY NAME so the
    /// coverage it bought cannot be lost in a refactor: the LEVEL SETTINGS
    /// dispatch specifically.
    #[test]
    fn every_settings_key_is_in_the_keymap() {
        let f = FENCES
            .iter()
            .find(|f| f.context == Context::Settings)
            .expect("the settings dispatch is still fenced and scanned");
        assert_fence_is_covered(f);
    }

    /// One fence per context, one tag per fence — a duplicate tag would make two
    /// contexts scan the SAME block (and the second one's real keys go unproven).
    #[test]
    fn every_fence_is_unique() {
        for (i, f) in FENCES.iter().enumerate() {
            for g in &FENCES[i + 1..] {
                assert_ne!(f.tag, g.tag, "two fences share the tag '{}'", f.tag);
                assert_ne!(
                    f.context, g.context,
                    "two fences claim {:?} — one context, one dispatch block",
                    f.context
                );
            }
        }
    }

    /// Every fenced modal's rows reach the generated `?` help sheet, tagged with
    /// their own context so overloads (`x`, `q`, `i`, `D`, `Enter`) read
    /// unambiguously. The generalisation of
    /// `settings_bindings_reach_the_help_sheet`.
    #[test]
    fn every_fenced_context_reaches_the_help_sheet() {
        let help = help_lines().join("\n");
        for f in FENCES {
            let rows: Vec<&Binding> = KEYMAP.iter().filter(|b| b.context == f.context).collect();
            assert!(
                !rows.is_empty(),
                "{:?} is scanned but has no KEYMAP rows at all",
                f.context
            );
            for b in rows {
                assert!(
                    help.contains(b.label),
                    "help must list the {:?} row '{}'",
                    f.context,
                    b.label
                );
            }
            let tag = format!("[{}]", f.context.label());
            assert!(
                help.contains(&tag),
                "{:?} rows must carry their context tag {tag} in the help sheet",
                f.context
            );
        }
    }

    /// The settings rows reach the generated `?` help sheet, tagged with their
    /// own context so the overloads (`a`, `x`, `n`, `r`) read unambiguously.
    /// (The original check, kept verbatim beside its generalisation above.)
    #[test]
    fn settings_bindings_reach_the_help_sheet() {
        let help = help_lines().join(
            "
",
        );
        assert!(
            help.contains("-- level settings --"),
            "the group has a banner"
        );
        for b in KEYMAP.iter().filter(|b| b.context == Context::Settings) {
            assert!(help.contains(b.label), "help must list '{}'", b.label);
        }
        assert!(
            help.contains("[settings]"),
            "settings rows carry their context tag in the help sheet"
        );
    }

    /// The hint bar switches to the settings modal's OWN keys while it is open —
    /// the mirror of `hint_bar_follows_the_open_row_editor`.
    #[test]
    fn hint_bar_follows_the_open_settings_modal() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let root = hint_line(&s);
        assert_eq!(active_context(&s), Context::Root);

        crate::settings::open(&mut s);
        assert_eq!(
            active_context(&s),
            Context::Settings,
            "the modal owns the context"
        );
        let modal = hint_line(&s);
        assert_ne!(modal, root, "opening the form changes the hint line");
        for want in ["Tab=field", "a=add", "x=del", "R=resize", "Enter=apply"] {
            assert!(
                modal.contains(want),
                "settings hint missing '{want}': {modal}"
            );
        }

        // It outranks the tool contexts below it: an open form still hints ITS keys.
        s.tree.open = true;
        assert_eq!(
            active_context(&s),
            Context::Settings,
            "modal outranks the outliner"
        );
        s.tree.open = false;

        s.settings = None;
        assert_eq!(
            active_context(&s),
            Context::Root,
            "closing restores the root context"
        );
    }

    // ── modals with no discoverability variant would leave the hint bar
    // showing the wrong keys ─────────────────────────────────────────────────
    //
    // Each test below opens ONE modal on a real `EditorState` and asserts (a)
    // `active_context` names it and (b) `hint_line` advertises that modal's own
    // key. Without a `Context` variant, a modal falls through to the context
    // UNDERNEATH, so the bar would teach keys that do nothing.

    /// The CRASH-RECOVERY offer is the frontmost modal in `edit::apply_key`.
    #[test]
    fn hint_bar_follows_the_autosave_offer() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.autosave_offer = Some("levels/demo.lvl.autosave".into());
        assert_eq!(active_context(&s), Context::AutosaveOffer);
        let h = hint_line(&s);
        assert!(h.contains("y/Y=restore"), "recovery hint: {h}");
        assert!(h.contains("n/N=discard"), "recovery hint: {h}");
        // Nothing underneath outranks it — it answers first or not at all.
        s.tree.open = true;
        assert_eq!(active_context(&s), Context::AutosaveOffer);
        s.autosave_offer = None;
        assert_eq!(
            active_context(&s),
            Context::Tree,
            "closing hands focus back"
        );
    }

    /// The F2 RENAME prompt (routed through the focus stack).
    #[test]
    fn hint_bar_follows_the_rename_prompt() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(1, 1, 0);
        s.place = Some(crate::state::EntityKind::Node);
        s.selected_node = Some(0);
        assert_eq!(
            active_context(&s),
            Context::Object,
            "the surface underneath"
        );
        crate::rename::open(&mut s);
        assert!(s.rename.is_some(), "the prompt opened");
        assert_eq!(active_context(&s), Context::Rename);
        let h = hint_line(&s);
        assert!(h.contains("Enter=apply"), "rename hint: {h}");
        assert!(h.contains("Backspace=del"), "rename hint: {h}");
    }

    /// The emitter 3x3x3 DIRECTION PICKER.
    #[test]
    fn hint_bar_follows_the_direction_picker() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.dir_picker = Some(crate::picker::DirPickerState {
            emitter: 0,
            cx: 0,
            cy: 0,
            cz: 0,
        });
        assert_eq!(active_context(&s), Context::DirPicker);
        let h = hint_line(&s);
        assert!(h.contains("D=close"), "picker hint: {h}");
        assert!(h.contains("pick"), "picker hint: {h}");
    }

    /// The LOAD/OPEN overlay.
    #[test]
    fn hint_bar_follows_the_open_overlay() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open = Some(crate::state::OpenState {
            files: vec!["levels/demo.lvl".into()],
            cursor: 0,
        });
        assert_eq!(active_context(&s), Context::Open);
        let h = hint_line(&s);
        assert!(h.contains("O=close"), "open hint: {h}");
        assert!(h.contains("Enter=load"), "open hint: {h}");
    }

    /// The SAVE-AS filename prompt.
    #[test]
    fn hint_bar_follows_the_saveas_prompt() {
        let mut s = EditorState::new_empty(8, 8, 2);
        crate::saveas::open_saveas(&mut s);
        assert!(s.saveas.is_some(), "the prompt opened");
        assert_eq!(active_context(&s), Context::SaveAs);
        let h = hint_line(&s);
        assert!(h.contains("Enter=save"), "save-as hint: {h}");
        assert!(h.contains("Backspace=del"), "save-as hint: {h}");
    }

    /// The NEW-LEVEL wizard.
    #[test]
    fn hint_bar_follows_the_new_level_wizard() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.wizard = Some(crate::wizard::Wizard::default());
        assert_eq!(active_context(&s), Context::Wizard);
        let h = hint_line(&s);
        assert!(h.contains("Tab=field"), "wizard hint: {h}");
        assert!(h.contains("Enter=create"), "wizard hint: {h}");
    }

    /// The object-TYPE CREATE palette (`i`) — not the `:` command palette.
    #[test]
    fn hint_bar_follows_the_create_palette() {
        let mut s = EditorState::new_empty(8, 8, 2);
        crate::tree::toggle_palette(&mut s);
        assert!(s.palette.is_some(), "the create palette opened");
        assert_eq!(active_context(&s), Context::ObjectPalette);
        let h = hint_line(&s);
        assert!(h.contains("i=close"), "create hint: {h}");
        assert!(h.contains("Enter=pick"), "create hint: {h}");
    }

    /// The PART MASK modal's `Context` must have a row for all 9 keys its
    /// dispatch handles, or its hint bar would name only a few of them.
    #[test]
    fn hint_bar_follows_the_mask_modal() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.mask_mode = true;
        assert_eq!(active_context(&s), Context::Mask);
        let h = hint_line(&s);
        for want in ["=move", "[/]=layer", "Space=toggle", "x=clear", "q=exit"] {
            assert!(h.contains(want), "mask hint missing '{want}': {h}");
        }
    }

    /// THE PRECEDENCE LADDER: with several surfaces open at once the hint bar
    /// must name the one that will actually eat the next key. Each rung opens
    /// one MORE surface and asserts the context flips to it, walking
    /// `edit::apply_key`'s real resolution order backwards.
    #[test]
    fn the_hint_context_mirrors_the_dispatch_precedence() {
        let mut s = EditorState::new_empty(8, 8, 2);
        assert_eq!(active_context(&s), Context::Root);

        crate::tree::toggle_palette(&mut s); // `i` — the create palette
        assert_eq!(active_context(&s), Context::ObjectPalette, "create > root");

        crate::settings::open(&mut s);
        assert_eq!(active_context(&s), Context::Settings, "settings > create");

        s.wizard = Some(crate::wizard::Wizard::default());
        assert_eq!(active_context(&s), Context::Wizard, "wizard > settings");

        crate::saveas::open_saveas(&mut s);
        assert_eq!(active_context(&s), Context::SaveAs, "save-as > wizard");

        s.open = Some(crate::state::OpenState {
            files: Vec::new(),
            cursor: 0,
        });
        assert_eq!(active_context(&s), Context::Open, "open > save-as");

        s.validation_open = true;
        assert_eq!(
            active_context(&s),
            Context::Validation,
            "the focus stack > open"
        );

        s.dir_picker = Some(crate::picker::DirPickerState {
            emitter: 0,
            cx: 0,
            cy: 0,
            cz: 0,
        });
        assert_eq!(
            active_context(&s),
            Context::DirPicker,
            "dir-picker > validation"
        );

        s.open_help();
        assert_eq!(active_context(&s), Context::Help, "help > dir-picker");

        s.rename = Some(crate::rename::RenameState {
            target: crate::rename::RenameTarget::Node(0),
            buf: "n".into(),
        });
        assert_eq!(active_context(&s), Context::Rename, "rename > dir-picker");

        s.cmd_palette = Some(CmdPaletteState {
            query: String::new(),
            sel: 0,
            files: Vec::new(),
        });
        assert_eq!(
            active_context(&s),
            Context::CmdPalette,
            "the : palette > rename"
        );

        s.mask_mode = true;
        assert_eq!(active_context(&s), Context::Mask, "mask > the focus stack");

        s.autosave_offer = Some("levels/demo.lvl.autosave".into());
        assert_eq!(
            active_context(&s),
            Context::AutosaveOffer,
            "recovery > mask"
        );

        s.preview_captured = true;
        assert_eq!(
            active_context(&s),
            Context::Preview,
            "free-look capture > all"
        );
    }

    /// The FILTER keeps a row on any of the three things a reader might type:
    /// its chord, its label, or its group name.
    #[test]
    fn the_help_filter_matches_chord_label_or_group() {
        // Group: every row of "level settings" survives its own group name.
        let by_group = help_lines_filtered("level settings");
        for b in KEYMAP.iter().filter(|b| b.group == "level settings") {
            assert!(
                by_group.iter().any(|l| l.contains(b.label)),
                "the group query keeps '{}'",
                b.label
            );
        }
        // Label: a word from one label finds that row.
        assert!(
            help_lines_filtered("staircase")
                .iter()
                .any(|l| l.contains("staircase stamp")),
            "a label word finds its row"
        );
        // Chord: the chord column is searchable too (`F5` is a chord, not a word
        // anyone would find by label).
        assert!(
            help_lines_filtered("F5")
                .iter()
                .any(|l| l.starts_with("F5 ")),
            "a chord finds its row"
        );
        // Case-insensitive, both ways.
        assert_eq!(help_lines_filtered("VOX"), help_lines_filtered("vox"));
        // A query that matches nothing yields an EMPTY sheet, not the whole one
        // — the failure mode that makes a filter look broken.
        assert!(
            help_lines_filtered("zzzznotakey").is_empty(),
            "no matches means no rows"
        );
    }

    /// A `-- group --` banner never survives its rows: a heading over nothing is
    /// a row of the sheet that lies about having content under it.
    #[test]
    fn a_filtered_banner_always_has_a_row_under_it() {
        for q in ["mask", "vox", "colour", "e", "save"] {
            let lines = help_lines_filtered(q);
            for (i, l) in lines.iter().enumerate() {
                if l.starts_with("-- ") {
                    let next = lines.get(i + 1);
                    assert!(
                        next.is_some_and(|n| !n.starts_with("-- ")),
                        "query {q:?}: banner {l:?} has no rows under it"
                    );
                }
            }
            // The banner also carries the surviving count while filtering.
            if let Some(b) = lines.iter().find(|l| l.starts_with("-- ")) {
                assert!(b.contains('('), "query {q:?}: banner {b:?} shows its count");
            }
        }
    }

    /// The UNFILTERED sheet is byte-identical to the sheet before the filter
    /// existed: every group banner, every row, and the closing line.
    #[test]
    fn the_unfiltered_sheet_is_the_whole_table() {
        let lines = help_lines();
        assert_eq!(lines, help_lines_filtered(""), "no query = no filtering");
        assert_eq!(lines, help_lines_filtered("   "), "whitespace is no query");
        for &g in GROUPS {
            assert!(
                lines.iter().any(|l| l == &format!("-- {g} --")),
                "banner for {g}"
            );
        }
        for b in KEYMAP {
            assert!(
                lines.iter().any(|l| l.contains(b.label)),
                "row for '{}'",
                b.label
            );
        }
        assert_eq!(
            lines.last().map(String::as_str),
            Some("press ? or Esc to close")
        );
        assert_eq!(help_match_count(""), KEYMAP.len(), "the count is every row");
    }

    /// The help sheet's OWN keys are in the table, so the hint bar teaches them
    /// while the sheet is up. Before `Context::Help` existed the bar showed the
    /// keys of whatever surface was UNDERNEATH the popup.
    #[test]
    fn the_hint_bar_teaches_the_help_sheets_own_keys() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let root = hint_line(&s);
        s.open_help();
        let help = hint_line(&s);
        assert_ne!(root, help, "the hint bar follows the open sheet");
        assert!(help.contains("scroll"), "it advertises scrolling: {help}");
        assert!(help.contains("filter"), "and filtering: {help}");
        assert!(help.contains("close"), "and how to leave: {help}");
    }

    /// The `Enter` root binding reaches the help sheet AND the palette (the
    /// drift invariant, asserted for the one new palette-eligible action).
    #[test]
    fn the_enter_row_binding_is_discoverable() {
        let s = EditorState::new_empty(4, 4, 2);
        let b = KEYMAP
            .iter()
            .find(|b| b.context == Context::Root && b.chord == "Enter")
            .expect("Enter is in the table");
        assert!(b.palette, "Enter is palette-eligible");
        assert!(
            help_lines()
                .join(
                    "
"
                )
                .contains(b.label),
            "help lists it"
        );
        assert!(
            palette_entries(&s).iter().any(|e| e.text == b.label),
            "the : palette lists it"
        );
    }
}
