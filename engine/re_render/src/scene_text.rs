//! `scene_text` — the PER-LEVEL text-surface registry: the owned wall panels and
//! the pipe word pools the shade path reads.
//!
//! ## What replaced what, and why
//!
//! The code wall used to resolve its panel through a process-global
//! `OnceLock<fn(u32) -> &'static CodePanel>`: the host registered a function
//! handing out `'static` panels. That works only for text embedded at compile time
//! (`include_str!` of our own source). Per-level `[note]` placard bodies are
//! `String`s parsed at load, so they can never be `&'static` without either
//! `Box::leak`-per-level-load (a real, unbounded leak — one set per level
//! transition) or this: a registry of OWNED panels, replaced wholesale when a level
//! loads, so the previous level's panels are simply DROPPED.
//!
//! [`TextSurfaceRegistry`] is that registry. It has two index spaces, deliberately
//! separate:
//!
//! * **code panels** — the embedded-source panels a `CodeWall { panel: N }` recipe
//!   selects. `N` keeps meaning exactly what it meant before (the host registers
//!   them in the same order), so `09-code-wall.lvl` is unaffected by the move.
//! * **note panels** — one per `[note]` in the loaded level, in parse order, so a
//!   `NoteWall { note: N }` recipe means "the N-th placard of this level" with no
//!   offset arithmetic shared between the game and the renderer.
//!
//! It also carries the level's [`PoolTable`] (pipe tag → owning node's words). That
//! is text too, and it has the same lifecycle — built at load, dropped at teardown
//! — so it rides the same publication point instead of inventing a second one.
//!
//! ## How the shade path reaches it (and why it is not a threaded `&`)
//!
//! The two consumers sit at very different depths:
//!
//! * The pipe-word lookup happens in the crate's `shade_cell`, which could take a
//!   borrow as a parameter.
//! * The wall-panel lookup happens inside `Material::sample`, whose only per-cell
//!   input is `ascii_shader::SampleCtx` — a `Copy`, lifetime-free struct that is
//!   also a cache KEY (`WorldTexCache`) and is stored in the per-frame sample memo.
//!   Giving it a lifetime is a 13k-line refactor of the shader and would make the
//!   cross-frame texture cache un-keyable.
//!
//! So the registry is published for the process and read back through a borrow
//! obtained INSIDE the access ([`sample_code_panel`], [`sample_note_panel`],
//! [`with_pipe_pool`] — the last of which hands the caller a real `&NodePool` borrow
//! for its closure body, which is exactly the shape `PipeTextCtx<'a>` wants). This
//! is the same shape the crate already uses for cast-scoped data it cannot thread
//! (`crate::state_bus` for the state map, `crate::config` for the render knobs), and
//! it is deliberately an `RwLock<Arc<…>>` rather than a `thread_local!`: the
//! registry is installed by the LEVEL-LOAD system and read by the RENDER system,
//! which Bevy may schedule on different threads — a thread-local would silently
//! read empty there.
//!
//! ## Cost
//!
//! * Install: once per level load (one `Arc` swap; the old registry drops).
//! * Read: an uncontended `RwLock` read guard + two slice index reads, and ONLY on
//!   a code-wall / note-wall / powered-pipe cell. An ordinary wall never touches
//!   this module, so the frame budget for every existing level is unchanged.
//! * No allocation on any read path.

use std::sync::{Arc, PoisonError, RwLock};

use re_core::{NodePool, PoolTable};

use crate::code_wall::{CodeGlyph, TextPanel, NOTE_COLS, NOTE_ROWS, PANEL_COLS, PANEL_ROWS};
use crate::note_style::NoteRegister;
use crate::pipe_text::PoolRegister;
use crate::stencil_variants::{self, ParsedNote, StencilVariants};

/// One registered CODE panel. `code_wall`'s `sample_code_panel*` family reads only
/// the grid itself (`panel.rows()`, not a cached `used_rows`) and the code recipe has
/// its own fixed palette (no per-panel register), so — unlike a NOTE panel
/// ([`RegisteredNote`]) — there is nothing else to cache here.
#[derive(Debug)]
struct Registered {
    panel: TextPanel,
}

/// What a NOTE panel index resolves to: the historical ONE static panel, a live,
/// sample-time-selected [`StencilVariants`] set (state-swapped art / authored LOD
/// tiers — `crate::stencil_variants`), or a live, TIME-selected rotation of panels
/// (a `[note banner] cycle = …` placard — `crate::banner::build_cycle_panels`). A
/// CODE panel is always [`Simple`](NoteEntry::Simple); only a `[note stencil]`/
/// `[note banner]` body can produce [`Variants`](NoteEntry::Variants) or
/// [`Cycled`](NoteEntry::Cycled).
#[derive(Debug)]
enum NoteEntry {
    Simple { panel: TextPanel, used_rows: usize },
    Variants(StencilVariants),
    Cycled(Vec<BannerFrame>),
}

/// One precomputed panel plus its cached `used_rows` (mirrors [`Registered`] /
/// `stencil_variants::TierPanel` — recomputing `used_rows()` per SAMPLE would be an
/// O(cols·rows) scan on every rendered cell). Used both as ONE frame of a
/// non-tiered rotation-less panel (historically) and now as ONE SIZE TIER of a
/// [`BannerFrame`] — the field names stay panel/used_rows either way.
#[derive(Debug)]
struct CyclePanel {
    panel: TextPanel,
    used_rows: usize,
}

impl CyclePanel {
    fn new(panel: TextPanel) -> CyclePanel {
        let used_rows = panel.used_rows();
        CyclePanel { panel, used_rows }
    }
}

/// One WORD of a `[note banner]` rotation, in its three size tiers (the second
/// attempt at the legibility fix): [`select_banner_tier`]
/// picks among them by the MEASURED on-screen face size, so a word too small to show
/// its `full` pixel-font raster at 1:1 falls back to `compact` (bare characters) and
/// then `accent` (a single mark) rather than minifying into noise.
#[derive(Debug)]
struct BannerFrame {
    full: CyclePanel,
    compact: CyclePanel,
    accent: CyclePanel,
}

/// A registered NOTE panel: its content ([`NoteEntry`]) plus the register it speaks
/// in. A separate type from [`Registered`] (rather than widening that one) because
/// `code` panels never carry variants and gain nothing from the extra enum indirection.
#[derive(Debug)]
struct RegisteredNote {
    entry: NoteEntry,
    register: NoteRegister,
    /// The note's authored `colors = GLYPH:#hex …` key (glyph, RGB), empty unless a
    /// `[note stencil]`/`[note banner]` body authored one — see
    /// `crate::stencil_variants`'s module docs, "Colour key" section, for why this
    /// side table exists instead of a field on `CodeGlyph`. Consulted per-sample by
    /// [`sample_note_panel_anchored`] to fill [`NoteSample::color_override`].
    color_key: Vec<stencil_variants::ColorKeyEntry>,
}

/// The per-level set of text surfaces + word pools.
///
/// Built by the host at level load, published via [`install`], read by the shade
/// path. Owns everything it holds, so replacing it frees the previous level's text.
#[derive(Debug, Default)]
pub struct TextSurfaceRegistry {
    /// Embedded-source panels, addressed by a `CodeWall { panel }` index.
    code: Vec<Registered>,
    /// `[note]` placard panels, addressed by a `NoteWall { note }` index, in the
    /// level's parse order.
    notes: Vec<RegisteredNote>,
    /// This level's pipe-tag → owning-node word pools.
    pools: PoolTable,
    /// This level's authored `[effects] glitch` density `[0,1]` — the narrative decay
    /// scalar. It rides the text registry because what it selects IS
    /// text: above [`POOL_ALT_GLITCH_THRESHOLD`](crate::pipe_text::POOL_ALT_GLITCH_THRESHOLD) a powered pipe streams the owning
    /// node's corrupted `pool_alt` register instead of its clean `pool`. `0.0` (the
    /// default, and every level that authors no glitch) ⇒ the clean register.
    glitch: f32,
}

impl TextSurfaceRegistry {
    /// An empty registry: no panels, no pools. `const` so it can back a static
    /// default with no allocation — the "no host content" state in which code walls
    /// draw blank and pipes fall back to their built-in word list (exactly the
    /// behaviour of an unregistered provider before this module existed).
    pub const fn new() -> TextSurfaceRegistry {
        TextSurfaceRegistry {
            code: Vec::new(),
            notes: Vec::new(),
            pools: PoolTable::new(),
            glitch: 0.0,
        }
    }

    /// Register a CODE panel; returns its index (the value a `CodeWall { panel }`
    /// recipe must carry). Registration order defines the index space.
    pub fn push_code_panel(&mut self, panel: TextPanel) -> u32 {
        self.code.push(Registered { panel });
        (self.code.len() - 1) as u32
    }

    /// Register a NOTE panel in the [`NoteRegister::Plain`] register; returns its
    /// index (the value a `NoteWall { note }` recipe must carry). Push notes in the
    /// level's parse order and the index IS the note index.
    pub fn push_note_panel(&mut self, panel: TextPanel) -> u32 {
        self.push_note_panel_styled(panel, NoteRegister::Plain)
    }

    /// Register a NOTE panel that speaks in `register` — the shader inks it with that
    /// register's palette. (The register's LAYOUT is already baked into `panel`; see
    /// [`push_note_text_styled`](Self::push_note_text_styled).)
    pub fn push_note_panel_styled(&mut self, panel: TextPanel, register: NoteRegister) -> u32 {
        let used_rows = panel.used_rows();
        self.notes.push(RegisteredNote {
            entry: NoteEntry::Simple { panel, used_rows },
            register,
            color_key: Vec::new(),
        });
        (self.notes.len() - 1) as u32
    }

    /// Register a NOTE as a live [`StencilVariants`] set (state-swapped art / LOD
    /// tiers) rather than one static panel — see [`push_note_text_styled`]'s Stencil
    /// branch, the only caller.
    fn push_note_variants_styled(
        &mut self,
        variants: StencilVariants,
        register: NoteRegister,
    ) -> u32 {
        let color_key = variants.colors().to_vec();
        self.notes.push(RegisteredNote {
            entry: NoteEntry::Variants(variants),
            register,
            color_key,
        });
        (self.notes.len() - 1) as u32
    }

    /// Register a NOTE as a live, TIME-and-SIZE-selected panel rotation (a
    /// `[note banner]` placard) — see [`push_note_text_styled`]'s Banner branch, the
    /// only caller. `words` is never empty for a value produced by
    /// `crate::banner::build_cycle_panels`, but an empty `Vec` is accepted (renders
    /// blank at sample time, see [`select_cycle_frame`]) rather than panicking here.
    fn push_note_cycle_styled(
        &mut self,
        words: Vec<crate::banner::BannerWordTiers>,
        register: NoteRegister,
    ) -> u32 {
        let frames = words
            .into_iter()
            .map(|w| BannerFrame {
                full: CyclePanel::new(w.full),
                compact: CyclePanel::new(w.compact),
                accent: CyclePanel::new(w.accent),
            })
            .collect();
        self.notes.push(RegisteredNote {
            entry: NoteEntry::Cycled(frames),
            register,
            color_key: Vec::new(),
        });
        (self.notes.len() - 1) as u32
    }

    /// Build + register a code panel from raw source text, using the standard code
    /// grid ([`PANEL_COLS`] × [`PANEL_ROWS`]).
    pub fn push_code_source(&mut self, label: &str, src: &str) -> u32 {
        self.push_code_panel(TextPanel::build(label, src, PANEL_COLS, PANEL_ROWS))
    }

    /// Build + register a note panel from a placard body, word-wrapped onto the
    /// standard note grid ([`NOTE_COLS`] × [`NOTE_ROWS`]), styled by the register the
    /// panel's LABEL declares ([`NoteRegister::from_label`] — the game layer's label
    /// is `note{i}:{NoteKind:?}`).
    ///
    /// A label with no recognised register suffix yields [`NoteRegister::Plain`], i.e.
    /// the pre-styling monochrome plate — so an unlabelled host renders exactly as it
    /// did before. Prefer [`push_note_text_styled`](Self::push_note_text_styled) when
    /// the register is known as a value.
    pub fn push_note_text(&mut self, label: &str, body: &str) -> u32 {
        self.push_note_text_styled(label, body, NoteRegister::from_label(label))
    }

    /// Build + register a note panel in an explicit [`NoteRegister`]: the register's
    /// body transform and line layout are baked into the grid here (once, at load),
    /// and the register is recorded so the shader can ink it.
    ///
    /// [`NoteRegister::Stencil`] takes a DETOUR through `crate::stencil_variants`
    /// first (see that module's docs for the full `bind =` / `colors =` / `--- … ---`
    /// grammar): a body with none of that syntax parses to
    /// [`stencil_variants::ParsedNote::Legacy`] and falls straight
    /// through to the historical single-verbatim-panel path below, so every stencil
    /// placard authored before this module existed is byte-identical.
    pub fn push_note_text_styled(
        &mut self,
        label: &str,
        body: &str,
        register: NoteRegister,
    ) -> u32 {
        if register == NoteRegister::Stencil {
            return match stencil_variants::parse(label, body) {
                ParsedNote::Legacy {
                    body: plain,
                    accent,
                    colors,
                } => {
                    // `register` is `Stencil` in this branch, so `.body()` is exactly
                    // the dedent `NoteRegister::Stencil.body` has always applied here.
                    let styled = register.body(&plain);
                    let mut panel =
                        TextPanel::build_verbatim(label, styled.as_ref(), NOTE_COLS, NOTE_ROWS);
                    panel.reclass_accent(&accent);
                    let idx = self.push_note_panel_styled(panel, register);
                    // PER-GLYPH colour override (item 3): `push_note_panel_styled` is
                    // the shared constructor every non-stencil register also calls, so
                    // the colour key is attached here rather than threaded through it —
                    // see `crate::stencil_variants`'s "Colour key" doc section.
                    self.notes[idx as usize].color_key = colors;
                    idx
                }
                ParsedNote::Banner(block) => {
                    // One panel per `cycle =` word (or a single-element `Vec` for the
                    // ordinary static case) — `push_note_cycle_styled` stores whatever
                    // it's handed, so a non-cycling banner still rides `NoteEntry::Cycled`
                    // with exactly one frame (its `select_cycle_frame` short-circuits on
                    // `len() <= 1` without ever reading the clock).
                    let panels = crate::banner::build_cycle_panels(label, &block);
                    self.push_note_cycle_styled(panels, register)
                }
                ParsedNote::Variants(variants) => {
                    self.push_note_variants_styled(variants, register)
                }
            };
        }

        let styled_body = register.body(body);
        // A register that PRESERVES LAYOUT (the stencilled fixture plate) is laid
        // out line-for-line: its body is hand-drawn ASCII whose meaning is its
        // geometry, and word-wrapping would shred the frame. Every other register
        // carries prose and is re-flowed onto the plate as before.
        let panel = if register.preserves_layout() {
            TextPanel::build_verbatim(label, styled_body.as_ref(), NOTE_COLS, NOTE_ROWS)
        } else {
            TextPanel::build_wrapped_styled(
                label,
                styled_body.as_ref(),
                NOTE_COLS,
                NOTE_ROWS,
                &register.layout(),
            )
        };
        self.push_note_panel_styled(panel, register)
    }

    /// Install this level's pool table (pipe tag → owning node's words).
    pub fn set_pools(&mut self, pools: PoolTable) {
        self.pools = pools;
    }

    /// Install this level's authored `[effects] glitch` density (clamped to `[0,1]`).
    ///
    /// The ONE knob that decides whether powered pipes stream the clean `pool` or the
    /// corrupted `pool_alt` register — a whole-level scalar (there is no per-pipe
    /// corrupt flag and none is added). A host that never calls this keeps `0.0` ⇒
    /// the clean register.
    pub fn set_glitch(&mut self, glitch: f32) {
        self.glitch = if glitch.is_finite() {
            glitch.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    /// This level's glitch density `[0,1]`.
    #[inline]
    pub fn glitch(&self) -> f32 {
        self.glitch
    }

    /// Which word register this level's pipes stream: [`PoolRegister::Alt`] once the
    /// glitch density passes [`POOL_ALT_GLITCH_THRESHOLD`](crate::pipe_text::POOL_ALT_GLITCH_THRESHOLD), else
    /// [`PoolRegister::Clean`].
    #[inline]
    pub fn pool_register(&self) -> PoolRegister {
        PoolRegister::for_glitch(self.glitch)
    }

    /// Number of registered code panels.
    #[inline]
    pub fn code_len(&self) -> usize {
        self.code.len()
    }

    /// Number of registered note panels.
    #[inline]
    pub fn note_len(&self) -> usize {
        self.notes.len()
    }

    /// Whether the registry carries no panels and no pools.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.code.is_empty() && self.notes.is_empty() && self.pools.is_empty()
    }

    /// The register note panel `index` speaks in ([`NoteRegister::Plain`] when the
    /// note is absent — a missing placard is a plain blank plate).
    #[inline]
    pub fn note_register(&self, index: u32) -> NoteRegister {
        self.note(index).map_or(NoteRegister::Plain, |r| r.register)
    }

    /// The level's pool table (empty when none was installed).
    #[inline]
    pub fn pools(&self) -> &PoolTable {
        &self.pools
    }

    /// Borrow code panel `index`. An out-of-range index resolves to panel 0 when
    /// any panel exists (an author who fat-fingers `panel = 999` gets the anchor
    /// panel, not a surprise or a blank), and to `None` when the registry is empty.
    #[inline]
    fn code(&self, index: u32) -> Option<&Registered> {
        if self.code.is_empty() {
            return None;
        }
        Some(self.code.get(index as usize).unwrap_or(&self.code[0]))
    }

    /// Borrow note panel `index`. Out-of-range is `None` — unlike a code panel there
    /// is no meaningful "anchor note" to substitute, and drawing the WRONG placard
    /// would be a content lie; a missing note renders as a blank plate.
    #[inline]
    fn note(&self, index: u32) -> Option<&RegisteredNote> {
        self.notes.get(index as usize)
    }
}

/// The process-wide active registry. Swapped wholesale on level load; read (never
/// mutated in place) by the shade path. `Arc` so a reader could hold it across a
/// swap if that is ever needed; `RwLock` because the writer is a different system —
/// and possibly a different thread — from the readers.
static ACTIVE: RwLock<Option<Arc<TextSurfaceRegistry>>> = RwLock::new(None);

/// The registry every read falls back to when no host has installed one: empty, so
/// code walls draw blank and pipes use their built-in words. This is the standalone
/// engine behaviour (a `re_render` unit test, or `re_imgqa`, with no game content).
static FALLBACK: TextSurfaceRegistry = TextSurfaceRegistry::new();

/// Read the active registry, recovering from a poisoned lock rather than panicking:
/// a panic in an unrelated reader must not take the renderer down, and every reader
/// here is read-only so the data cannot be logically torn.
#[inline]
fn read_active<R>(f: impl FnOnce(&TextSurfaceRegistry) -> R) -> R {
    let guard = ACTIVE.read().unwrap_or_else(PoisonError::into_inner);
    match guard.as_deref() {
        Some(reg) => f(reg),
        None => f(&FALLBACK),
    }
}

/// Publish `registry` as the active text surfaces, replacing (and dropping) any
/// previous one. Call once per level load.
pub fn install(registry: TextSurfaceRegistry) {
    install_shared(Arc::new(registry));
}

/// Publish an already-shared registry. Same as [`install`], for a host that keeps
/// its own handle (e.g. to re-publish an unchanged panel set without rebuilding it).
pub fn install_shared(registry: Arc<TextSurfaceRegistry>) {
    let mut guard = ACTIVE.write().unwrap_or_else(PoisonError::into_inner);
    *guard = Some(registry);
}

/// Drop the active registry — back to the empty fallback. Call at level teardown so
/// a menu/idle frame carries no stale level text.
pub fn clear() {
    let mut guard = ACTIVE.write().unwrap_or_else(PoisonError::into_inner);
    *guard = None;
}

/// Number of code panels currently published (`0` with no registry). Lets a host or
/// a test assert the registry LIFECYCLE — in particular that loading a second level
/// replaces rather than accumulates panels.
pub fn code_panel_count() -> usize {
    read_active(TextSurfaceRegistry::code_len)
}

/// Number of note panels currently published (`0` with no registry).
pub fn note_panel_count() -> usize {
    read_active(TextSurfaceRegistry::note_len)
}

/// Sample code panel `index` at a wall coordinate, plus the panel's row count (the
/// code recipe needs `rows()` for its world-anchored scanline parity).
///
/// Returns the blank glyph and `0` rows when no panel is published, so the shader
/// has a defined answer with no host content.
#[inline]
pub fn sample_code_panel(index: u32, u: f32, v: f32, u_chars_per_unit: f32) -> (CodeGlyph, usize) {
    read_active(|reg| {
        reg.code(index).map_or((BLANK_GLYPH, 0), |r| {
            (r.panel.sample(u, v, u_chars_per_unit), r.panel.rows())
        })
    })
}

/// Sample code panel `index` as a CONTINUOUS text plane flowing along a wall run at a
/// screen-matched density, returning the glyph and the SOURCE ROW it came from (the
/// code recipe keys its world-anchored CRT scanline on that row).
///
/// This is what a cast that measured the wall's screen span uses; [`sample_code_panel`]
/// remains the mapping for any caller that could not measure it. See
/// [`TextPanel::sample_flowing`](crate::code_wall::TextPanel::sample_flowing).
#[inline]
pub fn sample_code_panel_flowing(
    index: u32,
    run_u: f32,
    wall_v: f32,
    scale: crate::code_wall::FlowScale,
) -> (CodeGlyph, i64) {
    read_active(|reg| {
        reg.code(index).map_or((BLANK_GLYPH, 0), |r| {
            r.panel.sample_flowing(run_u, wall_v, scale)
        })
    })
}

/// [`sample_code_panel_flowing`] with the face's exact screen extent, so a magnified
/// face widens its spaces instead of splitting its words
/// ([`crate::code_wall::CodePanel::sample_flowing_justified`]).
#[inline]
pub fn sample_code_panel_justified(
    index: u32,
    run_u: f32,
    wall_v: f32,
    scale: crate::code_wall::FlowScale,
    u_cols: f32,
    span_cols: f32,
) -> (CodeGlyph, i64) {
    read_active(|reg| {
        reg.code(index).map_or((BLANK_GLYPH, 0), |r| {
            r.panel
                .sample_flowing_justified(run_u, wall_v, scale, u_cols, span_cols)
        })
    })
}

// ─────────────────────── live cycle clock (`[note banner] cycle =`) ─────────
//
// `ascii_shader::Material::sample_note_wall` calls
// `sample_note_panel_anchored(note_index, u, v, anchor, face_rows)` — a signature
// this module owns but cannot add a `time` PARAMETER to without also editing that
// call site. So, EXACTLY like `crate::state_bus` publishes the live `CellStateMap`
// for `stencil_variants::sample`'s `bind =` lookup, the live cast TIME is published
// here via a thread-local and read back inside `select_cycle_frame` with no new
// argument anywhere in the call chain.

thread_local! {
    /// Elapsed cast time (seconds) for [`NoteEntry::Cycled`]'s frame pick. Default
    /// `0.0` ⇒ every cycling banner samples `frames[0]` — the deterministic answer
    /// `--dump`, every golden, and every focused unit test that never calls
    /// [`set_cycle_time`] gets for free (see [`select_cycle_frame`]).
    static CYCLE_CLOCK: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
}

/// Publish `t` (elapsed cast seconds, `SampleCtx::time`'s own value) as the live
/// cycle clock for this thread. Called once per shaded cell from `crate::shade_cell`
/// — a single `Cell::set`, cheap enough to pay unconditionally, and idempotent
/// within one cast since every cell of a frame carries the same `time`.
#[inline]
pub(crate) fn set_cycle_time(t: f32) {
    CYCLE_CLOCK.with(|c| c.set(t));
}

/// The live cycle clock for this thread (`0.0` if [`set_cycle_time`] was never
/// called on it).
#[inline]
fn cycle_time() -> f32 {
    CYCLE_CLOCK.with(std::cell::Cell::get)
}

/// Which frame of a cycling banner is live right now: `⌊time / CYCLE_WORD_SECS⌋`,
/// wrapped to `frames.len()` — so `time = 0.0` always selects `frames[0]`
/// (`crate::banner::CYCLE_WORD_SECS` carries the cadence rationale). A
/// length-`0`-or-`1` rotation always answers `frames.first()` without ever reading
/// the clock, so the ordinary non-cycling banner (`NoteEntry::Cycled` with exactly
/// one frame) costs nothing extra. `None` only for the defensive empty-`Vec` case;
/// every value `crate::banner::build_cycle_panels` produces has at least one frame.
/// Non-finite `time` (should not occur; `SampleCtx::time` is always a real elapsed
/// duration) falls back to `0.0` rather than propagating a NaN index.
fn select_cycle_frame(frames: &[BannerFrame], time: f32) -> Option<&BannerFrame> {
    if frames.len() <= 1 {
        return frames.first();
    }
    let t = if time.is_finite() { time.max(0.0) } else { 0.0 };
    let idx = (t / crate::banner::CYCLE_WORD_SECS) as usize % frames.len();
    frames.get(idx)
}

/// Which SIZE TIER of a banner's live word is legible right now: the largest one
/// whose own panel extent the MEASURED face actually clears — `face_rows` for the
/// vertical axis (published per-column by the caster, same value every other note
/// sampler already receives) and `anchor.span_cols` for the horizontal one (the
/// face's real projected column width — `crate::face_span::FaceAnchor`, measured
/// per [`crate::face_span::face_anchor_for`], or the legacy `u * face_cols` estimate
/// when only [`sample_note_panel`]'s `face_cols` is known).
///
/// This is the reason a banner needs BOTH axes checked, unlike
/// `stencil_variants::select_tier`'s face-rows-only rule: `stencil_variants` panels
/// are always cropped to the fixed [`crate::code_wall::NOTE_COLS`]/`NOTE_ROWS`
/// budget regardless of body content, so a row-only cutoff is a fair proxy; a
/// banner's FULL tier is a pixel-font raster whose WIDTH is what actually blew the
/// budget in attempt 1 (a 4-letter word at `scale = 3` rasterises far wider than it
/// is tall), so the horizontal check is load-bearing, not decoration.
///
/// `0`/unknown measurements (`face_rows <= 0.0` or `span_cols <= 0.0` — a focused
/// unit test, or a caster path that never measured the face) keep the FULL tier,
/// mirroring `stencil_variants::select_tier`'s own "zero/unknown ⇒ the historical
/// full body" fallback, so every pre-existing (non-anchored) caller is unaffected.
fn select_banner_tier(frame: &BannerFrame, face_rows: f32, span_cols: f32) -> &CyclePanel {
    if face_rows <= 0.0 || span_cols <= 0.0 {
        return &frame.full;
    }
    let clears =
        |t: &CyclePanel| face_rows >= t.panel.rows() as f32 && span_cols >= t.panel.cols() as f32;
    if clears(&frame.full) {
        return &frame.full;
    }
    if clears(&frame.compact) {
        return &frame.compact;
    }
    &frame.accent
}

/// Whether note `index` is a `[note banner]` placard (as opposed to an ordinary
/// `[note]`/`[note stencil]`) — i.e. whether it resolved to [`NoteEntry::Cycled`],
/// the entry kind only `crate::banner::build_cycle_panels` produces. Read by
/// `crate::material_from_def`'s per-material [`crate::lod::LodOverride`] pick: a
/// banner's whole point is staying legible at range, so it gets the same
/// "hold full contrast/no-fade out to a distance" lever `puzzle_node`/the flat-image
/// fix already use, rather than fading through the universal depth-contrast pass
/// like an ordinary placard. `false` for an unknown index (an absent note is not a
/// banner) — the safe, non-panicking answer.
#[inline]
pub(crate) fn is_banner(index: u32) -> bool {
    read_active(|reg| matches!(reg.note(index), Some(n) if matches!(n.entry, NoteEntry::Cycled(_))))
}

/// Sample note panel `index` fitted across a unit wall face, together with the
/// REGISTER it speaks in (which selects the shader's ink/plate palette).
///
/// Returns the blank glyph in the [`NoteRegister::Plain`] register when the note is
/// not published — a missing placard is a plain blank plate, never another note's
/// text. One read guard, one slice index, one grid index: both facts come from the
/// SAME guard, so a styled note costs no extra lock.
///
/// `face_cols` / `face_rows` are how many terminal COLUMNS and ROWS this placard's wall
/// face spans on screen ([`crate::face_span`]). They let the panel REFUSE TO MAGNIFY: a
/// face wider/taller than the text block draws it at integer 1:1 scale, centred, instead
/// of nearest-neighbour stretching it into doubled letters (`thhe latttice`) and doubled
/// LINES that re-deal as the player walks. `0.0` (or any non-finite / smaller-than-block
/// value) selects the historical stretch mapping on that axis, so a caller that cannot
/// measure the face is unchanged.
#[inline]
pub fn sample_note_panel(index: u32, u: f32, v: f32, face_cols: f32, face_rows: f32) -> NoteSample {
    sample_note_panel_anchored(
        index,
        u,
        v,
        crate::face_span::FaceAnchor::estimated(u.clamp(0.0, 1.0), face_cols),
        face_rows,
    )
}

/// [`sample_note_panel`] with the horizontal placement given as a MEASURED
/// [`FaceAnchor`](crate::face_span::FaceAnchor) — the exact projected offset of this
/// screen column from the face's `u = 0` edge, which is what the shade path threads.
/// See [`TextPanel::sample_fitted_anchored_2d`](crate::code_wall::TextPanel::sample_fitted_anchored_2d).
#[inline]
pub fn sample_note_panel_anchored(
    index: u32,
    u: f32,
    v: f32,
    anchor: crate::face_span::FaceAnchor,
    face_rows: f32,
) -> NoteSample {
    read_active(|reg| {
        reg.note(index).map_or(
            NoteSample {
                glyph: BLANK_GLYPH,
                register: NoteRegister::Plain,
                color_override: None,
            },
            |r| {
                let glyph = match &r.entry {
                    NoteEntry::Simple { panel, used_rows } => {
                        panel.sample_fitted_anchored_2d(u, v, *used_rows, anchor, face_rows)
                    }
                    NoteEntry::Variants(variants) => {
                        stencil_variants::sample(variants, u, v, anchor, face_rows)
                    }
                    NoteEntry::Cycled(frames) => match select_cycle_frame(frames, cycle_time()) {
                        Some(f) => {
                            let tier = select_banner_tier(f, face_rows, anchor.span_cols);
                            tier.panel.sample_fitted_anchored_2d(
                                u,
                                v,
                                tier.used_rows,
                                anchor,
                                face_rows,
                            )
                        }
                        None => BLANK_GLYPH,
                    },
                };
                // PER-GLYPH colour override (item 3): a linear scan over the note's
                // own (tiny — placard-sized) colour key, keyed by the CHARACTER just
                // sampled. Empty on every note with no `colors =` key (the overwhelming
                // majority), so this costs one slice-is-empty check on the hot path.
                let color_override = char::from_u32(glyph.glyph)
                    .and_then(|ch| r.color_key.iter().find(|e| e.glyph == ch).map(|e| e.rgb));
                NoteSample {
                    glyph,
                    register: r.register,
                    color_override,
                }
            },
        )
    })
}

/// One sampled placard cell: the glyph the panel holds there, the note's register,
/// and an optional per-glyph colour override. `Copy` — the shader's per-cell note
/// read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteSample {
    /// The grid glyph + its class (`Comment` marks the register's structural prefix).
    pub glyph: CodeGlyph,
    /// The voice this placard speaks in.
    pub register: NoteRegister,
    /// The note's authored `colors =` RGB for THIS glyph character
    /// (`0x00RRGGBB`, `ascii_shader::pack_rgb`'s own format), or `None` when this
    /// glyph is unkeyed (the overwhelming majority of cells, and every note that
    /// authors no `colors =` key at all). `ascii_shader::Material::sample_note_wall`
    /// applies it as the FINAL override, after its class-based colour pick — see
    /// `crate::stencil_variants`'s module docs, "Colour key" section.
    pub color_override: Option<u32>,
}

/// The blank cell returned when a panel is absent. Mirrors the panel's own padding
/// cell so an absent panel and an empty region of a present one look identical.
const BLANK_GLYPH: CodeGlyph = CodeGlyph {
    glyph: b' ' as u32,
    class: crate::code_wall::CodeClass::Blank,
};

/// Run `f` with the word pool owned by the node that owns `pipe_tag`, or with
/// `None` when there is no such entry (no registry, no table, unknown tag, or an
/// unauthored pool). The pool is a real borrow for the closure body — this is how
/// the pipe surface receives it without a lifetime on `SampleCtx`.
///
/// Hot path: one read guard + two slice index reads, and only for a POWERED pipe
/// cell.
#[inline]
pub fn with_pipe_pool<R>(pipe_tag: u16, f: impl FnOnce(Option<&NodePool>) -> R) -> R {
    read_active(|reg| f(reg.pools.pool_for_pipe(pipe_tag)))
}

/// [`with_pipe_pool`] plus the level's WORD REGISTER (clean vs. corrupted), both
/// resolved under ONE read guard.
///
/// This is what the shade path calls: the pool answers "which node's words", the
/// register answers "clean or `pool_alt`" — and the register is a whole-level fact
/// (the authored `[effects] glitch` density vs. [`POOL_ALT_GLITCH_THRESHOLD`](crate::pipe_text::POOL_ALT_GLITCH_THRESHOLD)), so
/// taking a second guard for it would be pure waste.
#[inline]
pub fn with_pipe_pool_register<R>(
    pipe_tag: u16,
    f: impl FnOnce(Option<&NodePool>, PoolRegister) -> R,
) -> R {
    read_active(|reg| f(reg.pools.pool_for_pipe(pipe_tag), reg.pool_register()))
}

/// The word register the ACTIVE level's pipes stream in (diagnostics + tests; the
/// shade path gets it from [`with_pipe_pool_register`] for free).
pub fn active_pool_register() -> PoolRegister {
    read_active(TextSurfaceRegistry::pool_register)
}

/// An RAII scope that installs a registry and RESTORES the previous one on drop.
///
/// Tests use this so they never leak state into each other; the game uses
/// [`install`] / [`clear`] at the level boundary. Restores on unwind too.
pub struct SceneTextScope {
    prev: Option<Arc<TextSurfaceRegistry>>,
}

impl SceneTextScope {
    /// Install `registry` until the guard drops.
    pub fn new(registry: TextSurfaceRegistry) -> SceneTextScope {
        let prev = {
            let guard = ACTIVE.read().unwrap_or_else(PoisonError::into_inner);
            guard.clone()
        };
        install(registry);
        SceneTextScope { prev }
    }
}

impl Drop for SceneTextScope {
    fn drop(&mut self) {
        let mut guard = ACTIVE.write().unwrap_or_else(PoisonError::into_inner);
        *guard = self.prev.take();
    }
}

/// TEST-ONLY serialisation of the active registry.
///
/// The registry is process-global by design (see the module docs), but `cargo test`
/// runs a crate's tests in parallel THREADS of one process — so a test that
/// installs/clears the registry would otherwise race any other test that renders a
/// code wall. Every test in this crate that touches the active registry takes this
/// lock first, which makes them mutually exclusive without making the production
/// path pay anything.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_wall::CodeClass;
    use crate::pipe_text::POOL_ALT_GLITCH_THRESHOLD;

    fn reg_with(code: &[&str], notes: &[&str]) -> TextSurfaceRegistry {
        let mut r = TextSurfaceRegistry::new();
        for (i, src) in code.iter().enumerate() {
            r.push_code_source(&format!("code{i}"), src);
        }
        for (i, body) in notes.iter().enumerate() {
            r.push_note_text(&format!("note{i}"), body);
        }
        r
    }

    #[test]
    fn no_registry_reads_blank_and_no_pool() {
        let _serial = test_lock();
        // No scope installed on this path: the fallback answers.
        let (g, rows) = read_active(|reg| {
            assert!(reg.is_empty());
            (BLANK_GLYPH, reg.code_len() + reg.note_len())
        });
        assert_eq!(g.class, CodeClass::Blank);
        assert_eq!(rows, 0);
    }

    #[test]
    fn code_panel_index_space_is_registration_order() {
        let _serial = test_lock();
        let _s = SceneTextScope::new(reg_with(&["AAA", "BBB"], &[]));
        assert_eq!(code_panel_count(), 2);
        // u = 0.5 with 1 char per unit lands on grid column 0 of the top row.
        let (g0, rows) = sample_code_panel(0, 0.5, 0.001, 1.0);
        let (g1, _) = sample_code_panel(1, 0.5, 0.001, 1.0);
        assert_eq!(g0.glyph, u32::from('A'));
        assert_eq!(g1.glyph, u32::from('B'));
        assert_eq!(rows, PANEL_ROWS, "the code recipe needs the grid height");
    }

    #[test]
    fn out_of_range_code_panel_falls_back_to_panel_zero() {
        let _serial = test_lock();
        let _s = SceneTextScope::new(reg_with(&["AAA", "BBB"], &[]));
        let (g_hi, _) = sample_code_panel(999, 0.5, 0.001, 1.0);
        let (g_0, _) = sample_code_panel(0, 0.5, 0.001, 1.0);
        assert_eq!(g_hi.glyph, g_0.glyph);
    }

    #[test]
    fn note_panel_is_fitted_across_the_face() {
        let _serial = test_lock();
        let _s = SceneTextScope::new(reg_with(&[], &["hello"]));
        assert_eq!(note_panel_count(), 1);
        // One 5-char word wrapped onto a 26-col grid: it occupies the first row, so
        // the whole (single) used row spans v ∈ [0,1].
        let g = sample_note_panel(0, 0.02, 0.5, 0.0, 0.0);
        assert_eq!(g.glyph.glyph, u32::from('h'));
        // Far right of the plate is padding.
        assert_eq!(
            sample_note_panel(0, 0.99, 0.5, 0.0, 0.0).glyph.class,
            CodeClass::Blank
        );
    }

    #[test]
    fn missing_note_panel_is_blank_not_another_note() {
        let _serial = test_lock();
        let _s = SceneTextScope::new(reg_with(&[], &["first"]));
        assert_eq!(
            sample_note_panel(0, 0.02, 0.5, 0.0, 0.0).glyph.glyph,
            u32::from('f')
        );
        let missing = sample_note_panel(7, 0.02, 0.5, 0.0, 0.0);
        assert_eq!(
            missing.glyph.class,
            CodeClass::Blank,
            "an out-of-range note index must NOT show a different placard"
        );
        assert_eq!(
            missing.register,
            NoteRegister::Plain,
            "an absent placard is a PLAIN blank plate (a defined answer)"
        );
    }

    // ── per-register note styling reaches the sampler ──

    /// The register a panel's LABEL declares rides through registration to the
    /// sample, and it drives the LAYOUT: a `system_log` body comes back `>`-prefixed
    /// (with the prefix classified as structure), a `model_thought` body comes back
    /// parenthesised, a `researcher` body is indented by its ragged cycle.
    #[test]
    fn note_register_from_the_label_reaches_the_sample_and_the_layout() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text("note0:Researcher", "alpha");
        r.push_note_text("note1:SystemLog", "beta");
        r.push_note_text("note2:ModelThought", "gamma");
        r.push_note_text("note3", "delta"); // no register suffix ⇒ Plain
        let _s = SceneTextScope::new(r);

        // Read the top row of each plate, one sample per grid column.
        let row = |index: u32| -> String {
            (0..NOTE_COLS)
                .map(|c| {
                    let u = (c as f32 + 0.5) / NOTE_COLS as f32;
                    let s = sample_note_panel(index, u, 0.01, 0.0, 0.0);
                    char::from_u32(s.glyph.glyph).unwrap_or('?')
                })
                .collect::<String>()
                .trim_end()
                .to_string()
        };

        assert_eq!(
            sample_note_panel(0, 0.5, 0.01, 0.0, 0.0).register,
            NoteRegister::Researcher
        );
        assert_eq!(
            row(0),
            "alpha",
            "researcher row 0 has indent 0 in its cycle"
        );

        assert_eq!(
            sample_note_panel(1, 0.5, 0.01, 0.0, 0.0).register,
            NoteRegister::SystemLog
        );
        assert_eq!(row(1), "> beta", "a system log line is `>`-prefixed");
        // The prefix is CLASSIFIED as structure so the shader inks it separately.
        let first = sample_note_panel(1, 0.5 / NOTE_COLS as f32, 0.01, 0.0, 0.0);
        assert_eq!(first.glyph.glyph, u32::from('>'));
        assert_eq!(first.glyph.class, CodeClass::Comment);

        assert_eq!(
            sample_note_panel(2, 0.5, 0.01, 0.0, 0.0).register,
            NoteRegister::ModelThought
        );
        assert_eq!(row(2), " (gamma)", "a thought is parenthetical and inset");

        assert_eq!(
            sample_note_panel(3, 0.5, 0.01, 0.0, 0.0).register,
            NoteRegister::Plain
        );
        assert_eq!(
            row(3),
            "delta",
            "an unlabelled note keeps the flush 1.0 layout"
        );
    }

    /// The explicit typed API selects the register regardless of the label — the
    /// path a host should use once it threads the kind as a value.
    #[test]
    fn explicit_register_overrides_the_label() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled("whatever", "one", NoteRegister::SystemLog);
        let _s = SceneTextScope::new(r);
        assert_eq!(
            sample_note_panel(0, 0.5, 0.01, 0.0, 0.0).register,
            NoteRegister::SystemLog
        );
    }

    // ── item 3: per-glyph `colors =` reaches `NoteSample::color_override` ──

    /// A keyed glyph carries its OWN authored RGB in `color_override`; an unkeyed
    /// glyph in the SAME body carries `None` — proving the override is per-CHARACTER,
    /// not a single shared accent (the legacy `reclass_accent` behaviour this
    /// replaces for colour purposes).
    #[test]
    fn a_legacy_stencil_colors_key_gives_each_keyed_glyph_its_own_override() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        // Two DISTINCT keyed glyphs with two DISTINCT colours, plus one unkeyed.
        r.push_note_text_styled(
            "n",
            "colors = [:#e0a040 ]:#40a0ff\n[X]",
            NoteRegister::Stencil,
        );
        let _s = SceneTextScope::new(r);

        let cell = |col: usize| -> NoteSample {
            let u = (col as f32 + 0.5) / NOTE_COLS as f32;
            sample_note_panel(0, u, 0.01, 0.0, 0.0)
        };
        // Column 0 = '[', column 1 = 'X' (unkeyed), column 2 = ']'.
        assert_eq!(cell(0).glyph.glyph, u32::from('['));
        assert_eq!(
            cell(0).color_override,
            Some(0xe0a040),
            "'[' keeps its own colour"
        );
        assert_eq!(cell(1).glyph.glyph, u32::from('X'));
        assert_eq!(cell(1).color_override, None, "unkeyed 'X' is unaffected");
        assert_eq!(cell(2).glyph.glyph, u32::from(']'));
        assert_eq!(
            cell(2).color_override,
            Some(0x40a0ff),
            "']' has its OWN colour, not '['s"
        );
    }

    /// The same per-glyph key reaches every tier of a state-swapped/tiered stencil
    /// (the `Variants` path), not just the legacy single-body path.
    #[test]
    fn a_variants_stencil_colors_key_also_reaches_the_sample() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled(
            "n",
            "colors = X:#123456\n--- 0 ---\nX Y",
            NoteRegister::Stencil,
        );
        let _s = SceneTextScope::new(r);
        let cell = |col: usize| -> NoteSample {
            let u = (col as f32 + 0.5) / NOTE_COLS as f32;
            sample_note_panel(0, u, 0.01, 0.0, 0.0)
        };
        assert_eq!(cell(0).color_override, Some(0x123456));
        assert_eq!(cell(2).glyph.glyph, u32::from('Y'));
        assert_eq!(cell(2).color_override, None);
    }

    /// A note with NO `colors =` key (the overwhelming majority, and every shipped
    /// stencil today) carries `color_override: None` on every cell — the golden-safe
    /// default this feature must not disturb.
    #[test]
    fn no_colors_key_means_no_override_anywhere() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled("n", "[X]", NoteRegister::Stencil);
        let _s = SceneTextScope::new(r);
        for col in 0..3 {
            let u = (col as f32 + 0.5) / NOTE_COLS as f32;
            assert_eq!(sample_note_panel(0, u, 0.01, 0.0, 0.0).color_override, None);
        }
    }

    // ── the level glitch scalar selects the pipe word register ──

    /// Below/at the threshold the pipes stream the CLEAN pool; strictly above it they
    /// stream the corrupted `pool_alt` register. The default (no glitch authored) is
    /// clean.
    #[test]
    fn level_glitch_selects_the_pipe_word_register() {
        let _serial = test_lock();
        let clean = reg_with(&[], &[]);
        assert_eq!(clean.glitch(), 0.0, "no glitch authored ⇒ 0");
        assert_eq!(clean.pool_register(), PoolRegister::Clean);

        let mut at = reg_with(&[], &[]);
        at.set_glitch(POOL_ALT_GLITCH_THRESHOLD);
        assert_eq!(
            at.pool_register(),
            PoolRegister::Clean,
            "the threshold is STRICTLY exceeded, so at the bar it is still clean"
        );

        let mut over = reg_with(&[], &[]);
        over.set_glitch(POOL_ALT_GLITCH_THRESHOLD + 0.01);
        assert_eq!(over.pool_register(), PoolRegister::Alt);

        // …and it reaches the shade path's reader through the active registry.
        let _s = SceneTextScope::new(over);
        assert_eq!(active_pool_register(), PoolRegister::Alt);
        assert_eq!(
            with_pipe_pool_register(0, |_pool, reg| reg),
            PoolRegister::Alt,
            "the register must ride the same read guard as the pool lookup"
        );
    }

    /// An out-of-range or non-finite glitch value is CLAMPED, never propagated.
    #[test]
    fn glitch_is_clamped_to_the_unit_range() {
        let mut r = TextSurfaceRegistry::new();
        r.set_glitch(9.0);
        assert_eq!(r.glitch(), 1.0);
        r.set_glitch(-3.0);
        assert_eq!(r.glitch(), 0.0);
        r.set_glitch(f32::NAN);
        assert_eq!(
            r.glitch(),
            0.0,
            "NaN must not leak into the register choice"
        );
        assert_eq!(r.pool_register(), PoolRegister::Clean);
    }

    /// REGISTRY LIFECYCLE (the `Box::leak` rejection, proven): installing a second
    /// level's registry REPLACES the first — counts reset, they do not accumulate —
    /// and clearing returns to the empty fallback.
    #[test]
    fn installing_a_second_level_replaces_the_first() {
        let _serial = test_lock();
        let _s = SceneTextScope::new(reg_with(&["A"], &["n1", "n2", "n3"]));
        assert_eq!((code_panel_count(), note_panel_count()), (1, 3));

        // "Load" a second level: one code panel, ONE note.
        install(reg_with(&["A"], &["only"]));
        assert_eq!(
            (code_panel_count(), note_panel_count()),
            (1, 1),
            "a second level load must REPLACE the panel set, not append to it"
        );

        clear();
        assert_eq!((code_panel_count(), note_panel_count()), (0, 0));
    }

    #[test]
    fn pool_table_reaches_the_reader_through_the_registry() {
        let _serial = test_lock();
        let mut r = reg_with(&[], &[]);
        r.set_pools(PoolTable::from_parts(
            vec![re_core::NodePool {
                pool: vec!["seen".into(), "named".into()],
                pool_alt: Vec::new(),
                anchor: 1,
            }],
            &[0],
        ));
        let _s = SceneTextScope::new(r);
        let anchor = with_pipe_pool(0, |p| p.map(|p| p.ordered_word(0).unwrap().to_string()));
        assert_eq!(anchor.as_deref(), Some("named"));
        assert!(with_pipe_pool(5, |p| p.is_none()), "unknown tag ⇒ no pool");
    }

    #[test]
    fn scope_restores_the_previous_registry() {
        let _serial = test_lock();
        let _outer = SceneTextScope::new(reg_with(&["A", "B"], &[]));
        assert_eq!(code_panel_count(), 2);
        {
            let _inner = SceneTextScope::new(reg_with(&["C"], &[]));
            assert_eq!(code_panel_count(), 1);
        }
        assert_eq!(
            code_panel_count(),
            2,
            "inner drop restores the outer registry"
        );
    }

    // ── live cycle clock (`[note banner] cycle =`, study report Pass 2 item 6) ──

    /// A one-tier-matters-here [`BannerFrame`] for the cycle-selection tests below:
    /// all three size tiers hold the SAME single character, so reading `.full`
    /// (what these tests do) is enough to identify which frame was picked.
    fn uniform_frame(ch: char) -> BannerFrame {
        let panel = || CyclePanel::new(TextPanel::build_verbatim("n", &ch.to_string(), 1, 1));
        BannerFrame {
            full: panel(),
            compact: panel(),
            accent: panel(),
        }
    }

    #[test]
    fn select_cycle_frame_of_len_le_1_never_reads_the_clock() {
        let one = vec![uniform_frame('A')];
        // An absurd time value would pick a wildly wrong index if the clock were
        // consulted; the single-frame case must ignore it entirely.
        let f = select_cycle_frame(&one, 999_999.0).expect("one frame ⇒ always Some");
        assert_eq!(f.full.panel.cell(0, 0).glyph, u32::from('A'));

        let empty: Vec<BannerFrame> = Vec::new();
        assert!(
            select_cycle_frame(&empty, 0.0).is_none(),
            "the defensive empty case is None, not a panic"
        );
    }

    #[test]
    fn select_cycle_frame_picks_by_elapsed_time_and_wraps() {
        let frames: Vec<BannerFrame> = ['A', 'B', 'C'].iter().map(|&c| uniform_frame(c)).collect();
        let word_at = |t: f32| -> char {
            char::from_u32(
                select_cycle_frame(&frames, t)
                    .unwrap()
                    .full
                    .panel
                    .cell(0, 0)
                    .glyph,
            )
            .unwrap()
        };
        let period = crate::banner::CYCLE_WORD_SECS;
        assert_eq!(
            word_at(0.0),
            'A',
            "t=0 must always select the first frame (dump/golden determinism)"
        );
        assert_eq!(
            word_at(period * 0.5),
            'A',
            "still within the first frame's window"
        );
        assert_eq!(word_at(period * 1.1), 'B');
        assert_eq!(word_at(period * 2.1), 'C');
        assert_eq!(
            word_at(period * 3.1),
            'A',
            "wraps back to the first frame after a full rotation"
        );
        // A non-finite time (should never happen in production) falls back to 0.0
        // rather than propagating a NaN/garbage index.
        assert_eq!(word_at(f32::NAN), 'A');
    }

    #[test]
    fn set_cycle_time_is_read_back_by_cycle_time() {
        set_cycle_time(0.0); // baseline, in case an earlier test left this thread dirty
        assert_eq!(cycle_time(), 0.0);
        set_cycle_time(4.25);
        assert_eq!(cycle_time(), 4.25);
        set_cycle_time(0.0); // restore, since this thread is reused by later tests
    }

    #[test]
    fn a_cycling_banner_note_samples_the_live_clocks_word() {
        let _serial = test_lock();
        set_cycle_time(0.0);
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled(
            "note0",
            "--- banner ---\ncycle = a, b\nscale = 1\nframe = off",
            NoteRegister::Stencil,
        );
        let _s = SceneTextScope::new(r);

        // Bare (unframed) scale-1 'A'/'B' panels are exactly FONT_COLS × FONT_ROWS
        // (3×5); their top-left font pixel is unlit for 'A' (`0b010` first row) and
        // lit for 'B' (`0b110` first row) — see `banner::glyph_bits`.
        let u0 = 0.5 / 3.0;
        let top_left_class = || sample_note_panel(0, u0, 0.01, 0.0, 0.0).glyph.class;

        assert_eq!(top_left_class(), CodeClass::Blank, "t=0 ⇒ cycle[0] = 'A'");
        set_cycle_time(crate::banner::CYCLE_WORD_SECS * 1.1);
        assert_eq!(
            top_left_class(),
            CodeClass::Code,
            "past one period ⇒ cycle[1] = 'B'"
        );
        set_cycle_time(0.0); // restore, since this thread is reused by later tests
    }

    #[test]
    fn a_non_cycling_banner_note_ignores_the_clock() {
        let _serial = test_lock();
        set_cycle_time(crate::banner::CYCLE_WORD_SECS * 50.0); // absurdly far into "a rotation"
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled(
            "note0",
            "--- banner ---\nword = a\nscale = 1\nframe = off",
            NoteRegister::Stencil,
        );
        let _s = SceneTextScope::new(r);

        let u0 = 0.5 / 3.0;
        assert_eq!(
            sample_note_panel(0, u0, 0.01, 0.0, 0.0).glyph.class,
            CodeClass::Blank,
            "no cycle= ⇒ always 'A', regardless of elapsed time"
        );
        set_cycle_time(0.0); // restore, since this thread is reused by later tests
    }

    // ── select_banner_tier: the "don't minify to noise" half of the fix ───────

    fn tiered_frame() -> BannerFrame {
        BannerFrame {
            full: CyclePanel::new(TextPanel::build_verbatim("n", "FULL", 8, 3)),
            compact: CyclePanel::new(TextPanel::build_verbatim("n", "C", 4, 1)),
            accent: CyclePanel::new(TextPanel::build_verbatim("n", "A", 1, 1)),
        }
    }

    #[test]
    fn a_face_that_clears_the_full_tier_gets_the_full_tier() {
        let f = tiered_frame();
        let tier = select_banner_tier(&f, 3.0, 8.0);
        assert_eq!(
            tier.panel.cell(0, 0).glyph,
            u32::from('F'),
            "8×3 face ⇒ the 8×3 full tier"
        );
    }

    #[test]
    fn a_face_too_narrow_for_full_but_wide_enough_for_compact_falls_back() {
        let f = tiered_frame();
        // Enough ROWS for the full tier (3) but not enough COLUMNS (8) — the exact
        // shape of attempt 1's failure (a wide pixel-font word on a narrow face).
        let tier = select_banner_tier(&f, 3.0, 5.0);
        assert_eq!(
            tier.panel.cell(0, 0).glyph,
            u32::from('C'),
            "narrow-but-tall face ⇒ compact, not a stretched full tier"
        );
    }

    #[test]
    fn a_face_too_small_for_either_multi_char_tier_falls_back_to_the_accent_mark() {
        let f = tiered_frame();
        let tier = select_banner_tier(&f, 1.0, 1.0);
        assert_eq!(
            tier.panel.cell(0, 0).glyph,
            u32::from('A'),
            "tiny face ⇒ the single accent mark"
        );
    }

    #[test]
    fn zero_or_unknown_measurement_keeps_the_full_tier() {
        let f = tiered_frame();
        assert_eq!(
            select_banner_tier(&f, 0.0, 0.0).panel.cell(0, 0).glyph,
            u32::from('F')
        );
        assert_eq!(
            select_banner_tier(&f, 3.0, 0.0).panel.cell(0, 0).glyph,
            u32::from('F')
        );
        assert_eq!(
            select_banner_tier(&f, 0.0, 8.0).panel.cell(0, 0).glyph,
            u32::from('F')
        );
    }

    // ── is_banner: the LOD-override query `material_from_def` reads ───────────

    #[test]
    fn is_banner_is_true_only_for_a_banner_note() {
        let _serial = test_lock();
        let mut r = TextSurfaceRegistry::new();
        r.push_note_text_styled("note0", "--- banner ---\nword = a", NoteRegister::Stencil);
        r.push_note_text_styled(
            "note1",
            "plain stencil art, not a banner",
            NoteRegister::Stencil,
        );
        r.push_note_text("note2", "an ordinary plain note");
        let _s = SceneTextScope::new(r);

        assert!(is_banner(0), "a banner note ⇒ NoteEntry::Cycled");
        assert!(!is_banner(1), "a legacy stencil note is not a banner");
        assert!(!is_banner(2), "a plain note is not a banner");
        assert!(!is_banner(99), "an unknown index is not a banner");
    }
}
