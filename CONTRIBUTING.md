# Contributing to terminal3d

Thank you for helping. Contributions are merged into this repository and maintained
here; you do not need to carry a fork. This page covers what you need to get a change
in.

- [Picking something to work on](#picking-something-to-work-on)
- [Development setup](#development-setup)
- [Building and running](#building-and-running)
- [Tests](#tests)
- [Golden framebuffer tests](#golden-framebuffer-tests)
- [Lints and code rules](#lints-and-code-rules)
- [The layering rule](#the-layering-rule)
- [Levels](#levels)
- [Assets and provenance](#assets-and-provenance)
- [Sign-off (DCO)](#sign-off-dco)
- [Pull request checklist](#pull-request-checklist)

---

## Picking something to work on

1. Look at [ROADMAP.md](ROADMAP.md) and the issue tracker. The
   [good first issues](ROADMAP.md#good-first-issues) are small and self-contained.
2. Comment on the issue, or open one, before you start anything larger than a small
   fix. A short plan in the issue saves a rewrite in review.
3. Items marked "open question" in the roadmap need a design discussion before code.
4. One pull request, one topic. A renderer fix and an editor feature are two PRs.

Bug reports help most with your OS, terminal emulator and its size, colour mode
(truecolor or 256), the level, and for visual bugs a frame dump of that level.

---

## Development setup

- **Rust:** stable, via [rustup](https://rustup.rs). `rust-toolchain.toml` pins the
  channel and adds `rustfmt` and `clippy`.
- **Windows 10+:** the MSVC build tools ("Desktop development with C++"). The C
  runtime is linked statically (`.cargo/config.toml`), so built `.exe` files need no
  redistributable.
- **Linux (Debian/Ubuntu):**

  ```sh
  sudo apt install pkg-config libasound2-dev libudev-dev
  ```

  `libasound2-dev` is for audio (ALSA through kira/cpal) and `libudev-dev` for the
  gamepad (gilrs). Other distributions: the equivalent `alsa-lib` and `systemd`/`udev`
  development packages. `re_present`'s optional `x11` feature (OS cursor grab) also
  needs the X11 libraries at runtime.
- **Terminal:** Windows Terminal, or any modern Linux terminal with truecolor and
  UTF-8.

The first build compiles Bevy and its dependencies and takes a while. The `dev`
profile keeps only line tables for debug info, so `target/` stays a reasonable size.

`.cargo/config.toml` sets `TS_CONTENT_ROOT` to the repository root for every
`cargo run` and `cargo test`, so levels and assets resolve no matter which directory
you run cargo from.

---

## Building and running

The root `Cargo.toml` is a virtual workspace, so pass `-p <crate>`.

```sh
cargo build --workspace
cargo run --release -p ts_editor -- levels/22-lit-cellar.lvl      # editor with live 3D preview
cargo run --release -p re_lvlcheck -- levels/                     # lint levels
cargo run --release -p re_imgqa -- levels/22-lit-cellar.lvl       # render-quality report
cargo run --release -p re_voximport --bin vox2lvl -- in.vox in.toml > out.lvl
cargo run --release -p re_render --example perfbench              # stage-by-stage render timing
```

Useful environment variables:

| Variable | Effect |
|---|---|
| `TS_AUDIO_SILENT=1` | Never open an audio device. Use it for tests, CI and quiet runs. |
| `TS_SAVE_DIR=<dir>` | Where save and settings files go. Engine crates write none today, but games built on it do, and any new code that persists state must honour this variable. |
| `TS_COLS` + `TS_ROWS` | Fix the render grid; set both (same as `--size COLSxROWS`, which wins). |
| `TS_PLAY_CMD=<program>` | The player the editor launches for a playtest (the level path is appended). |
| `TS_BLESS_GOLDEN=1` | Rewrite the golden store instead of checking it. See below. |

### Looking at a frame

- Open the level in the editor; the right-hand pane is the real renderer.
- `re_render`'s `turn_shots` example writes plain-glyph frames to files
  (`cargo run -p re_render --example turn_shots -- <out_dir>`).
- `renderqa` writes truecolor HTML "photos" of a level from many camera poses.
- A game built on the engine can print one frame to stdout; the reference game's
  `terminal-state --dump <level.lvl> --size 100x32` (add `--ansi` for colour) accepts
  any level in this repository.

A dump before and after your change, diffed, is the fastest way to see what a render
change did. A frame in a real terminal is still the final check for anything that
moves or depends on input.

---

## Tests

Run the tests with no audio device and a throwaway save directory:

```sh
# Linux / macOS / Git Bash
export TS_AUDIO_SILENT=1 TS_SAVE_DIR="$(mktemp -d)"
cargo test --workspace --no-fail-fast

# Windows PowerShell
$env:TS_AUDIO_SILENT = "1"; $env:TS_SAVE_DIR = "$env:TEMP\t3d-test-save"
cargo test --workspace --no-fail-fast
```

Pass `--no-fail-fast`. Without it, cargo stops at the first failing test binary and
you do not see the rest.

While you iterate, test only the crate you are changing:

```sh
cargo test -p re_render
cargo test -p re_level
cargo test -p re_runtime
cargo test -p ts_editor
```

**Bug fixes start with a failing test.** Write a test that reproduces the bug, watch
it fail, then fix the code. If a bug cannot be captured in a test, say why in the PR.
Do not "fix" a failing test by weakening its assertion; if a test's premise is
wrong, explain why and change it deliberately.

---

## Golden framebuffer tests

`re_golden` renders a curated set of levels exactly as a `--dump` does, at a fixed
100×32 grid, and compares two SHA-256 digests per level with
`tests/golden/engine_fb_hashes.txt`:

- **shade** covers `{glyph, fg, bg}`: did the *look* change?
- **geom** covers `{depth, side, surface, material}`: did the *cast* change?

```sh
cargo test -p re_golden --test engine_corpus
```

Reading a failure:

- **Only shade moved:** a shader, palette or glyph change did it.
- **Both moved:** the geometry changed. Treat it as a raycaster regression until you
  prove otherwise.
- **Only geom moved:** something changed that is not visible. Understand it before
  you go further.

The suite fails closed. A missing level, a level that no longer parses, a key with no
hash and an orphaned hash each fail it, and none of them is fixed by blessing.

### Re-blessing

Blessing rewrites the stored hashes. Only bless when the render change is
**intentional and reviewed**:

1. Dump every level whose hash moved, **before** and **after** your change (with
   colour if colour changed), and put the frames or the relevant rows in the PR.
2. Say which digest moved (shade, geom or both) and why.
3. Re-bless:

   ```sh
   TS_BLESS_GOLDEN=1 cargo test -p re_golden --test engine_corpus
   ```

4. Commit the updated `tests/golden/engine_fb_hashes.txt` in the same commit as the
   render change, and add a dated entry to
   [tests/golden/README.md](tests/golden/README.md) naming the levels and the reason.
5. Games built on the engine keep their own golden stores. If your change moves
   pixels, say so in the PR, so the game side can re-bless when it takes the new tag.

A PR that re-blesses without before/after frames will not be merged. Never bless to
make an unexpected failure go away.

---

## Lints and code rules

```sh
cargo clippy --workspace --all-targets
```

- `clippy::all` is **deny**: the PR must have zero errors. `pedantic` and `nursery`
  are advisory warnings; do not add new ones in the code you write.
- `float_cmp` and `disallowed_types` are deny. Test modules may allow `float_cmp` with
  `#![cfg_attr(test, allow(clippy::float_cmp))]` at the crate root, as the existing
  crates do.
- Do not hide a real finding with `#[allow(...)]`. If you think a lint is wrong for a
  case, explain it in the PR.
- **No `unsafe`.** `unsafe_code = "forbid"` is set workspace-wide. The only exception
  is the existing OS FFI in `re_present` (cursor grab and console colour mode). Do not
  add more.
- **Formatting: match the surrounding code.** The tree is not uniformly
  `rustfmt`-formatted, so do not run `cargo fmt` over whole files or the workspace;
  that buries your change in unrelated diffs. Format the lines you write in the style
  of the file around them.
- **Docs:** new and moved modules get a `//!` header that says what the module is for
  and why; public items get `///` docs. Comments explain *why*, not *what*. Keep an
  existing header true when you change behaviour.
- **Determinism:** render, synth, director and effect code is seeded and
  reproducible. Do not add wall-clock time or unseeded randomness to those paths.
- **Dependencies:** a new dependency needs a reason in the PR, must be headless (no
  `wgpu`, `winit` or `bevy_render`), and must pass `deny.toml`. Prefer a maintained
  crate over hand-rolling a subsystem.

---

## The layering rule

```
apps/*  ──►  engine/*  ◄──  editor/*
```

- `engine/` depends only on `engine/` and external crates, including for tests.
- `editor/` may use `engine/` and `editor/`. `apps/` may use `engine/` only.
- **Only `re_present` touches the terminal.** crossterm and ratatui types are banned
  elsewhere through `clippy.toml`. Everything else must compile and test with no
  terminal attached.
- **Only `re_audio::playback` opens an audio device.**
- **The engine never names a game.** When engine code needs data a game owns, add a
  seam (a trait, a registry, or a resource the game writes) instead of an upward
  dependency. `re_runtime` plugins are generic over the host's `States`.
- `Framebuffer` and `Cell` belong to `re_core`. Extend them there; never redefine them.
- **Cross-crate changes:** if you change a public signature, update every caller in
  the workspace (grep for it) in the same PR. Crate boundaries do not stop an arity
  break in a sibling crate.

`cargo xtask layers` checks the directory rules; the same check runs inside
`cargo test --workspace`. The full rules and their reasons are in
[docs/LAYERING.md](docs/LAYERING.md).

---

## Levels

Levels are plain-text `.lvl` files in `levels/`
([docs/LEVEL_FORMAT.md](docs/LEVEL_FORMAT.md)).

1. Open or create one in the editor:
   `cargo run --release -p ts_editor -- levels/my-level.lvl`. `W` opens the new-level
   wizard, `?` shows the keymap, `:` opens the command palette.
2. Save with `s`. The editor runs the linter on save and shows the verdict.
3. Lint from the command line: `cargo run --release -p re_lvlcheck -- levels/my-level.lvl`.
   It must pass; `re_lvlcheck/tests/example_levels.rs` lints every file in `levels/`.
4. Add an entry for the file to [PROVENANCE.toml](PROVENANCE.toml).

Level submissions are licensed MIT OR Apache-2.0, like the code and every other
level here. Confirm it is your own work and include the `lvlcheck` output.

---

## Assets and provenance

Every file under `assets/`, `authoring/` and `levels/` must have a known, compatible
origin and an entry in [PROVENANCE.toml](PROVENANCE.toml). We accept only:

- work you made yourself and license to the project;
- files generated by code in this repository (for example the texture fixtures from
  `gen_texfixture` and the models from `gen_voxmodels`), with the command that
  regenerates them;
- third-party files under **CC0** or **CC BY**, committed together with their licence
  file and an attribution entry.

We do **not** accept art, models, textures, fonts or sounds downloaded without a
clear licence, fan art, or anything derived from someone else's IP, even if only a
test uses it. If you are unsure, ask in the issue first. Sounds are synth recipes in
`re_audio`, never audio files.

---

## Sign-off (DCO)

We use the [Developer Certificate of Origin](DCO), not a CLA. Sign off every commit:

```sh
git commit -s -m "re_render: clamp far-wall depth in the second plane"
```

That adds a `Signed-off-by: Your Name <you@example.com>` line, which certifies that
you have the right to submit the work under the project's licences. Unsigned commits
cannot be merged; `git rebase --signoff` fixes a branch after the fact.

Commit messages start with the crate or area (`re_render:`, `ts_editor:`, `levels:`,
`docs:`), then say what changed and why.

---

## Pull request checklist

- [ ] One topic. The description names the crates touched and why.
- [ ] Every commit is signed off (`git commit -s`).
- [ ] Tests pass for the affected crates with `TS_AUDIO_SILENT=1` and `TS_SAVE_DIR`
      set; `cargo test --workspace --no-fail-fast` for cross-crate changes.
- [ ] `cargo clippy --workspace --all-targets` shows no errors and no new warnings in
      your code.
- [ ] No new `unsafe`, no terminal types outside `re_present`, no forbidden layer edge.
- [ ] Bug fix: includes a test that failed before the fix.
- [ ] Visual change: before/after frames in the description.
- [ ] Golden hashes unchanged, **or** re-blessed with before/after frames and a
      `tests/golden/README.md` entry.
- [ ] New levels pass `lvlcheck`; new files under `assets/`, `authoring/`, `levels/`
      have a `PROVENANCE.toml` entry.
- [ ] Docs updated if behaviour, a `.lvl` key or a CLI flag changed.
- [ ] Formatting matches the surrounding code; no unrelated reformatting.
