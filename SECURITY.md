# Security Policy

## Supported versions

Security fixes go into the latest `main` and the latest release tag. Older tags are
not patched; please update first and check whether the problem still reproduces.

| Version            | Supported |
|--------------------|-----------|
| latest `main`      | yes       |
| latest release tag | yes       |
| anything older     | no        |

## What counts

terminal3d is a library and a set of tools. It reads level, model and image files
and draws to your terminal. Apart from the experimental networking crates, it makes
no network calls. The interesting attack surface is what it parses and what it
prints. In scope, for example:

- a crash, hang, runaway memory use or out-of-bounds access caused by a malicious or
  malformed `.lvl` level, `.vox` model, or image texture (PNG, GIF, WebP) loaded
  through `re_level`, `re_voximport`, the editor or the linter;
- terminal escape injection: level, note or model data that makes `re_present` or a
  tool emit control sequences which reach your terminal and change its state, run
  commands, or write outside the program's own output;
- anything that makes the engine or a tool read or write files outside the paths it
  is meant to use (for example a texture path in a level that escapes the content
  root, or an editor save that writes somewhere unexpected);
- the experimental networking crates `ts_net` and `ts_netplay`. They are a testbed and
  not meant for untrusted networks, but bugs in them are welcome, since they are the
  one part of the codebase that talks to a network.

Ordinary bugs with no security impact (a wrong glyph, a physics glitch, a lint that
misfires) belong in the public issue tracker instead.

## How to report

Please **do not** open a public issue for a security problem. Report it privately by
email to **artparse90@gmail.com** with `[SECURITY]` at the start of the subject line.
Include what you did, what happened, the version or commit, your OS and terminal, and
a proof-of-concept file if you have one.

## What to expect

- Response time is **best effort, usually within a week**. This is a small volunteer
  project.
- We will confirm the problem, work on a fix, and credit you in the release notes
  unless you ask us not to.
- There is **no bug bounty**.
