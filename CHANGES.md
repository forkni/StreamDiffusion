# Changes in this fork

This repository is a fork of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
by Akio Kodaira, Chenfeng Xu and the other StreamDiffusion authors, licensed under the Apache
License 2.0 (see `LICENSE`). Upstream's license, notices and source files are kept as they are.

Changes by Alexander Korneev, made from 2026-07-13 (last updated 2026-10-22).

## Default branch (`main`)

Compared with upstream `main`, this branch only **adds** files. No upstream file is modified.

- `.github/workflows/claude-code-review.yml` (added 2026-07-13, updated 2026-08-19): an
  automated pull-request review workflow.
- `.charlie/config.yml` and `.agents/daemons/pr-review/` (added 2026-08-06): configuration for
  a pull-request review agent.
- `CHANGES.md` (this file, added 2026-10-22).

## Other branches

The fork's other branches (for example `SDTD_040_beta_release` and the `pr/*` branches) track
[dotsimulate/StreamDiffusion](https://github.com/dotsimulate/StreamDiffusion) and carry their
own changes. Their git history records what was changed and when.
