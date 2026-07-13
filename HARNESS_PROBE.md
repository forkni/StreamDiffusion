# Charlie fork-harness feasibility probe

Throwaway scratch file used to test whether the Charlie GitHub App reviews
intra-fork pull requests (base and head both in `forkni/StreamDiffusion`)
once installed on the fork.

This file and its branch (`test/charlie-harness-probe`) are deleted once the
probe PR has been evaluated. Not part of any real change.

Bump: re-trigger PR sync now that the Claude Code + Copilot reviewer harness
is wired on `forkni:SDTD_032_dev` (workflow + path instructions).

Bump 2: re-trigger after landing claude-code-review.yml + .charlie/config.yml
on `forkni:main` (default-branch validation requirement for Claude Actions;
Charlie config previously absent entirely).
