# silicritter — project-local Claude instructions

Analog-neural-silicon exploration, licensed AGPL-3.0-or-later. Aim: silicon that emulates living-creature neural networks, with developmental dynamics (synaptic exuberance + lifelong sculpting), not just adult steady-state plasticity.

## Start here
- `DECISIONS.md` — architectural decision log (immutable entries, D001+).
- `SESSION-HANDOFF.md` — origin context from the 2026-04-21 session that spawned this project.
- `~/.claude/projects/-home-ed-silicritter/memory/` — durable project memory (framing, design sketches in progress).

## Mode
Experimental / blue-sky per global `~/.claude/CLAUDE.md`. **Not a deliverable.** No CD pipeline, build scaffolding, or directory skeleton predicted in advance — structure emerges per experiment. Review discipline still applies to functional commits (global pre-commit pipeline: quality gates + Gemini review + clean-Claude review).

Blue-sky collaboration mode is active: Ed brings intuition, framing, taste; I bring literature review, prior art, formal apparatus. Don't push Ed to "do his own research first" and don't police focus when adjacent-project threads interleave.

## License and posture
silicritter is licensed **GNU AGPL-3.0-or-later** (D004, supersedes D002). Ed Hodapp is sole author and sole copyright holder. **No external contributions accepted** — pull requests and patches are declined to preserve consolidated copyright ownership (so Ed retains the right to relicense, dual-license for commercial clients, or open to contributions under a CLA later). The AGPL-as-consulting-driver pattern is explicit: commercial users needing to deploy without AGPL obligations route to Ed for a commercial-license / consulting engagement.

Repository is **public** on GitHub at `edhodapp/silicritter`, flipped from private to public on 2026-04-22 (D005). With AGPL-3.0-or-later in place and the no-external-contributions policy explicit, there was no load-bearing reason to keep it private.

- Any MPW shuttle submission (TinyTapeout, Efabless chipIgnite, etc.) still requires Ed's explicit go-ahead — license compatibility is a solved problem, but Ed owns the when/where call.
- Commercial clients who cannot accept AGPL's network-copyleft route to Ed for a commercial license / consulting engagement.

## Downstream directionality
AGPL code cannot be pulled into proprietary sibling projects under BSD-style terms without violating the AGPL. silicritter can serve as reference / teaching material for Ed's proprietary projects, but code does not flow from silicritter into proprietary repos without Ed explicitly relicensing (which he can do unilaterally as sole copyright holder, if and when he chooses).
