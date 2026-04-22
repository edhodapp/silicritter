# silicritter — session handoff

**Purpose:** context for a fresh Claude session opened at `~/silicritter/` to pick up where the prior session (cwd `~/math/moments/`) left off on 2026-04-21.

Read this file, then read `DECISIONS.md`. The global `~/.claude/CLAUDE.md` already carries the load-bearing directives (About Ed, Blue-sky collaboration mode, Decision log convention, Planned / deferred projects section). Don't re-derive any of that here.

---

## What silicritter is

Committed direction: **analog silicon for emulating "living creature neural networks."** Ed's own phrase, 2026-04-21. The aim is biologically faithful silicon, not abstract neuromorphic computing.

Context lineage:
- Ed has been mulling analog-compute-for-neural-nets for a while. The motivation is the biology vs. digital efficiency gap (≈20 W brain vs. megawatts of silicon) and the mismatch between digital binary precision and neural-net imprecision.
- Global `CLAUDE.md` has a "Planned / deferred projects" section that captures the analog-silicon direction at a higher level — SkyWater 130nm via TinyTapeout or Efabless chipIgnite, toolchain Magic / Xschem / ngspice / KLayout / netgen, etc. That section can be read as background.
- Ed's transistor-level background is L-Edit-on-DOS-era, 1.5µm/0.6µm MOSIS. He is a returning practitioner; do not re-teach MOSFETs, MPW shuttles, or DRC/LVS.

This project was activated from the "deferred" slot immediately after the `iomoments` article landed (`https://hodapp.com/posts/honest-moments/`). Sister project is `~/iomoments/` — see its `DECISIONS.md` and `CD-PIPELINE-PROPOSAL.md` as examples of the decision-log style we expect here.

---

## Where we left off

Two things were locked; one thing was open.

### Locked

- **Name: `silicritter`** (see `DECISIONS.md` D001). Ed's first pick was "sili-critter," then he said "lose the hyphen, I like that more." Namespace verified clean 2026-04-21 — zero GitHub repos for `sili-critter` or `silicritter`, zero code hits, no meaningful web hits.
- **Collaboration mode for this project is blue-sky.** Per global `CLAUDE.md`: Ed brings intuition, AI brings literature review. Do not push him to "do his own research first." When he gives a vague intuition, go dig and come back with prior art + honest assessment.

### Open — this is where the next session picks up

**Target biological scale / fidelity is not yet chosen.** Ed's exact words: *"C. elegans is close to what I had in mind, but not exactly."*

Five candidate interpretations were offered to him; he has not yet responded. The options as posed:

1. **Simpler than C. elegans.** A smaller target with a mapped nervous system, or a *subset* of C. elegans — e.g., just the 75-neuron locomotion circuit, or a reflex arc.
2. **A different real organism.** *Aplysia californica* (~20,000 neurons, gill-withdrawal reflex classically mapped by Kandel / Castellucci), the leech *Hirudo* (heartbeat / swim CPG), *Drosophila* larva (partial connectome now published), simple vertebrate swimmer like a tadpole.
3. **A synthetic / toy critter.** Not a specific species — a "minimal plausible organism" with sensors, CPG, actuators, closed-loop with a simulated body. Biologically *plausible* but not *faithful*. Valentino Braitenberg's *Vehicles* (1984) occupies this territory.
4. **A creature-class parameterizable template.** Generic substrate for "very simple swimmer" or "very simple reflex forager" that parameterizes into specific organisms.
5. **Something else** — Ed may have an intuition that wasn't in the four options above. In blue-sky mode, his "not exactly" is a signal he has something specific but vague in mind. Receive whatever he says, however loose.

**My guess (prior session, not Ed's answer):** option 2 or option 5. The "C. elegans close but not exactly" framing reads as "same ambition level but a different creature or a customization of the worm." But this is a guess.

---

## First actions for this session

1. **Ask Ed which direction resonates.** If (1)-(4), confirm and proceed. If (5), receive the framing however loose it is.
2. **Do the literature dig on the chosen target.** Per the blue-sky mode directive (global CLAUDE.md):
   - What's been tried in silicon? (Mahowald silicon retina 1988, Mahowald & Douglas HH neuron 1991, Sarpeshkar silicon cochlea 1998, Indiveri / Kirchhoff Institute / BrainScaleS work, Mead *Analog VLSI and Neural Systems* 1989, etc.)
   - What's the best-available wet-lab or simulation reference for the specific target? (OpenWorm if C. elegans; Kandel's Aplysia work; etc.)
   - What neuron model fits the target? (Hodgkin-Huxley vs. Izhikevich vs. integrate-and-fire — trades between biological fidelity and transistor count.)
   - What's still open / not tried / worth Ed's bet?
3. **Scope a first prototype.** What software simulation demonstrates the silicon design would work? What does "first tape-out success" look like observably? At what MPW tier (TinyTapeout $300-500, Efabless chipIgnite low thousands)?
4. **Only then, minimal scaffolding.** This project is experimental-tier (not CD-first deliverable). Per global CLAUDE.md "CD-first for deliverables; review-discipline-first for experiments":
   - `git init`
   - DECISIONS.md with D001 (name, already drafted) plus whatever decisions emerge
   - Maybe a project-local `CLAUDE.md` delta if there's anything specific to note beyond the global file (probably minimal)
   - **No** Makefile, **no** directory skeleton predicted in advance
   - Structure emerges per experiment
5. **Defer decisions that don't need making yet:** license, specific neuron model, process node beyond SkyWater-130nm-default, simulation software, whether to simulate-only vs. aim at tape-out. Let those crystallize as the experiment shape becomes clear.

---

## What's already in place

- This directory: `~/silicritter/`
- `DECISIONS.md` preamble plus D001 (name)
- This handoff document
- Global `~/.claude/CLAUDE.md` has been recently restructured — the "About Ed," "Decision log convention," "Pitch writing for the serious audience," "Blue-sky collaboration mode," and "Planned / deferred projects" sections are all new as of 2026-04-21. Read them.

---

## What's NOT in place (and is not premature)

- No `git init` yet (waiting on target decision; minimal scaffolding emerges *after* target, not before)
- No simulation environment
- No schematic or layout work
- No project memory directory at `~/.claude/projects/-home-ed-silicritter/memory/` yet — create when the first durable memory entry arises
- License not chosen (AGPL-3.0-or-later matches Ed's other open-source projects; BSD-3-Clause would keep the door open to flowing design into proprietary siblings; ask when the moment is right)
- GitHub repo not created (defer until code exists worth committing)

---

## Open questions (in rough priority order, for Ed)

1. **Target direction** — which of 1-5 above, or something else entirely?
2. **Success criterion for first prototype** — what observable thing demonstrates "it works"? (Answered partly after target is chosen, but worth asking.)
3. **Simulate-only first, or aim at tape-out?** — TinyTapeout's $300-500 tier makes "just tape it out" affordable; simulation-first is safer but slower.
4. **License** — AGPL-3.0-or-later (consistent with his open-source pattern) or BSD-3-Clause (flow-freedom into his proprietary product line)?
5. **Collaboration model for this project** — same as iomoments (single-author, SQLite / no-PR model)? Default yes unless Ed says otherwise.

---

## Previous session summary (one paragraph)

Prior session began as iomoments CD-pipeline scaffolding work, pivoted to writing "Honest Moments" as a shareable blog post for Ed to DM to mathematicians and software devs. Article was drafted in `~/wasm_play/draft_iomoments.md`, claims-reviewed with Gemini + clean Claude (38 claims, 4 STRONG_FLAGs and 6 LOW-confidence items were addressed along with 6 math errors caught by self-audit), then published to `~/hodapp.com/content/posts/honest-moments.md`. Cover image added from Gemini generation. Post is live: `https://hodapp.com/posts/honest-moments/`. After publish, Ed pivoted to the long-deferred analog-silicon-for-neural-nets direction and named it `silicritter`. Memory was moved from `-home-ed-math-moments/memory/` to `-home-ed-iomoments/memory/` for iomoments-specific entries; cross-project entries were promoted to global `~/.claude/CLAUDE.md`. This handoff document is the last act of that session.
