# Architecture Decision Log

Chronological record of design decisions for silicritter. Each entry captures
the decision, the justification, and the date. Entries are numbered
sequentially (D001, D002, ...) and never renumbered.

**Entries are immutable in content.** Once written, the decision text and
rationale are never edited or deleted - the log is a historical record.
Review this log before making new decisions to avoid re-litigating settled
questions.

## Supersession and traceability

When a decision is revised or reversed, **both entries are linked** so
traceability works in either direction without scanning the whole log. Per
the global rule in `~/.claude/CLAUDE.md` (`Decision log convention`):

1. New entry `D00N` opens with a back-pointer: `**Supersedes:** D003
   (deprecated YYYY-MM-DD HH:MM UTC). [reason]`.
2. Superseded entry `D003` gets an **append-only deprecation annotation**
   prepended: `**DEPRECATED YYYY-MM-DD HH:MM UTC - superseded by D00N.**
   [reason]`. The original body stays exactly as written, below the
   annotation.

The annotation is the one permitted addition to an old entry - it records
the later event of supersession, not a revision of the original decision.

## 2026-04-21

### D001: Project name - silicritter

Project is named `silicritter`. Aim: analog silicon for emulating "living
creature neural networks" (Ed's phrase, 2026-04-21).

Ed's first pick was `sili-critter`; he then asked to lose the hyphen, so
the canonical form is `silicritter`.

**Namespace verified clean on 2026-04-21:**
- GitHub: zero repositories matching `sili-critter` or `silicritter`;
  zero code search hits for the exact token.
- Web search: no meaningful matches.
- PyPI / crates.io not checked - silicritter is not expected to ship as a
  package; re-verify if that changes.

**Alternatives considered and rejected** (prior session proposed these;
Ed picked `silicritter` directly): `wetware`, `neurosilicon`,
`silicon-soma`, `critter`, `oscillon`, `c-silicon`.

Name reads as "silicon + critter" - fits Ed's naming pattern (clear,
narrow, not overselling; compare ws_pi5, fireasmserver, iomoments) and
honors the "living creature" framing without claiming a specific species.

### D002: License - copyright Ed Hodapp, all rights reserved (proprietary, for now)

**DEPRECATED 2026-04-22 03:04 UTC - superseded by D004.** Ed reassessed
his IP posture: patents on silicritter-scale work are a long-shot and
not worth optimizing for. AGPL-3.0-or-later with no external
contributions accepted better supports his consulting-based monetization
while preserving future licensing flexibility through consolidated
copyright ownership. Original entry body preserved below.

Initial license stance: fully proprietary. No rights granted to any party.
This is a reservation, not a final choice - it keeps all downstream
licensing options open (proprietary product, BSD-3-Clause flow into
sibling projects, AGPL-3.0-or-later alignment with Ed's open-source
pattern, dual licensing). The decision between those can be made when
the shape of the work and its audience are clearer.

**Rationale: patent optionality.** US patent law gives a 1-year grace
period on the inventor's own disclosures (35 USC 102(b)), but most
foreign jurisdictions (EU, China, Japan) enforce absolute novelty - any
public disclosure permanently bars foreign patents there. "All rights
reserved, private repo" keeps that clock unstarted. An open-source
license chosen now would be effectively irreversible; a proprietary
license chosen now can become any license later.

**Candidates deferred** (revisit when patent / productization intent
crystallizes):
- AGPL-3.0-or-later - consistent with Ed's other open-source projects.
- BSD-3-Clause - would permit flow of design into proprietary siblings.
- Dual license (commercial + open source) - requires retained full
  copyright ownership, which this posture preserves.

### D003: Hosting - private GitHub repository under `edhodapp` account

**DEPRECATED 2026-04-22 03:04 UTC - visibility clause superseded by
D005.** Repository flipped from private to public on 2026-04-22 once
AGPL-3.0-or-later licensing was in place (D004). The
"hosted-on-GitHub-under-edhodapp-account" aspect of D003 is unchanged;
only the private visibility is superseded. Original entry body
preserved below.

`silicritter` is hosted as a **private** GitHub repository owned by the
`edhodapp` account. Repo-local git config sets `user.name = edhodapp`
and `user.email = ed@hodapp.com`; no global git config modified.

**Rationale:** a private remote satisfies the backup / working-copy-in-
cloud need while preserving the non-disclosure posture required to keep
patent options open (per D002). Private-to-public is trivial; public-
to-private does not recover lost foreign patent rights.

**Not committed to the repo** (see `.gitignore`):
- `.claude/settings.local.json` - per-machine Claude Code settings.

Note on timing: `SESSION-HANDOFF.md` (written earlier today) had
deferred `git init` "until after target decision." Ed overrode that
sequence - infrastructure first, target decision still open. The
experimental-tier rule ("no build scaffolding or directory skeleton
predicted in advance") is still honored; the repo holds only
`LICENSE`, `.gitignore`, `CLAUDE.md`, `DECISIONS.md`, and
`SESSION-HANDOFF.md` at initial commit.

## 2026-04-22

### D004: License - AGPL-3.0-or-later, no external contributions accepted

**Supersedes:** D002 (deprecated 2026-04-22 03:04 UTC). Patent
optionality, the primary rationale for D002's all-rights-reserved
stance, is no longer load-bearing - Ed's assessment: patents on
silicritter-scale work are a long-shot and not worth optimizing for.

silicritter is licensed under the **GNU Affero General Public License,
version 3, or (at your option) any later version** ("AGPL-3.0-or-later").
Full license text in `LICENSE`.

**No external contributions accepted** at this time. Pull requests,
patches, and similar upstream submissions are declined. Rationale:
consolidated copyright ownership (Ed Hodapp as sole author) preserves
the widest set of future licensing options - relicense, dual-license
(e.g., AGPL + commercial for clients who cannot accept network-copyleft
terms), or open to contributions under a CLA later. Accepting external
contributions under AGPL would lock those options down.

**Consulting-compatible.** AGPL's strong copyleft (including Section
13's network-use provision) means commercial users deploying
silicritter-based services must either release their derivative works
under AGPL or obtain a commercial license from Ed. The
commercial-license path routes those users to Ed for consulting /
licensing engagements. Standard AGPL-as-consulting-driver pattern
(MongoDB pre-SSPL, Qt, Sentry, etc.).

**Downstream directionality:** AGPL code cannot flow into proprietary
sibling projects under BSD-style terms without violating AGPL.
silicritter serves as reference / teaching artifact for Ed's proprietary
projects; code does not flow in that direction without Ed unilaterally
relicensing (which he can do as sole copyright holder, if and when
desired).

**Scope:** "for most things now" (Ed, 2026-04-22) - the whole silicritter
codebase is AGPL-3.0-or-later as a default. Specific future components
that need different licensing (e.g., MPW shuttle submission artifacts
under shuttle-specific terms) are case-by-case decisions at that time,
facilitated by Ed's sole-copyright-holder status.

### D005: Visibility - GitHub repository flipped from private to public

**Supersedes:** D003's visibility clause (deprecated 2026-04-22 03:04
UTC). The "hosted on GitHub under edhodapp account" aspect of D003 is
unchanged; only the private-to-public visibility change is superseded.

Repository `edhodapp/silicritter` flipped from private to public on
2026-04-22, once AGPL-3.0-or-later licensing was in place (D004).

**Rationale:** Ed, 2026-04-22: "Let's open up silicritter for fun."
With the license in place and the contribution policy explicit, there
is no longer a principled reason to keep the repo private - the
pre-release / discretion argument was contingent on patent optionality
(which D004 removed) and a sense of "not yet ready to show." Ed chose
to drop the latter.

**Consequences:**
- Commit history and content become publicly visible. Nothing in the
  current repo contains sensitive material (no credentials, no
  proprietary-sibling code, no commercial details).
- AGPL binds on any distribution to third parties, including downstream
  forks. Expected.
- No-contributions-accepted policy (D004) stands; pull requests will
  be declined regardless of public visibility.
- External readers can read, fork, and comply with AGPL terms; they
  cannot contribute upstream.

### D006: Sim-first phased build — full behavioral simulation before lowering

silicritter is built in four separable phases of increasing concreteness.
Architecture is finalized at each phase before moving to the next;
lowering-tool constraints from later phases do not casually alter
earlier-phase decisions.

1. **Idealized behavioral simulation** (JAX; current phase). Clean
   abstractions, full expressive power, float32 ideal numerics, no
   process noise. Validates system dynamics and architectural
   choices. The current step 1-5 ladder (LIF, slot pool, three-factor
   STDP, GA on the outer loop) lives here.
2. **Noisy behavioral simulation** (still JAX). Introduces process
   variation, transistor mismatch, capacitor leakage, quantized
   per-slot v. Validates architectural robustness against silicon
   reality before any circuit-level work. The inner plasticity loop
   and outer GA run over populations of simulated dies rather than a
   single ideal model; this is where the Thompson 1996 trap is
   pre-emptively disarmed.
3. **Circuit-level validation** (ngspice). Specific analog primitives
   (LIF cell, slot cell, valence broadcast line, charge-write
   circuitry, multi-neuromodulator bias lines) designed and proven to
   reproduce the primitive the behavioral sim assumed. Primitive-by-
   primitive, not the whole network.
4. **Layout and tape-out** (Magic + KLayout + DRC / LVS). Physical
   design informed by all three prior phases. Fab path per D004 /
   D005 posture: Efabless chipIgnite is the default; TinyTapeout is
   license-compatible and acceptable by fab feature match.

**Rationale:** Ed's methodological inheritance from a Verilog
instructor who was one of the early employees at Gateway Design
Automation, the company that created Verilog in 1984 as a simulation
language before synthesis existed. The teacher's principle: use the
simulation language's full expressive power to get the design right
before starting RTL. Starting in a restricted lowering subset means
the synthesis tool's constraints shape architecture rather than
design intent - you end up arguing about what you can build before
deciding what you want. Separating "what do we want?" from "how do
we build it?" is the actual win.

Iteration cost scales sharply per phase (JAX: hours per cycle;
ngspice: days; layout: weeks), so premature lowering spends the
budget in the wrong place. Grothendieck's "rising sea" (see global
CLAUDE.md on assembly philosophy and Lextrait reference): build
foundational abstractions until the architecture feels inevitable,
then the lowering writes itself. Naur's "programming is theory
building" argues the same from a different direction.

**Phase transitions:** declare architecture "frozen at Phase N"
before beginning Phase N+1 work. Each phase has entry criteria
(prior phase complete and validated) and exit criteria (architecture
stable at this level). Revisiting an earlier phase in response to a
lower-phase constraint is fine but must be deliberate, not
incidental.

**Not a waterfall:** within a phase, iteration is normal. The
phases are about which level of concreteness is currently
load-bearing, not gate-kept sequential development.

**Known concession:** some abstractions used in Phase 1 are known
to be silicon-hostile (perfectly exponential trace decay with a
fixed tau, for instance - analog leakage is process-variable).
These are flagged when introduced and revisited in Phase 2 rather
than shaping Phase 1.
