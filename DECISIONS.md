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

### D007: Inhibition substrate - canonical E/I values adopted provisionally

**DEPRECATED 2026-04-23 18:00 UTC — superseded by D008.** Step 13's
5-seed multi-seed perturbation sweep confirmed that `(i_fraction,
i_mult) = (0.2, 8.0)` outperforms the D007 canonical `(0.2, 4.0)` by
6.5% open-loop and 11.6% closed-loop, with inter-condition gaps
>10× the single-point std. The fraction stays at 0.2; only the
multiplier changes. See D008 for the successor.

silicritter adopts canonical cortical E/I balanced-network values
as the inhibitory substrate, added to `silicritter.slotpool` and
`silicritter.paired` with optional `pre_is_inhibitory` parameters:

- **E:I ratio = 4:1** (80 % excitatory, 20 % inhibitory neurons per
  population). Inhibitory neurons occupy the last 20 % of indices
  by default via `assign_ei_identity(n_neurons, 0.2)`.
- **I-weight multiplier = 4.0** (inhibitory-sourced contributions
  are negated and scaled 4x relative to excitatory-sourced). This
  matches the classical van Vreeswijk & Sompolinsky 1996-style
  balanced-network condition where 20 % of pres deliver 4x per-
  synapse current to balance the 80 % excitatory majority.

These values are literature-canonical (Vogels/Abbott 2005,
balanced-network foundations), not silicritter-specific
derivations. Step 9 validated that the substrate behaves as
predicted at these values (canonical balanced cross produces near-
zero mean input; E-targeted cross preserves step 7c fitness; I-
targeted cross produces silence).

**Provisional, not validated for silicritter's specific
architecture.** Literature canonical values were derived for
continuous-weight rate-coded cortical models. silicritter uses
discrete slot-pool representations, structural plasticity,
multi-modulator chemistry, and spike-based LIF dynamics. The
canonical values may need tuning for stable dynamics in this
specific substrate. We commit them as the starting prior rather
than sweeping them.

**Configurability preserved by design.** `synaptic_current`,
`step_paired`, and `simulate_paired` all take
`i_weight_multiplier` as a parameter with default 4.0. The E/I
identity bool array is passed explicitly, so the fraction can be
varied at the call site without touching the library. Future
perturbation experiments can sweep these values without refactor.

**What we owe.** A step in the Step 11 range (per the post-step-
9/10 plan) runs a systematic perturbation sweep across E:I ratio
and i_weight_multiplier, validating that our commitment was
correct or measuring where the real optimum sits for silicritter's
architecture. Punch list items to probe first:

1. i_weight_multiplier variation (primary axis - does 4.0 really
   balance at our v_max of 2.0?).
2. Inhibitory fraction (fewer / more I neurons).
3. Interaction with structural release (do I-slots release
   faster / slower than E-slots under the default rule?).
4. Interaction with task drive regime (strongly excited tasks
   may prefer different balance than weakly excited ones).

**Not yet adopted:** inhibitory-specific STDP rules (Vogels 2011).
STDP currently treats E and I slots identically. Anti-Hebbian
STDP on inhibitory synapses is the natural follow-up once closed-
loop adrenaline (Step 10) lands.

### D008: E/I canonical tightened to (i_fraction, i_mult) = (0.2, 8.0)

**Supersedes:** D007 (deprecated 2026-04-23 18:00 UTC). D007 adopted
`(0.2, 4.0)` provisionally with an explicit "may be following the
herd" caveat and a promise to validate by perturbation. Step 13
delivered the perturbation sweep and Step 13's multi-seed
follow-up confirmed a better point.

silicritter now uses `(inhibitory_fraction, i_weight_multiplier) =
(0.2, 8.0)` as the default E/I configuration. The inhibitory
fraction stays at D007's 0.2 (validated as the best row of the 5×5
grid); only the multiplier doubles.

**Evidence (step 13 multi-seed, n_seeds=5, stride=37, base_seed=0):**

| point           | condition    | mean fit   | std      | gap vs canonical |
|---|---|---:|---:|---:|
| (0.2, 4.0)      | open-loop    | −1.561e−4  | 8.5e−7   | (baseline)       |
| (0.2, 4.0)      | closed-loop  | −5.595e−5  | 0.0e+00  | (baseline)       |
| (0.2, 8.0)      | open-loop    | −1.460e−4  | 9.4e−7   | 6.5% better      |
| (0.2, 8.0)      | closed-loop  | −4.944e−5  | 5.9e−7   | 11.6% better     |

Inter-point gaps (1.01e−5 open-loop, 6.5e−6 closed-loop) exceed each
std by ~10×. Separation is robust.

**Mechanism note: closed-loop std shifts from zero to positive at
the new point.** At D007 canonical, closed-loop std was exactly
0.0 across 5 seeds — adrenaline was rail-limited at `ADR_MAX =
3.0`, so B's firing pattern became deterministic regardless of
seed. At D008's `(0.2, 8.0)`, closed-loop std is 5.9e−7 (small but
non-zero), meaning the controller is NOT saturated. Stronger
per-synapse inhibition compresses B's firing range enough that the
controller operates in the middle of its output band, which is why
the tracking residual is smaller. The fitness improvement is
*driven by* moving the operating point off the rail.

**Cliff awareness preserved.** Step 13's single-seed grid showed
`(0.4, 8.0)` collapsing to −1.06e−4 (same sub-threshold physics as
step 12's tonic cliff). D008 sits in the safe interior of that
grid — raising i_mult any further at this fraction still looks
safe, but raising i_fraction along with i_mult is dangerous.
Future tuning should perturb i_mult independently, not jointly
with i_fraction.

**Impact on code.** `I_WEIGHT_MULTIPLIER: float = 4.0` is hardcoded
at the top of experiments/step10, step11, step12, step13. Updating
these to 8.0 (or ideally reading from a shared default) is the next
mechanical change. Existing perf_history entries measured at 4.0
remain valid as historical data; re-runs at the D008 default would
give ~7–12% better numbers across the board but don't invalidate
the architectural findings.

**What's still owed.** Inhibitory-specific STDP (Vogels 2011
anti-Hebbian rule on I synapses) remains deferred per D007's
closing note. The Vogels rule may interact with D008's stronger
i_mult in ways neither the step 9 nor step 13 sweeps probed.

**Naming note.** The "canonical" literature value is still
`(0.2, 4.0)`. D008 is silicritter-specific tuning, not a claim
about biology. Expect this point to drift again as the substrate
evolves (recurrent growth, inhibitory STDP, learned plasticity
rates), and plan for further supersession.
