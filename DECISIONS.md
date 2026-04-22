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
